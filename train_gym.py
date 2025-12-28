# /// script
# dependencies = [
#     "accelerate",
#     "fire",
#     "gymnasium[box2d]>=1.0.0",
#     "locoformer>=0.0.12",
#     "moviepy",
#     "tqdm"
# ]
# ///

from fire import Fire
from shutil import rmtree
from tqdm import tqdm

from accelerate import Accelerator

import gymnasium as gym

import torch
from torch import nn
from torch.nn import Module
from torch import from_numpy, randint, tensor, is_tensor, stack, arange
import torch.nn.functional as F
from torch.optim import Adam

from einops import rearrange, einsum

from locoformer.locoformer import Locoformer

from locoformer.replay_buffer import ReplayBuffer

from x_mlps_pytorch import Feedforwards, MLP

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# get rgb snapshot from env

def get_snapshot(env, shape):
    vision_state = from_numpy(env.render())
    vision_state = rearrange(vision_state, 'h w c -> 1 c h w')
    reshaped = F.interpolate(vision_state, shape, mode = 'bilinear')
    return rearrange(reshaped / 255., '1 c h w -> c h w')

# a contrived inter-trial construction
# it randomly selects pairs of episodes, and then order them with increasing cumulative rewards
# allow researchers to construct it however they wish

def create_episode_mapping_from_replay(buffer):
    episodes = torch.arange(buffer.num_episodes)
    cum_rewards = torch.from_numpy(buffer.meta_memmaps['cum_rewards'][:buffer.num_episodes])

    # pair the episode previous with the next episode, just for testing

    two_episodes = torch.stack((episodes, torch.roll(episodes, 1, dims = (0,))), dim = -1)
    two_cum_rewards  = torch.stack((cum_rewards, torch.roll(cum_rewards, 1, dims = (0,))), dim = -1)

    # now we sort and rearrange the episode indices

    sorted_cum_rewards = two_cum_rewards.sort(dim = -1).indices

    paired_episodes_sorted_by_cum_reward = two_episodes.gather(-1, sorted_cum_rewards)

    return paired_episodes_sorted_by_cum_reward

# main function

def main(
    fixed_env_index = -1,
    num_learning_cycles = 1_000,
    num_episodes_before_learn = 32,
    alternate_env_every = 1,
    max_timesteps = 500,
    replay_buffer_size = 128,
    use_vision = False,
    embed_past_action = False,
    vision_height_width_dim = 64,
    clear_folders = True,
    video_folder = 'recordings',
    record_every_episode = 250,
    learning_rate = 8e-4,
    discount_factor = 0.99,
    betas = (0.9, 0.99),
    gae_lam = 0.95,
    ppo_eps_clip = 0.2,
    ppo_entropy_weight = .01,
    state_entropy_bonus_weight = .05,
    batch_size = 16,
    epochs = 3,
    reward_range = (-300., 300.),
    test_episode_mapping_constructor = False,
    test_mock_internal_states = False,
    test_mock_reward_shaping = False
):

    if clear_folders:
        rmtree(video_folder, ignore_errors = True)
        rmtree('./replay', ignore_errors = True)

    # possible envs

    envs = [
        ('CartPole-v1', False, 1),
        ('LunarLander-v3', False, 0),
        ('LunarLander-v3', True, 0),
    ]

    # accelerate

    accelerator = Accelerator()
    device = accelerator.device

    # testing reward shaping, and storing of the values

    reward_shaping_fns = None

    if test_mock_reward_shaping:
        reward_shaping_fns = [
            [lambda state: state.norm()],
            [lambda state: state.norm(), lambda state: state[2:3].abs()],
            []
        ]

    # model

    locoformer = Locoformer(
        reward_shaping_fns = reward_shaping_fns,
        embedder = dict(
            dim = 64,
            dim_state = [8, 4], # 8 for lunar lander, 4 for cartpole
            num_internal_states = 4,
            internal_states_selectors = [
                [0, 1],
                [2, 3],
                [0, 3]
            ]
        ),
        unembedder = dict(
            dim = 64,
            num_discrete = 6,
            num_continuous = 3,
            selectors = [
                [[4, 5]],        # cart pole discrete
                [[0, 1, 2, 3]],  # lunar lander discrete
                [0, 1],          # lunar lander continuous
            ]
        ),
        state_pred_network = Feedforwards(dim = 64, depth = 1),
        embed_past_action = embed_past_action,
        transformer = dict(
            dim = 64,
            dim_head = 32,
            heads = 4,
            depth = 4,
            window_size = 16,
            dim_cond = 2,
            gru_layers = True
        ),
        discount_factor = discount_factor,
        gae_lam = gae_lam,
        ppo_eps_clip = ppo_eps_clip,
        ppo_entropy_weight = ppo_entropy_weight,
        policy_network = Feedforwards(dim = 64, depth = 1),
        value_network = Feedforwards(dim = 64, depth = 2),
        dim_value_input = 64,
        reward_range = reward_range,
        hl_gauss_loss_kwargs = dict(),
        recurrent_cache = True,
        calc_gae_kwargs = dict(
            use_accelerated = False
        ),
        use_spo = False,
        asymmetric_spo = False
    ).to(device)

    optim_base = Adam(locoformer.transformer.parameters(), lr = learning_rate, betas = betas)
    optim_actor = Adam(locoformer.actor_parameters(), lr = learning_rate, betas = betas)
    optim_critic = Adam(locoformer.critic_parameters(), lr = learning_rate, betas = betas)

    optims = [optim_base, optim_actor, optim_critic]

    # replay buffers

    replay_buffers = dict()

    # loop

    pbar = tqdm(range(num_learning_cycles), desc = 'learning cycles')

    for learn_cycle in pbar:

        if fixed_env_index > -1:
            assert 0 <= fixed_env_index <= len(envs)
            env_index = fixed_env_index
        else:
            env_index = (learn_cycle // alternate_env_every) % len(envs)

        # environment

        env_name, continuous, state_id = envs[env_index]

        pbar.set_description(f'environment: {env_name} {"continuous" if continuous else "discrete"}')

        env_kwargs = dict()
        if continuous:
            env_kwargs = dict(continuous = continuous)

        env = gym.make(env_name, render_mode = 'rgb_array', **env_kwargs)

        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = video_folder,
            name_prefix = f'{learn_cycle}-env-video',
            episode_trigger = lambda eps: divisible_by(eps, record_every_episode),
            disable_logger = True
        )

        dim_state = env.observation_space.shape[0]
        dim_state_image_shape = (3, vision_height_width_dim, vision_height_width_dim)
        num_actions = env.action_space.n if not continuous else env.action_space.shape[0]

        # memory

        replay = replay_buffers.get(env_index, None)

        if not exists(replay):
            replay = ReplayBuffer(
                f'replay/env_{env_index}',
                replay_buffer_size,
                max_timesteps + 1, # one extra node for bootstrap node - not relevant for locoformer, but for completeness
                fields = dict(
                    state       = ('float', dim_state),
                    state_image = ('float', dim_state_image_shape),
                    action      = ('int', 1) if not continuous else ('float', num_actions),
                    action_log_prob = ('float', 1 if not continuous else num_actions),
                    reward      = 'float',
                    value       = 'float',
                    done        = 'bool',
                    condition   = ('float', 2),
                    cond_mask   = 'bool',
                    internal_state = ('float', 2)
                ),
                meta_fields = dict(
                    cum_rewards = 'float'
                )
            )

            replay_buffers[env_index] = replay

        # state embed kwargs

        if use_vision:
            state_embed_kwargs = dict(state_type = 'image')
        else:
            state_embed_kwargs = dict(state_type = 'raw')

        compute_state_pred_loss = True

        # specific embodiment state id and action selector id

        state_id_kwarg = dict(state_id = state_id)

        action_select_kwargs = dict(selector_index = env_index)

        if test_mock_internal_states:
            state_id_kwarg.update(internal_state_selector_id = state_id)

        # able to wrap the env for all values to torch tensors and back
        # all environments should follow usual MDP interface, domain randomization should be given at instantiation

        def get_snapshot_from_env(step_output, env):
            return get_snapshot(env, dim_state_image_shape[1:])

        transforms = dict(state_image = get_snapshot_from_env)

        # for testing of embedding shared internal states of robots across bodies

        def derive_internal_state(step_output, env):
            state = step_output['state']
            return state[:2]

        transforms.update(internal_state = derive_internal_state)

        wrapped_env_functions = locoformer.wrap_env_functions(
            env,
            env_output_transforms = transforms
        )

        for _ in tqdm(range(num_episodes_before_learn), leave = False):

            cum_reward = locoformer.gather_experience_from_env_(
                wrapped_env_functions,
                replay,
                max_timesteps = max_timesteps,
                use_vision = use_vision,
                action_select_kwargs = action_select_kwargs,
                env_index = env_index,
                state_embed_kwargs = state_embed_kwargs,
                state_id_kwarg = state_id_kwarg,
                state_entropy_bonus_weight = state_entropy_bonus_weight,
                embed_past_action = embed_past_action
            )

            pbar.set_postfix(reward = f'{cum_reward:.2f}')

        # learn if hit the number of learn timesteps

        locoformer.learn(
            optims,
            accelerator,
            replay,
            state_embed_kwargs,
            action_select_kwargs,
            state_id_kwarg,
            batch_size,
            epochs,
            use_vision,
            compute_state_pred_loss,
            create_episode_mapping_from_replay if test_episode_mapping_constructor else None
        )

# main

if __name__ == '__main__':
    Fire(main)
