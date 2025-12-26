# /// script
# dependencies = [
#     "accelerate",
#     "fire",
#     "gymnasium[mujoco]>=1.0.0",
#     "locoformer>=0.0.12",
#     "moviepy",
#     "tqdm",
#     "x-mlps-pytorch"
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
    vision_state = from_numpy(env.render().copy())
    vision_state = rearrange(vision_state, 'h w c -> 1 c h w')
    reshaped = F.interpolate(vision_state, shape, mode = 'bilinear')
    return rearrange(reshaped / 255., '1 c h w -> c h w')

# main function

def main(
    num_learning_cycles = 1_000,
    num_episodes_before_learn = 32,
    max_timesteps = 1000,
    replay_buffer_size = 48,
    use_vision = False,
    embed_past_action = True,
    vision_height_width_dim = 64,
    clear_video = True,
    video_folder = 'recordings_humanoid',
    record_every_episode = 32,
    learning_rate = 3e-4,
    discount_factor = 0.99,
    betas = (0.9, 0.99),
    gae_lam = 0.95,
    ppo_eps_clip = 0.2,
    ppo_entropy_weight = .01,
    state_entropy_bonus_weight = .01,
    batch_size = 16,
    epochs = 5,
    reward_range = (-10000., 10000.),
):

    if clear_video:
        rmtree(video_folder, ignore_errors = True)

    # environment configuration

    env_name = 'Humanoid-v5'
    
    # accelerate

    accelerator = Accelerator()
    device = accelerator.device

    # initialize environment once to get dimensions

    env = gym.make(env_name, render_mode = 'rgb_array')
    dim_state = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    env.close()

    # model

    locoformer = Locoformer(
        embedder = dict(
            dim = 128,
            dim_state = [dim_state],
        ),
        unembedder = dict(
            dim = 128,
            num_continuous = num_actions,
            selectors = [
                list(range(num_actions))
            ]
        ),
        state_pred_network = Feedforwards(dim = 128, depth = 1),
        embed_past_action = embed_past_action,
        transformer = dict(
            dim = 128,
            dim_head = 32,
            heads = 8,
            depth = 6,
            window_size = 32,
            dim_cond = 2,
            gru_layers = True
        ),
        discount_factor = discount_factor,
        gae_lam = gae_lam,
        ppo_eps_clip = ppo_eps_clip,
        ppo_entropy_weight = ppo_entropy_weight,
        use_spo = True,
        value_network = Feedforwards(dim = 128, depth = 1),
        dim_value_input = 128,
        reward_range = reward_range,
        hl_gauss_loss_kwargs = dict(),
        recurrent_cache = True,
        calc_gae_kwargs = dict(
            use_accelerated = False
        ),
        asymmetric_spo = True
    ).to(device)

    optim_base = Adam(locoformer.transformer.parameters(), lr = learning_rate, betas = betas)
    optim_actor = Adam(locoformer.actor_parameters(), lr = learning_rate, betas = betas)
    optim_critic = Adam(locoformer.critic_parameters(), lr = learning_rate, betas = betas)

    optims = [optim_base, optim_actor, optim_critic]

    # memory

    dim_state_image_shape = (3, vision_height_width_dim, vision_height_width_dim)

    replay = ReplayBuffer(
        'replay_humanoid',
        replay_buffer_size,
        max_timesteps + 1,
        fields = dict(
            state       = ('float', dim_state),
            state_image = ('float', dim_state_image_shape),
            action      = ('float', num_actions),
            action_log_prob = ('float', num_actions),
            reward      = 'float',
            value       = 'float',
            done        = 'bool',
            internal_state = ('float', 2),
            condition   = ('float', 2)
        ),
        meta_fields = dict(
            cum_rewards = 'float'
        )
    )

    # loop

    pbar = tqdm(range(num_learning_cycles), desc = f'environment: {env_name}')

    env = gym.make(env_name, render_mode = 'rgb_array')

    env = gym.wrappers.RecordVideo(
        env = env,
        video_folder = video_folder,
        name_prefix = 'humanoid-video',
        episode_trigger = lambda eps: divisible_by(eps, record_every_episode),
        disable_logger = True
    )

    # state embed kwargs

    if use_vision:
        state_embed_kwargs = dict(state_type = 'image')
        compute_state_pred_loss = False
    else:
        state_embed_kwargs = dict(state_type = 'raw')
        compute_state_pred_loss = True

    state_id_kwarg = dict(state_id = 0)
    action_select_kwargs = dict(selector_index = 0)

    # transforms for replay buffer

    def get_snapshot_from_env(step_output, env):
        return get_snapshot(env, dim_state_image_shape[1:])

    def derive_internal_state(step_output, env):
        return torch.zeros(2)

    transforms = dict(
        state_image = get_snapshot_from_env,
        internal_state = derive_internal_state
    )

    class Float32ObservationWrapper(gym.ObservationWrapper):
        def observation(self, obs):
            return obs.astype('float32')

    env = Float32ObservationWrapper(env)

    wrapped_env_functions = locoformer.wrap_env_functions(
        env,
        env_output_transforms = transforms
    )

    for learn_cycle in pbar:

        for _ in range(num_episodes_before_learn):

            cum_reward = locoformer.gather_experience_from_env_(
                wrapped_env_functions,
                replay,
                max_timesteps = max_timesteps,
                use_vision = use_vision,
                action_select_kwargs = action_select_kwargs,
                state_embed_kwargs = state_embed_kwargs,
                state_id_kwarg = state_id_kwarg,
                state_entropy_bonus_weight = state_entropy_bonus_weight,
                embed_past_action = embed_past_action
            )

            pbar.set_postfix(reward = f'{cum_reward:.2f}')

        # learn

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
            compute_state_pred_loss
        )

    env.close()

# main

if __name__ == '__main__':
    Fire(main)
