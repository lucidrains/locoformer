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
    fixed_env_index = -1,
    num_learning_cycles = 1_000,
    num_episodes_before_learn = 32,
    max_timesteps = 250,
    replay_buffer_size = 96,
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
    epochs = 4,
    reward_range = (-100., 100.),
):

    if clear_video:
        rmtree(video_folder, ignore_errors = True)

    # possible envs

    envs = [
        ('Humanoid-v5', 'humanoid', 348, 17, 100., (-0.4, 0.4)),
        ('HalfCheetah-v5', 'cheetah', 17, 6, 100., None),
    ]

    # replays

    replay_buffers = dict()

    # accelerate

    accelerator = Accelerator()
    device = accelerator.device

    # model

    locoformer = Locoformer(
        embedder = dict(
            dim = 128,
            dim_state = [348, 17],
        ),
        unembedder = dict(
            dim = 128,
            num_continuous = 17 + 6,
            continuous_squashed = True,
            selectors = [
                list(range(17)),
                list(range(17, 17 + 6))
            ],
            readout_kwargs = dict(
                continuous_softclamp_logvar = 15.,
            ),
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
        num_reward_bins = 2500,
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

    # loop

    pbar = tqdm(range(num_learning_cycles), desc = 'learning cycles')

    for learn_cycle in pbar:

        if fixed_env_index > -1:
            env_index = fixed_env_index
        else:
            env_index = learn_cycle % len(envs)

        env_name, env_short_name, dim_state, num_actions, reward_norm, rescale_range = envs[env_index]

        pbar.set_description(f'environment: {env_name}')

        env = gym.make(env_name, render_mode = 'rgb_array')

        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = video_folder,
            name_prefix = f'{learn_cycle}-{env_short_name}',
            episode_trigger = lambda eps: divisible_by(eps, record_every_episode),
            disable_logger = True
        )

        # memory

        dim_state_image_shape = (3, vision_height_width_dim, vision_height_width_dim)

        replay = replay_buffers.get(env_index, None)

        if not exists(replay):
            replay = ReplayBuffer(
                f'replay_{env_short_name}',
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
                    condition   = ('float', 2)
                ),
                meta_fields = dict(
                    cum_rewards = 'float'
                )
            )

            replay_buffers[env_index] = replay

        # state embed kwargs

        if use_vision:
            state_embed_kwargs = dict(state_type = 'image')
            compute_state_pred_loss = False
        else:
            state_embed_kwargs = dict(state_type = 'raw')
            compute_state_pred_loss = True

        state_id_kwarg = dict(state_id = env_index)
        action_select_kwargs = dict(selector_index = env_index)

        # transforms for replay buffer

        def state_transform(state):
            return state.float()

        def get_snapshot_from_env(step_output, env):
            return get_snapshot(env, dim_state_image_shape[1:])

        transforms = dict(
            state_image = get_snapshot_from_env,
        )

        wrapped_env_functions = locoformer.wrap_env_functions(
            env,
            env_output_transforms = transforms,
            state_transform = state_transform,
            reward_norm = reward_norm
        )

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
                embed_past_action = embed_past_action,
                action_rescale_range = rescale_range
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
