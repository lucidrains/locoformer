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

from locoformer.locoformer import Locoformer, check_has_param_attr, tensor_to_dict
from locoformer.replay_buffer import ReplayBuffer
from x_mlps_pytorch import Feedforwards, MLP

# humanoid observation config

HUMANOID_OBS_CONFIG = (
    ('height', 1),
    ('quat', 4),
    ('v_x', 1),
    ('v_y', 1),
    ('v_z', 1),
    ('rest', 340)
)

@check_has_param_attr('state', 'height')
@check_has_param_attr('state', 'v_x')
def reward_humanoid_walking(
    state,
    s_height = 1.0,
    s_vel = 1.25,
    height_nominal = 1.4,
    height_threshold = 0.4
):
    # maintaining nominal height
    height_error = (state.height - height_nominal).pow(2)
    height_reward = (-height_error / s_height).exp()

    # falling
    fall_penalty = (state.height < (height_nominal - height_threshold)).float() * -10.

    # forward velocity reward
    vel_reward = state.v_x * s_vel

    return height_reward + vel_reward + fall_penalty

def humanoid_reward_shaping(state):
    state_named = tensor_to_dict(state, HUMANOID_OBS_CONFIG)
    return reward_humanoid_walking(state_named)

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
    num_learning_cycles = 5_000,
    num_episodes_before_learn = 128,
    record_every_episode = 128,
    max_timesteps = 250,
    replay_buffer_size = 256,
    use_vision = False,
    embed_past_action = True,
    vision_height_width_dim = 64,
    clear_video = True,
    video_folder = 'recordings_animal',
    learning_rate = 3e-4,
    discount_factor = 0.99,
    betas = (0.9, 0.99),
    gae_lam = 0.95,
    ppo_eps_clip = 0.2,
    ppo_entropy_weight = .02,
    state_entropy_bonus_weight = .1,
    batch_size = 64,
    epochs = 3,
    reward_range = (-10000., 10000.),
    num_reward_bins = 25_000,
    cpu = False
):

    if clear_video:
        rmtree(video_folder, ignore_errors = True)

    # possible envs

    envs = [
        ('Humanoid-v5', 'humanoid', 348, 17, 0.001),
        ('HalfCheetah-v5', 'cheetah', 17, 6, 0.05),
    ]

    # replays

    replay_buffers = dict()

    # accelerate

    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device

    # model

    locoformer = Locoformer(
        embedder = dict(
            dim = 256,
            dim_state = [348, 17],
        ),
        unembedder = dict(
            dim = 256,
            num_continuous = 17 + 6,
            selectors = [
                list(range(17)),
                list(range(17, 17 + 6))
            ]
        ),
        state_pred_loss_weight = 0.005,
        state_pred_network = Feedforwards(dim = 256, depth = 2, final_norm = True),
        embed_past_action = embed_past_action,
        transformer = dict(
            dim = 256,
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
        ppo_soft_constrain_action_max = 1.,
        policy_network = Feedforwards(dim = 256, depth = 2, final_norm = True),
        value_network = Feedforwards(dim = 256, depth = 4, final_norm = True),
        dim_value_input = 256,
        num_reward_bins = num_reward_bins,
        reward_range = reward_range,
        hl_gauss_loss_kwargs = dict(),
        recurrent_cache = True,
        calc_gae_kwargs = dict(
            use_accelerated = False
        ),
        reward_shaping_fns = [
            [humanoid_reward_shaping],
            []
        ],
        use_spo = True,
        asymmetric_spo = False
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

        env_name, env_short_name, dim_state, num_actions, env_state_pred_loss_weight = envs[env_index]

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
            if not use_vision:
                return None

            return get_snapshot(env, dim_state_image_shape[1:])

        transforms = dict(
            state_image = get_snapshot_from_env,
        )

        wrapped_env_functions = locoformer.wrap_env_functions(
            env,
            env_output_transforms = transforms,
            state_transform = state_transform
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
                env_index = env_index,
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
            compute_state_pred_loss,
            env_state_pred_loss_weight
        )

        env.close()

# main

if __name__ == '__main__':
    Fire(main)
