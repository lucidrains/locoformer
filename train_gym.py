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

from types import SimpleNamespace

from accelerate import Accelerator

import gymnasium as gym

import torch
from torch import nn
from torch.nn import Module
from torch import from_numpy, randint, tensor, is_tensor, stack, arange
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.distributions import Normal

from einops import rearrange, einsum
from einops.layers.torch import Rearrange, Reduce

from locoformer.locoformer import (
    Locoformer,
    ReplayBuffer
)

from discrete_continuous_embed_readout import Embed

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

# learn

def learn(
    model,
    optims,
    accelerator,
    replay,
    state_embed_kwargs: dict,
    action_select_kwargs: dict,
    batch_size = 16,
    epochs = 2,
    use_vision = False,
    compute_state_pred_loss = False
):
    state_field = 'state_image' if use_vision else 'state'

    dl = replay.dataloader(
        batch_size = batch_size,
        shuffle = True
    )

    model, dl, *optims = accelerator.prepare(model, dl, *optims)

    for _ in range(epochs):
        for data in dl:

            data = SimpleNamespace(**data)

            actor_loss, critic_loss = model.ppo(
                state = getattr(data, state_field),
                action = data.action,
                old_action_log_prob = data.action_log_prob,
                reward = data.reward,
                old_value = data.value,
                mask = data.learnable,
                condition = data.condition,
                episode_lens = data._lens,
                optims = optims,
                state_embed_kwargs = state_embed_kwargs,
                action_select_kwargs = action_select_kwargs,
                compute_state_pred_loss = compute_state_pred_loss,
                accelerator = accelerator
            )

            accelerator.print(f'actor: {actor_loss.item():.3f} | critic: {critic_loss.item():.3f}')

# main function

def main(
    env_index = 0,
    num_episodes_before_learn = 64,
    num_episodes = 50_000,
    max_timesteps = 500,
    replay_buffer_size = 5_000,
    use_vision = False,
    embed_past_action = False,
    vision_height_width_dim = 64,
    clear_video = False,
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
    reward_range = (-300., 300.)
):

    # possible envs

    envs = [
        ('LunarLander-v3', False),
        ('LunarLander-v3', True),
        ('CartPole-v1', False),
    ]

    env_name, continuous = envs[env_index]

    # accelerate

    accelerator = Accelerator()
    device = accelerator.device

    # environment

    env_kwargs = dict()
    if continuous:
        env_kwargs = dict(continuous = continuous)

    env = gym.make(env_name, render_mode = 'rgb_array', **env_kwargs)

    if clear_video:
        rmtree(video_folder, ignore_errors = True)

    env = gym.wrappers.RecordVideo(
        env = env,
        video_folder = video_folder,
        name_prefix = 'lunar-video',
        episode_trigger = lambda eps: divisible_by(eps, record_every_episode),
        disable_logger = True
    )

    dim_state = env.observation_space.shape[0]
    dim_state_image_shape = (3, vision_height_width_dim, vision_height_width_dim)
    num_actions = env.action_space.n if not continuous else env.action_space.shape[0]

    # memory

    replay = ReplayBuffer(
        'replay',
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
            learnable   = 'bool',
            condition   = ('float', 2)
        ),
        meta_fields = dict(
            cum_rewards = 'float'
        )
    )

    class StateEmbedder(nn.Module):
        def __init__(
            self,
            dim,
            dim_state,
            num_internal_state = None,
            internal_state_selectors = None
        ):
            super().__init__()
            dim_hidden = dim * 2

            self.image_to_token = nn.Sequential(
                Rearrange('b t c h w -> b c t h w'),
                nn.Conv3d(3, dim_hidden, (1, 7, 7), padding = (0, 3, 3)),
                nn.ReLU(),
                nn.Conv3d(dim_hidden, dim_hidden, (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1)),
                nn.ReLU(),
                nn.Conv3d(dim_hidden, dim_hidden, (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1)),
                Reduce('b c t h w -> b t c', 'mean'),
                nn.Linear(dim_hidden, dim)
            )

            self.state_to_token = MLP(dim_state, 64, bias = False)

            # internal state embeds for each robot

            self.internal_state_embedder = None

            if exists(num_internal_state) and exists(internal_state_selectors):
                self.internal_state_embedder = Embed(
                    dim_state,
                    num_continuous = num_internal_state,
                    internal_state_selectors = internal_state_selectors
                )

        def forward(
            self,
            state,
            state_type,
            internal_state = None,
            internal_state_selector_id: int | None = None
        ):

            if state_type == 'image':
                token_embeds = self.image_to_token(state)
            elif state_type == 'raw':
                token_embeds = self.state_to_token(state)
            else:
                raise ValueError('invalid state type')

            if exists(internal_state_selector_id):
                internal_state_embed = self.internal_state_embedder(internal_state, selector_id = internal_state_selector_id)

                token_embeds = token_embeds + internal_state_embed

            return token_embeds

    # state embed kwargs

    if use_vision:
        state_embed_kwargs = dict(state_type = 'image')
        compute_state_pred_loss = False
    else:
        state_embed_kwargs = dict(state_type = 'raw')
        compute_state_pred_loss = True

    # networks

    action_select_kwargs = dict(selector_index = env_index)

    locoformer = Locoformer(
        embedder = StateEmbedder(64, dim_state),
        unembedder = dict(
            dim = 64,
            num_discrete = 6,
            num_continuous = 3,
            selectors = [
                [[0, 1, 2, 3]],  # lunar lander discrete
                [0, 1],          # lunar lander continuous
                [[4, 5]],        # cart pole discrete
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
        use_spo = True,
        value_network = Feedforwards(dim = 64, depth = 1),
        dim_value_input = 64,
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

    # able to wrap the env for all values to torch tensors and back
    # all environments should follow usual MDP interface, domain randomization should be given at instantiation

    env_reset, env_step = locoformer.wrap_env_functions(env)

    # loop

    for episodes_index in tqdm(range(num_episodes)):

        state, *_ = env_reset()

        state_image = get_snapshot(env, dim_state_image_shape[1:])

        timestep = 0

        stateful_forward = locoformer.get_stateful_forward(
            has_batch_dim = False,
            has_time_dim = False,
            inference_mode = True
        )

        cum_rewards = 0.

        with replay.one_episode() as final_meta_data_store_dict:
            while True:

                rand_command = torch.randn(2)

                # predict next action

                state_for_model = state_image if use_vision else state

                (action_logits, state_pred), value = stateful_forward(state_for_model, state_embed_kwargs = state_embed_kwargs, action_select_kwargs = action_select_kwargs, condition = rand_command, return_values = True, return_state_pred = True)

                action = locoformer.unembedder.sample(action_logits, **action_select_kwargs)

                # pass to environment

                next_state, reward, terminated, truncated, *_ = env_step(action)

                next_state_image = get_snapshot(env, dim_state_image_shape[1:])

                # maybe state entropy bonus

                if state_entropy_bonus_weight > 0. and exists(state_pred):

                    entropy = locoformer.state_pred_head.entropy(state_pred)

                    state_entropy_bonus = (entropy * state_entropy_bonus_weight).sum()

                    reward = reward + state_entropy_bonus.item() # the entropy is directly related to log variance

                # cum rewards

                cum_rewards += reward

                # append to memory

                exceeds_max_timesteps = timestep == (max_timesteps - 1)
                done = truncated or terminated or tensor(exceeds_max_timesteps)

                # get log prob of action

                action_log_prob = locoformer.unembedder.log_prob(action_logits, action, **action_select_kwargs)

                memory = replay.store(
                    state = state,
                    state_image = state_image,
                    action = action,
                    action_log_prob = action_log_prob,
                    reward = reward,
                    value = value,
                    done = done,
                    learnable = tensor(True),
                    condition = rand_command
                )

                # increment counters

                timestep += 1

                # break if done or exceed max timestep

                if done:

                    # handle bootstrap value, which is a non-learnable timestep added with the next value for GAE
                    # only if terminated signal not detected

                    if not terminated:
                        next_state_for_model = next_state_image if use_vision else next_state

                        _, next_value = stateful_forward(next_state_for_model, condition = rand_command, return_values = True, state_embed_kwargs = state_embed_kwargs, action_select_kwargs = action_select_kwargs)

                        memory = memory._replace(
                            state = next_state,
                            state_image = next_state_image,
                            value = next_value,
                            reward = next_value,
                            learnable = tensor(False)
                        )

                        replay.store(**memory._asdict())

                    # store the final cumulative reward into meta data

                    final_meta_data_store_dict.update(cum_rewards = cum_rewards)

                    break

                state = next_state
                state_image = next_state_image

            # learn if hit the number of learn timesteps

            if divisible_by(episodes_index + 1, num_episodes_before_learn):

                learn(
                    locoformer,
                    optims,
                    accelerator,
                    replay,
                    state_embed_kwargs,
                    action_select_kwargs,
                    batch_size,
                    epochs,
                    use_vision,
                    compute_state_pred_loss
                )
# main

if __name__ == '__main__':
    Fire(main)
