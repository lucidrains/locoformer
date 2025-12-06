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

from locoformer.locoformer import Locoformer, ReplayBuffer, calc_entropy
from x_mlps_pytorch import MLP

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gumbel_noise(t):
    return -log(-log(torch.rand_like(t)))

def gumbel_sample(logits, temperature = 1., eps = 1e-6):
    noise = gumbel_noise(logits)
    return ((logits / max(temperature, eps)) + noise).argmax(dim = -1)

def gaussian_sample(logits):
    mean, log_var = logits.unbind(dim = -1)
    return mean + (0.5 * log_var).exp() * torch.randn_like(mean)

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
    action_unembed_kwargs: dict,
    batch_size = 16,
    epochs = 2,
    use_vision = False,
    compute_state_pred_loss = False,
    continuous = False
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
                action_unembed_kwargs = action_unembed_kwargs,
                compute_state_pred_loss = compute_state_pred_loss,
                continuous = continuous,
                accelerator = accelerator
            )

            accelerator.print(f'actor: {actor_loss.item():.3f} | critic: {critic_loss.item():.3f}')

# main function

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 50_000,
    max_timesteps = 500,
    num_episodes_before_learn = 32,
    max_discrete_actions = 4,
    max_continuous_actions = 3,
    continuous = False,
    use_vision = False,
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

    # accelerate

    accelerator = Accelerator()
    device = accelerator.device

    # environment

    env = gym.make(env_name, continuous = continuous, render_mode = 'rgb_array')

    if clear_video:
        rmtree(video_folder, ignore_errors = True)

    env = gym.wrappers.RecordVideo(
        env = env,
        video_folder = video_folder,
        name_prefix = 'lunar-video',
        episode_trigger = lambda eps: divisible_by(eps, record_every_episode),
        disable_logger = True
    )

    if not continuous:
        num_discrete_actions = env.action_space.n
        assert num_discrete_actions <= max_discrete_actions
        discrete_action_types = torch.arange(num_discrete_actions)
    else:
        num_continuous_actions = env.action_space.shape[0]
        assert num_continuous_actions <= max_continuous_actions
        continuous_action_types = torch.arange(num_continuous_actions)

    dim_state = env.observation_space.shape[0]
    dim_state_image_shape = (3, vision_height_width_dim, vision_height_width_dim)

    # memory

    replay = ReplayBuffer(
        'replay',
        num_episodes,
        max_timesteps + 1, # one extra node for bootstrap node - not relevant for locoformer, but for completeness
        fields = dict(
            state       = ('float', dim_state),
            state_image = ('float', dim_state_image_shape),
            action      = 'int' if not continuous else ('float', num_continuous_actions),
            action_log_prob = ('float', 1) if not continuous else ('float', num_continuous_actions),
            reward      = 'float',
            value       = 'float',
            done        = 'bool',
            learnable   = 'bool',
            condition   = ('float', 2)
        )
    )

    class StateEmbedder(nn.Module):
        def __init__(
            self,
            dim,
            dim_state,
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

        def forward(
            self,
            state,
            state_type
        ):

            if state_type == 'image':
                return self.image_to_token(state)
            elif state_type == 'raw':
                return self.state_to_token(state)
            else:
                raise ValueError('invalid state type')

    class ActionUnembedder(Module):
        def __init__(
            self,
            dim,
            num_discrete_actions = 0,
            num_continuous_actions = 0,
        ):
            super().__init__()
            assert (num_discrete_actions > 0) ^ (num_continuous_actions > 0)

            self.num_discrete_actions = num_discrete_actions
            self.num_continuous_actions = num_continuous_actions

            self.discrete_action_unembeds = nn.Embedding(num_discrete_actions, dim)
            self.continuous_action_unembeds = nn.Embedding(num_continuous_actions, dim * 2)

            nn.init.zeros_(self.continuous_action_unembeds.weight)

            self.register_buffer('default_discrete_action_types', torch.arange(num_discrete_actions), persistent = False)
            self.register_buffer('default_continuous_action_types', torch.arange(num_continuous_actions), persistent = False)

            self.register_buffer('dummy', tensor(0), persistent = False)

        @property
        def is_continuous(self):
            return self.num_continuous_actions > 0

        @property
        def device(self):
            return self.dummy.device

        def sample_action(
            self,
            dist
        ):
            if continuous:
                action = gaussian_sample(dist)
            else:
                action = gumbel_sample(dist)

            return action

        def log_prob(
            self,
            dist,
            value
        ):
            if self.is_continuous:
                mean, log_var = action_logits.unbind(dim = -1)
                std = (0.5 * log_var).exp()
                dist = Normal(mean, std)
                log_prob = dist.log_prob(value)
            else:
                if value.ndim == (dist.ndim - 1):
                    value = rearrange(value, '... -> ... 1')

                log_prob = dist.log_softmax(dim = -1).gather(-1, value)

            return log_prob

        def entropy(
            self,
            dist
        ):
            if self.is_continuous:
                mean, log_var = action_logits.unbind(dim = -1)
                std = (0.5 * log_var).exp()
                dist = Normal(mean, std)

                entropy = dist.entropy()
            else:
                entropy = calc_entropy(action_logits)
                entropy = rearrange(entropy, '... -> ... 1')

            return entropy

        def forward(
            self,
            embed,
            discrete_action_types = None,
            continuous_action_types = None
        ):
            discrete_action_types = default(discrete_action_types, self.default_discrete_action_types)
            continuous_action_types = default(continuous_action_types, self.default_continuous_action_types)

            if not is_tensor(discrete_action_types):
                discrete_action_types = tensor(discrete_action_types)

            if not is_tensor(continuous_action_types):
                continuous_action_types = tensor(continuous_action_types)

            discrete_action_types = discrete_action_types.to(self.device)
            continuous_action_types = continuous_action_types.to(self.device)

            if self.is_continuous:
                action_unembed = self.continuous_action_unembeds(continuous_action_types)
                action_unembed = rearrange(action_unembed, '... (d two) -> ... d two', two = 2)
                dist = einsum(embed, action_unembed, '... d, n d mu_sigma -> ... n mu_sigma')
            else:
                action_unembed = self.discrete_action_unembeds(discrete_action_types)
                dist = einsum(embed, action_unembed, '... d, n d -> ... n')

            return dist

    # state embed kwargs

    if use_vision:
        state_embed_kwargs = dict(state_type = 'image')
        compute_state_pred_loss = False
    else:
        state_embed_kwargs = dict(state_type = 'raw')
        compute_state_pred_loss = True

    if continuous:
        action_unembed_kwargs = dict(continuous_action_types = continuous_action_types)
    else:
        action_unembed_kwargs = dict(discrete_action_types = discrete_action_types)

    # networks

    locoformer = Locoformer(
        embedder = StateEmbedder(64, dim_state),
        unembedder = ActionUnembedder(
            64,
            num_discrete_actions = max_discrete_actions if not continuous else 0,
            num_continuous_actions = max_continuous_actions if continuous else 0
        ),
        state_pred_head = MLP(64, dim_state * 2, bias = False),
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
        value_network = MLP(64, 64),
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

        stateful_forward = locoformer.get_stateful_forward(has_batch_dim = False, has_time_dim = False, inference_mode = True)

        with replay.one_episode():
            while True:

                rand_command = torch.randn(2)

                # predict next action

                state_for_model = state_image if use_vision else state

                (action_logits, state_pred), value = stateful_forward(state_for_model, state_embed_kwargs = state_embed_kwargs, action_unembed_kwargs = action_unembed_kwargs, condition = rand_command, return_values = True, return_state_pred = True)

                action = locoformer.unembedder.sample_action(action_logits)

                # pass to environment

                next_state, reward, terminated, truncated, *_ = env_step(action)

                next_state_image = get_snapshot(env, dim_state_image_shape[1:])

                # maybe state entropy bonus

                if state_entropy_bonus_weight > 0. and exists(state_pred):
                    _, log_var = state_pred.chunk(2, dim = -1)

                    state_entropy = (log_var * state_entropy_bonus_weight).sum()

                    reward = reward + state_entropy.item() # the entropy is directly related to log variance

                # append to memory

                exceeds_max_timesteps = timestep == (max_timesteps - 1)
                done = truncated or terminated or tensor(exceeds_max_timesteps)

                # get log prob of action

                action_log_prob = locoformer.unembedder.log_prob(action_logits, action)

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

                        _, next_value = stateful_forward(next_state_for_model, condition = rand_command, return_values = True, state_embed_kwargs = state_embed_kwargs, action_unembed_kwargs = action_unembed_kwargs)

                        memory = memory._replace(value = next_value, learnable = False)

                        replay.store(**memory._asdict())

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
                    action_unembed_kwargs,
                    batch_size,
                    epochs,
                    use_vision,
                    compute_state_pred_loss,
                    continuous
                )
# main

if __name__ == '__main__':
    Fire(main)
