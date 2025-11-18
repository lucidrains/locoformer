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
from collections import deque
from types import SimpleNamespace

from accelerate import Accelerator

import gymnasium as gym

import torch
from torch import from_numpy, randint, tensor, stack, arange
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

from einops import rearrange

from locoformer.locoformer import Locoformer, ReplayBuffer
from x_mlps_pytorch import MLP

# helper functions

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gumbel_noise(t):
    return -log(-log(torch.rand_like(t)))

def gumbel_sample(logits, temperature = 1., eps = 1e-6):
    noise = gumbel_noise(logits)
    return ((logits / max(temperature, eps)) + noise).argmax(dim = -1)

# learn

def learn(
    model,
    actor_optim,
    critic_optim,
    accelerator,
    replay,
    batch_size = 16,
    epochs = 2,
):
    dl = replay.dataloader(batch_size = batch_size, shuffle = True)
    model, dl, actor_optim, critic_optim = accelerator.prepare(model, dl, actor_optim, critic_optim)

    for _ in range(epochs):
        for data in dl:

            data = SimpleNamespace(**data)

            actor_loss, critic_loss = model.ppo(
                state = data.state,
                action = data.action,
                old_action_log_prob = data.action_log_prob,
                reward = data.reward,
                old_value = data.value,
                mask = data.learnable,
                episode_lens = data._lens,
                actor_optim = actor_optim,
                critic_optim = critic_optim
            )

            accelerator.print(f'actor: {actor_loss.item():.3f} | critic: {critic_loss.item():.3f}')

# main function

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 50_000,
    max_timesteps = 500,
    num_episodes_before_learn = 32,
    clear_video = True,
    video_folder = 'recordings',
    record_every_episode = 250,
    learning_rate = 8e-4,
    discount_factor = 0.99,
    betas = (0.9, 0.99),
    gae_lam = 0.95,
    ppo_eps_clip = 0.2,
    ppo_entropy_weight = .01,
    batch_size = 16,
    epochs = 2
):

    # accelerate

    accelerator = Accelerator()
    device = accelerator.device

    # environment

    env = gym.make(env_name, render_mode = 'rgb_array')

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
    num_actions = env.action_space.n

    # memory

    replay = ReplayBuffer(
        'replay',
        num_episodes,
        max_timesteps + 1, # one extra node for bootstrap node - not relevant for locoformer, but for completeness
        fields = dict(
            state = ('float', (dim_state,)),
            action = 'int',
            action_log_prob = 'float',
            reward = 'float',
            value = 'float',
            done = 'bool',
            learnable = 'bool'
        )
    )

    # networks

    locoformer = Locoformer(
        embedder = MLP(dim_state, 64, bias = False),
        unembedder = MLP(64, num_actions, bias = False),
        value_network = MLP(64, 1, bias = False),
        transformer = dict(
            dim = 64,
            dim_head = 32,
            heads = 4,
            depth = 4,
            window_size = 16
        ),
        discount_factor = discount_factor,
        gae_lam = gae_lam,
        ppo_eps_clip = ppo_eps_clip,
        ppo_entropy_weight = ppo_entropy_weight,
        calc_gae_kwargs = dict(
            use_accelerated = False
        )
    ).to(device)

    optim_actor = Adam([*locoformer.transformer.parameters(), *locoformer.actor_parameters()], lr = learning_rate, betas = betas)
    optim_critic = Adam([*locoformer.transformer.parameters(), *locoformer.critic_parameters()], lr = learning_rate, betas = betas)

    # able to wrap the env for all values to torch tensors and back
    # all environments should follow usual MDP interface, domain randomization should be given at instantiation

    env_reset, env_step = locoformer.wrap_env_functions(env)

    # loop

    for episodes_index in tqdm(range(num_episodes)):

        state, *_ = env_reset()

        timestep = 0

        stateful_forward = locoformer.get_stateful_forward(has_batch_dim = False, has_time_dim = False, inference_mode = True)

        with replay.one_episode():
            while True:

                # predict next action

                action_logits, value = stateful_forward(state, return_values = True)

                action = gumbel_sample(action_logits)

                # pass to environment

                next_state, reward, truncated, terminated, *_ = env_step(action)

                # append to memory

                exceeds_max_timesteps = timestep == (max_timesteps - 1)
                done = truncated or terminated or tensor(exceeds_max_timesteps)

                # get log prob of action

                action_log_prob = action_logits.gather(-1, rearrange(action, '-> 1'))
                action_log_prob = rearrange(action_log_prob, '1 ->')

                memory = replay.store(
                    state = state,
                    action = action,
                    action_log_prob = action_log_prob,
                    reward = reward,
                    value = value,
                    done = done,
                    learnable = tensor(True)
                )

                # increment counters

                timestep += 1

                # break if done or exceed max timestep

                if done:

                    # handle bootstrap value, which is a non-learnable timestep added with the next value for GAE
                    # only if terminated signal not detected

                    if not terminated:
                        _, next_value = stateful_forward(next_state, return_values = True)

                        memory._replace(value = next_value, learnable = False)

                        replay.store(**memory._asdict())

                    break

                state = next_state

            # learn if hit the number of learn timesteps

            if divisible_by(episodes_index + 1, num_episodes_before_learn):

                learn(
                    locoformer,
                    optim_actor,
                    optim_critic,
                    accelerator,
                    replay,
                    batch_size,
                    epochs,
                )
# main

if __name__ == '__main__':
    Fire(main)
