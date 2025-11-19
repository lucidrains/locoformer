import pytest
param = pytest.mark.parametrize

import torch
from torch import nn
from x_mlps_pytorch import MLP

from einops import rearrange

from locoformer.locoformer import Locoformer

@param('recurrent_kv_cache', (False, True))
def test_locoformer(
    recurrent_kv_cache
):
    
    model = Locoformer(
        embedder = nn.Embedding(256, 128),
        unembedder = nn.Linear(128, 256, bias = False),
        value_network = MLP(128, 64, 32),
        dim_value_input = 32,
        reward_range = (-100., 100.),
        recurrent_kv_cache = recurrent_kv_cache,
        transformer = dict(
            dim = 128,
            depth = 1,
            window_size = 512
        )
    )

    seq = torch.randint(0, 256, (3, 512))

    (logits, values), cache = model(seq, return_values = True)
    (logits, values), cache = model(seq, return_values = True, cache = cache)
    (logits, values), cache = model(seq, return_values = True, cache = cache)

    assert logits.shape == (3, 512, 256)

    stateful_forward = model.get_stateful_forward(has_batch_dim = True, has_time_dim = True, return_values = True, inference_mode = True)

    for state in seq.unbind(dim = -1):
        state = rearrange(state, 'b -> b 1')

        logits, values = stateful_forward(state)
        assert logits.shape == (3, 1, 256)

def test_replay():
    from locoformer.locoformer import ReplayBuffer

    replay_buffer = ReplayBuffer(
        './replay_data',
        max_episodes = 10_000,
        max_timesteps = 501,
        fields = dict(
            state = ('float', (8,)),
            action = 'int',
            action_log_prob = 'float',
            reward = 'float',
            value = 'float',
            done = 'bool'
        )
    )

    lens = [3, 5, 4]

    for episode_len in lens:
        with replay_buffer.one_episode():
            for _ in range(episode_len):
                state = torch.randn((8,))
                action = torch.randint(0, 4, ())
                log_prob = torch.randn(())
                reward = torch.randn(())
                value = torch.randn(())
                done = torch.randint(0, 2, ()).bool()

                replay_buffer.store(
                    state = state,
                    action = action,
                    action_log_prob = log_prob,
                    reward = reward,
                    value = value,
                    done = done
                )

    dataset = replay_buffer.dataset()

    assert len(dataset) == 3

    assert torch.is_tensor(dataset[0]['state'])

    dataloader = replay_buffer.dataloader(batch_size = 3)

    assert next(iter(dataloader))['state'].shape[0] == 3

    # we will now consider consecutive pairs of episodes as 2 trials to be used for in-context adaptation
    # but realistically there will be a function that converts a given ReplayBuffer -> Int[batch, episode_indices]

    from torch import stack, arange

    episode_indices = arange(len(replay_buffer))
    remapped_episodes = stack((episode_indices[:-1], episode_indices[1:]))

    dataloader = replay_buffer.dataloader(
        batch_size = 1,
        episode_mapping = remapped_episodes
    )

    assert next(iter(dataloader))['_lens'][0] == (3 + 5) # first and second episodes are concatted together timewise

def test_reward_shaping():

    model = Locoformer(
        embedder = nn.Embedding(256, 128),
        unembedder = nn.Linear(128, 256, bias = False),
        value_network = MLP(128, 64, 32),
        dim_value_input = 32,
        reward_range = (-100., 100.),
        reward_shaping_fns = [
            lambda state: (state[3] - 2.5).pow(2).mean(),
            lambda state: state[4:6].norm(dim = -1)
        ],
        transformer = dict(
            dim = 128,
            depth = 1,
            window_size = 512
        )
    )

    import numpy as np

    class MockEnv:
        def reset(self):
            return np.random.normal(size = (10,))

        def step(self, *args, **kwargs):
            return np.random.normal(size = (10,))


    env = MockEnv()

    reset_fn, step_fn = model.wrap_env_functions(env)

    reset_fn()

    _, rewards = step_fn(3)

    assert len(rewards) == 2
