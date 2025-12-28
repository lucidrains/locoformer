import pytest
param = pytest.mark.parametrize

import torch
from torch import nn
from x_mlps_pytorch import MLP

from einops import rearrange

from locoformer.locoformer import Locoformer

@param('gru_layers', (False, True))
@param('recurrent_cache', (False, True))
@param('has_commands', (False, True))
@param('long_term_mem_layers', ((), (1, 2)))
def test_locoformer(
    gru_layers,
    recurrent_cache,
    has_commands,
    long_term_mem_layers
):
    
    model = Locoformer(
        embedder = nn.Embedding(256, 128),
        unembedder = nn.Linear(128, 256, bias = False),
        value_network = MLP(128, 64, 32),
        dim_value_input = 32,
        reward_range = (-100., 100.),
        recurrent_cache = recurrent_cache,
        transformer = dict(
            dim = 128,
            depth = 2,
            window_size = 512,
            gru_layers = gru_layers,
            dim_cond = 2 if has_commands else None,
            long_term_mem_layers = long_term_mem_layers
        )
    )

    seq = torch.randint(0, 256, (3, 512))

    commands = None
    if has_commands:
        commands = torch.randn(3, 512, 2)

    (logits, values), cache = model(seq, condition = commands, return_values = True)
    (logits, values), cache = model(seq, condition = commands, return_values = True, cache = cache)
    (logits, values), cache = model(seq, condition = commands, return_values = True, cache = cache)

    assert logits.shape == (3, 512, 256)

    stateful_forward = model.get_stateful_forward(has_batch_dim = True, has_time_dim = True, return_values = True, inference_mode = True)

    inference_command = torch.randn(1, 1, 2) if has_commands else None

    for state in seq.unbind(dim = -1):
        state = rearrange(state, 'b -> b 1')

        logits, values = stateful_forward(state, condition = inference_command)
        assert logits.shape == (3, 1, 256)

def test_replay():
    from locoformer.replay_buffer import ReplayBuffer

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
            lambda state, command: state[4:6].norm(dim = -1)
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
            return np.random.normal(size = (10,)), {}

        def step(self, *args, **kwargs):
            return np.random.normal(size = (10,)), 0., False, False, {}


    env = MockEnv()

    reset_fn, step_fn = model.wrap_env_functions(env)

    reset_fn()

    step_dict = step_fn(3)

    assert len(step_dict['shaped_rewards']) == 2

def test_tensor_to_dict():
    state = torch.randn(1, 3, 5)
    config = (('xyz', 3), 'vx', 'vy')

    from locoformer.locoformer import tensor_to_dict

    state_dict = tensor_to_dict(state, config)
    assert hasattr(state_dict, 'xyz') and state_dict.xyz.shape == (1, 3, 3)

def test_evo():

    model = Locoformer(
        embedder = nn.Embedding(256, 128),
        unembedder = nn.Linear(128, 256, bias = False),
        value_network = MLP(128, 64, 32),
        dim_value_input = 32,
        reward_range = (-100., 100.),
        transformer = dict(
            dim = 128,
            depth = 1,
            window_size = 512,
        )
    )

    model.evolve(lambda model: 1., num_generations = 1)

def test_unified_state():
    from torch.nn import Module, ModuleList
    from locoformer.locoformer import Locoformer

    class StateEmbed(Module):
        def __init__(self):
            super().__init__()
            self.embedders = ModuleList([
                nn.Embedding(256, 128),
                nn.Linear(2, 128)
            ])

        def forward(self, state, state_type):
            return self.embedders[state_type](state)

    model = Locoformer(
        embedder = StateEmbed(),
        unembedder = nn.Linear(128, 256, bias = False),
        value_network = MLP(128, 64, 32),
        dim_value_input = 32,
        reward_range = (-100., 100.),
        recurrent_cache = False,
        transformer = dict(
            dim = 128,
            depth = 1,
            window_size = 512,
        )
    )

    state1 = torch.randint(0, 256, (3, 512))
    state2 = torch.randn((3, 512, 2))

    logits, cache = model(state1, state_embed_kwargs = dict(state_type = 0))
    logits, cache = model(state2, state_embed_kwargs = dict(state_type = 1), cache = cache)
    logits, cache = model(state1, state_embed_kwargs = dict(state_type = 0), cache = cache)

def test_memory():
    from locoformer.locoformer import MemoryMLP

    memory = MemoryMLP(512)

    tokens = torch.randn(2, 32, 512)

    memories = None

    retrieved = memory(tokens, memories)

    tokens = tokens + retrieved

    memories = memory.store(tokens, memories)

    retrieved = memory(tokens, memories)

    tokens = tokens + retrieved

    memories = memory.store(tokens, memories)

    assert tokens.shape == (2, 32, 512)
