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
