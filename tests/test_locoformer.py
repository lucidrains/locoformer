import pytest
param = pytest.mark.parametrize

import torch
from x_mlps_pytorch import MLP

from einops import rearrange

def test_locoformer():
    from locoformer.locoformer import Locoformer
    from torch import nn
    
    model = Locoformer(
        embedder = nn.Embedding(256, 128),
        unembedder = nn.Linear(128, 256, bias = False),
        value_network = MLP(128, 32, 1),
        transformer = dict(
            dim = 128,
            depth = 1
        )
    )

    seq = torch.randint(0, 256, (3, 512))

    (logits, values), cache = model(seq, return_values = True)
    (logits, values), cache = model(seq, return_values = True, cache = cache)
    (logits, values), cache = model(seq, return_values = True, cache = cache)

    assert logits.shape == (3, 512, 256)

    stateful_forward = model.get_stateful_forward(256, return_values = True, inference_mode = True)

    for state in seq.unbind(dim = -1):
        state = rearrange(state, 'b -> b 1')

        logits, values = stateful_forward(state)
        assert logits.shape == (3, 1, 256)
