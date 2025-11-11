import pytest
param = pytest.mark.parametrize

import torch

def test_locoformer():
    from locoformer.locoformer import Locoformer
    from torch import nn
    
    model = Locoformer(
        embedder = nn.Embedding(256, 128),
        unembedder = nn.Linear(128, 256, bias = False),
        transformer = dict(
            dim = 128,
            depth = 1
        )
    )

    seq = torch.randint(0, 256, (2, 512))

    logits, cache = model(seq)
    logits, cache = model(seq, cache = cache)
    logits, cache = model(seq, cache = cache)

    assert logits.shape == (2, 512, 256)
