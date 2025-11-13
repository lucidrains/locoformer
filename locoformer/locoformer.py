from __future__ import annotations
from functools import partial

import torch
from torch import nn, cat, stack, arange, is_tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, RMSNorm, Identity, Sequential
from torch.utils._pytree import tree_map

import einx
from einops import rearrange, einsum
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding

from assoc_scan import AssocScan

LinearNoBias = partial(Linear, bias = False)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(arr):
    return arr[0]

def divisible_by(num, den):
    return (num % den) == 0

def tree_map_tensor(x, fn):
    return tree_map(lambda t: t if not is_tensor(t) else fn(t), x)

def detach_all(x):
    return tree_map_tensor(x, lambda t: t.detach())

def combine_kv_cache(cache1, cache2):
    combined_cache = []

    for layer_cache1, layer_cache2 in zip(cache1, cache2):
        next_cache = cat((layer_cache1, layer_cache2), dim = -2)
        combined_cache.append(next_cache)

    return combined_cache

# generalized advantage estimate

@torch.no_grad()
def calc_gae(
    rewards,
    values,
    masks,
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None
):
    assert values.shape[-1] == rewards.shape[-1]
    use_accelerated = default(use_accelerated, rewards.is_cuda)

    values = F.pad(values, (0, 1), value = 0.)
    values, values_next = values[..., :-1], values[..., 1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    gae = scan(gates, delta)

    returns = gae + values

    return returns

# transformer-xl mask w/ flex attn

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

def create_xl_mask(
    seq_len,
    kv_seq_len,
    window_size,
    episode_ids = None,  # (b n) - in the case that within the same batch there are multiple episodes
    lookback_blocks = 1, # in transformer-xl, lookback is one window size block, but can be multiple for longer context
    device = None
):
    assert kv_seq_len >= seq_len
    assert window_size <= seq_len

    offset = kv_seq_len - seq_len

    def create_block_mask_fn(b, __, q, k):
        offset_q = q + offset
        block_q = offset_q // window_size
        block_k = k // window_size

        causal_mask = offset_q >= k

        # in transformer-xl, the previous segment is fully attended to - may just double the segments and make this sliding for ease of inference logic

        block_mask = (block_q >= block_k) & (block_q <= (block_k + lookback_blocks))

        mask = causal_mask & block_mask

        # handle intra-episodic attention if needed

        if exists(episode_ids):
            q_episode = episodes[b, q + offset]
            k_episode = episodes[b, k]

            intra_episode_mask = q_episode == k_episode
            mask = mask & intra_episode_mask

        return mask

    create_kwargs = dict(device = device) if exists(device) else dict()
    return create_block_mask(create_block_mask_fn, B = None, H = None, Q_LEN = seq_len, KV_LEN = kv_seq_len, _compile = True, **create_kwargs)

def create_sliding_mask(
    seq_len,
    kv_seq_len,
    window_size,
    device = None
):
    assert kv_seq_len >= seq_len
    offset = kv_seq_len - seq_len

    def sliding_mask(_, __, q, k):
        offset_q = q + offset
        distance = offset_q - k

        backward_sliding_mask = distance <= window_size
        forward_sliding_mask = distance >= 0

        return backward_sliding_mask & forward_sliding_mask

    create_kwargs = dict(device = device) if exists(device) else dict()
    return create_block_mask(sliding_mask, B = None, H = None, Q_LEN = seq_len, KV_LEN = kv_seq_len, _compile = True, **create_kwargs)

# transformer-xl with ppo

class Attention(Module):
    def __init__(
        self,
        dim,
        window_size,
        dim_head = 64,
        heads = 8,
        pre_rmsnorm = True,
        fixed_window_size = False,
        accept_value_residual = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        self.norm = RMSNorm(dim) if pre_rmsnorm else Identity()

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.rotary_embed = RotaryEmbedding(dim_head)

        dim_inner = dim_head * heads
        self.to_q = LinearNoBias(dim, dim_inner)
        self.to_kv = LinearNoBias(dim, dim_inner * 2)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.to_v_gates = Sequential(
            LinearNoBias(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        )

        # value residual

        self.accept_value_residual = accept_value_residual

        if accept_value_residual:
            self.to_value_residual_mix = Sequential(
                LinearNoBias(dim, heads),
                Rearrange('b n h -> b h n 1'),
                nn.Sigmoid()                
            )

        # fixed window size

        self.fixed_window_size = fixed_window_size
        self.window_size = window_size

    def forward(
        self,
        tokens,
        value_residual = None,
        kv_cache = None,
        return_kv_cache = False,
    ):
        seq_len = tokens.shape[-2]
        assert seq_len <= self.window_size

        device = tokens.device

        tokens = self.norm(tokens)

        q, k, v = (self.to_q(tokens), *self.to_kv(tokens).chunk(2, dim = -1))

        q, k, v = map(self.split_heads, (q, k, v))

        orig_v = v

        q = q * self.scale

        if exists(value_residual):
            assert self.accept_value_residual
            mix = self.to_value_residual_mix(tokens)
            v = v.lerp(value_residual, mix)

        if exists(kv_cache):
            ck, cv = kv_cache
            k = cat((ck, k), dim = -2)
            v = cat((cv, v), dim = -2)

        if return_kv_cache:
            next_kv_cache = stack((k, v))

        q, k = self.rotary_embed.rotate_queries_with_cached_keys(q, k)

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        i, j = sim.shape[-2:]

        if self.fixed_window_size:
            i_seq = arange(i, device = device)
            j_seq = arange(j, device = device) - (j - i)
            dist = einx.subtract('i, j -> i j', i_seq, j_seq)
            causal_mask = (dist < 0) | (dist > self.window_size)
        else:
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = sim.device).triu(j - i + 1)

        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = out * self.to_v_gates(tokens)

        out = self.merge_heads(out)

        out = self.to_out(out)

        if not return_kv_cache:
            return out

        return out, (next_kv_cache, orig_v)

class FeedForward(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 4.,
        pre_rmsnorm = True
    ):
        super().__init__()
        self.norm = RMSNorm(dim) if pre_rmsnorm else Identity()

        dim_inner = int(dim * expansion_factor * 2 / 3)

        self.proj_in = Linear(dim, dim_inner * 2)
        self.proj_out = Linear(dim_inner, dim)

    def forward(
        self,
        x
    ):
        x = self.norm(x)

        x, gates = self.proj_in(x).chunk(2, dim = -1)

        x = x * F.gelu(gates)

        return self.proj_out(x)

class TransformerXL(Module):
    def __init__(
        self,
        dim,
        depth,
        window_size,
        dim_head = 64,
        heads = 8,
        expansion_factor = 4.,
        final_norm = True,
        fixed_window_size = False,
    ):
        super().__init__()

        layers = ModuleList([])

        for i in range(depth):
            is_first = i == 0

            attn = Attention(dim = dim, dim_head = dim_head, heads = heads, fixed_window_size = fixed_window_size, window_size = window_size, accept_value_residual = not is_first)

            ff = FeedForward(dim = dim, expansion_factor = expansion_factor)

            layers.append(ModuleList([
                attn, ff
            ]))

        self.layers = layers
        self.norm = RMSNorm(dim) if final_norm else Identity()

        # fixed window size

        self.fixed_window_size = fixed_window_size
        self.window_size = window_size

    def forward(
        self,
        x,
        cache = None,
        return_kv_cache = False
    ):

        cache = default(cache, (None,) * len(self.layers))

        next_kv_caches = []
        value_residual = None

        for (attn, ff), kv_cache in zip(self.layers, cache):

            attn_out, (next_kv_cache, values) = attn(x, value_residual = value_residual, kv_cache = kv_cache, return_kv_cache = True)

            x = attn_out + x
            x = ff(x) + x

            next_kv_caches.append(next_kv_cache)
            value_residual = default(value_residual, values)

        embed = self.norm(x)

        if not return_kv_cache:
            return embed

        next_kv_cache = stack(next_kv_caches)

        next_kv_cache = next_kv_cache[..., -self.window_size:, :]

        return embed, next_kv_cache

# class

class Locoformer(Module):
    def __init__(
        self,
        embedder: Module,
        unembedder: Module,
        transformer: dict | TransformerXL,
        value_network: Module | None = None
    ):
        super().__init__()

        if isinstance(transformer, dict):
            transformer = TransformerXL(**transformer)

        self.transformer = transformer

        self.embedder = embedder
        self.unembedder = unembedder

        self.value_network = value_network

        self.fixed_window_size = transformer.fixed_window_size
        self.window_size = transformer.window_size

    @property
    def device(self):
        return next(self.parameters()).device

    def get_stateful_forward(
        self,
        initial_states: Tensor | None = None,
        inference_mode = False,
        has_batch_dim = False,
        **kwargs
    ):
        window_size = self.window_size

        cache = None

        def stateful_forward(state: Tensor, **override_kwargs):
            nonlocal cache

            # handle no batch, for easier time rolling out against envs

            if not has_batch_dim:
                state = rearrange(state, '... -> 1 ...')

            # forwards

            out, cache = self.forward(state, cache = cache, **{**kwargs, **override_kwargs})

            # handle cache

            cache_len = cache.shape[-2]

            if self.fixed_window_size or divisible_by(cache_len, window_size * 2):
                cache = cache[..., -window_size:, :]

            # maybe remove batch

            if not has_batch_dim:
                out = tree_map_tensor(out, lambda t: rearrange(t, '1 ... -> ...'))

            return out

        if inference_mode:
            stateful_forward = torch.inference_mode()(stateful_forward)

        # handle prompt

        if not exists(initial_states):
            return stateful_forward

        initial_logits = []

        for state_segments in initial_states.split(self.window_size, dim = -1):

            logits = stateful_forward(state_segments, return_values = False)
            initial_logits.append(logits)

        initial_logits = cat(initial_logits, dim = -2)

        return stateful_forward, initial_logits

    def forward(
        self,
        state: Tensor,
        cache: Tensor | None = None,
        detach_cache = False,
        return_values = False
    ):

        tokens = self.embedder(state)

        embed, kv_cache = self.transformer(tokens, cache = cache, return_kv_cache = True)

        # unembed to actions - in language models this would be the next state

        action_logits = self.unembedder(embed)

        out = action_logits

        # maybe detach cache

        if detach_cache:
            kv_cache = detach_all(kv_cache)

        # handle returning of values

        if return_values:
            assert exists(self.value_network)

            values = self.value_network(embed)

            if values.ndim == 3:
                assert values.shape[-1] == 1
                values = rearrange(values, '... 1 -> ...')

            out = (out, values)

        # output and cache

        return out, kv_cache
