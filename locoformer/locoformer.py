from __future__ import annotations
import math
from typing import Callable, Any
from types import SimpleNamespace
from functools import partial, wraps

from pathlib import Path
from contextlib import contextmanager
from collections import namedtuple, deque

from glom import glom

from inspect import signature, Parameter

import numpy as np
from numpy import ndarray
from numpy.lib.format import open_memmap

from beartype import beartype
from beartype.door import is_bearable

import torch
from torch import nn, cat, stack, arange, Tensor, tensor, is_tensor, from_numpy, nested
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, RMSNorm, Identity, Sequential
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
from torch.optim import Optimizer

import einx
from einops import rearrange, repeat, einsum, reduce, pack
from einops.layers.torch import Rearrange, Reduce

from rotary_embedding_torch import RotaryEmbedding

from hl_gauss_pytorch import HLGaussLoss

from assoc_scan import AssocScan

from x_mlps_pytorch import MLP

from x_evolution import EvoStrategy

from discrete_continuous_embed_readout import EmbedAndReadout, Embed, Readout

from hyper_connections import mc_get_init_and_expand_reduce_stream_functions

from memmap_replay_buffer import ReplayBuffer, ReplayDataset

from torch_einops_utils import (
    pad_at_dim,
    lens_to_mask
)

# constants

LinearNoBias = partial(Linear, bias = False)

TransformerMemory = namedtuple('TransformerMemory', (
    'total_tokens',
    'kv_cache',
    'gru_cache',
    'mem_mlp_cache',
    'mem_mlp_hidden_states',
    'memory_segments',
    'episode_id_segments',
    'curr_episode_ids',
    'passed_episode_ids'
))

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def always(val):
    def inner(*args, **kwargs):
        return val

    return inner

def identity(t, *args, **kwargs):
    return t

def pick(data, keys):
    return tuple(data[k] for k in keys)

def first(arr):
    return arr[0]

def xnor(x, y):
    return not (x ^ y)

def divisible_by(num, den):
    return (num % den) == 0

def get_param_names(fn):
    parameters = signature(fn).parameters
    return list(parameters.keys())

def check_has_param_attr(
    param_name,
    param_attr,
    default_value = None
):
    def decorator(fn):
        sig = signature(fn)

        @wraps(fn)
        def inner(*args, **kwargs):

            bound_args = sig.bind(*args, **kwargs).arguments

            if not (
                param_name in bound_args and
                exists(bound_args[param_name]) and
                hasattr(bound_args[param_name], param_attr)
            ):
                return default_value

            return fn(*args, **kwargs)

        return inner
    return decorator

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

def is_empty(t):
    return t.numel() == 0

def tree_map_tensor(x, fn):
    return tree_map(lambda t: t if not is_tensor(t) else fn(t), x)

def safe_cat(t, next_t, dim = -1):
    if not exists(t):
        return next_t

    return cat((t, next_t), dim = dim)

def normalize(t, mask = None, eps = 1e-5):
    if exists(mask):
        assert mask.any()

    t_for_stats = t[mask] if exists(mask) else t
    var, mean = torch.var_mean(t_for_stats)

    return (t - mean) / var.sqrt().clamp_min(eps)

def tensor_to_dict(
    t: Tensor,
    config: tuple[tuple[str, int] | str],
    dim = -1,
    return_dottable = True
):
    config = tuple((c, 1) if isinstance(c, str) else c for c in config)

    names, sizes = zip(*config)
    assert sum(sizes) == t.shape[dim]

    t = t.split(sizes, dim = dim)
    tensor_dict = dict(zip(names, t))

    if not return_dottable:
        return tensor_dict

    return SimpleNamespace(**tensor_dict)

# dataset related

class RemappedReplayDataset(Dataset):
    def __init__(
        self,
        dataset: ReplayDataset,
        episode_mapping: Tensor | list[list[int]],
        shuffle_episodes = False,
        num_trials_select = None
    ):
        assert len(dataset) > 0
        self.dataset = dataset

        if is_tensor(episode_mapping):
            assert episode_mapping.dtype in (torch.int, torch.long) and episode_mapping.ndim == 2
            episode_mapping = episode_mapping.tolist()

        self.episode_mapping = episode_mapping
        self.shuffle_episodes = shuffle_episodes

        assert not (exists(num_trials_select) and num_trials_select <= 0)
        self.sub_select_trials = exists(num_trials_select)
        self.num_trials_select = num_trials_select

    def __len__(self):
        return len(self.episode_mapping)

    def __getitem__(self, idx):

        episode_indices = self.episode_mapping[idx]

        episode_indices = tensor(episode_indices)
        episode_indices = episode_indices[(episode_indices >= 0) & (episode_indices < len(self.dataset))]

        assert not is_empty(episode_indices)

        # shuffle the episode indices if either shuffle episodes is turned on, or `num_trial_select` passed in (for sub selecting episodes from a set)

        if (
            episode_indices.numel() > 1 and
            (self.shuffle_episodes or self.sub_select_trials)
        ):
            num_episodes = len(episode_indices)
            episode_indices = episode_indices[torch.randperm(num_episodes)]

        # crop out the episodes

        if self.sub_select_trials:
            episode_indices = episode_indices[:self.num_trials_select]

        # now select out the episode data and merge along time

        episode_data = [self.dataset[i] for i in episode_indices.tolist()]

        episode_lens = stack([data.pop('_lens') for data in episode_data])

        keys = first(episode_data).keys()

        values = [list(data.values()) for data in episode_data]

        values = [cat(field_values) for field_values in zip(*values)] # concat across time

        multi_episode_data = dict(zip(keys, values))

        multi_episode_data['_lens'] = episode_lens.sum()

        multi_episode_data['_episode_indices'] = cat([torch.full((episode_len,), episode_index) for episode_len, episode_index in zip(episode_lens, episode_indices)])

        return multi_episode_data

# reward functions - A.2

@check_has_param_attr('state_named', 'v_xy', default_value = 0.)
@check_has_param_attr('command', 'v_xy', default_value = 0.)
def reward_linear_velocity_command_tracking(
    state_named,
    command,
    s1 = 1.
):
    error = (state_named.v_xy - command.v_xy).norm(dim = -1).pow(2)
    return torch.exp(-error / s1)

@check_has_param_attr('state_named', 'w_z', default_value = 0.)
@check_has_param_attr('command', 'w_z', default_value = 0.)
def reward_angular_velocity_command_tracking(
    state_named,
    command,
    s2 = 1.
):
    error = (state_named.w_z - command.w_z).norm(dim = -1).pow(2)
    return torch.exp(-error / s2)

@check_has_param_attr('state_named', 'v_z', default_value = 0.)
def reward_base_linear_velocity_penalty(
    state_named
):
    return -state_named.v_z.norm(dim = -1).pow(2)

@check_has_param_attr('state_named', 'w_xy', default_value = 0.)
def reward_base_angular_velocity_penalty(
    state_named
):
    return -state_named.w_xy.norm(dim = -1).pow(2)

@check_has_param_attr('state_named', 'x_z', default_value = 0.)
def reward_base_height_penalty(
    state_named,
    x_z_nominal = 0.27
):
    return -(state_named.x_z - x_z_nominal).norm(dim = -1).pow(2)

@check_has_param_attr('state_named', 'joint_v', default_value = 0.)
def reward_joint_velocity_penalty(
    state_named
):
    return -state_named.joint_v.norm(dim = -1).pow(2)

def reward_torque_penalty(
    state_named = None,
    action = None
):
    # flexible torque penalty that can use either the explicit torques from state
    # or fallback to using the action as a proxy for effort/torque

    tau = None
    if exists(state_named) and hasattr(state_named, 'tau'):
        tau = state_named.tau
    elif exists(action):
        tau = action
    
    if not exists(tau):
        return 0.

    return -tau.norm(dim = -1).pow(2)

def reward_alive(
    state_named
):
    return 1.

# generalized advantage estimate

@torch.no_grad()
def calc_gae(
    rewards,
    values,
    masks = None,
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None
):
    assert values.shape[-1] == rewards.shape[-1]
    use_accelerated = default(use_accelerated, rewards.is_cuda)

    values = F.pad(values, (0, 1), value = 0.)
    values, values_next = values[..., :-1], values[..., 1:]

    if not exists(masks):
        masks = torch.ones_like(values)

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    gae = scan(gates, delta)

    returns = gae + values

    return gae, returns

# transformer-xl mask w/ flex attn

flex_attention = None
create_block_mask = None

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
            q_episode = episode_ids[b, q + offset]
            k_episode = episode_ids[b, k]

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

# normalization + conditioning (needed for the commands to the robot)

class MaybeAdaRMSNormWrapper(Module):
    def __init__(
        self,
        fn: Module,
        dim,
        dim_cond = None
    ):
        super().__init__()
        condition = exists(dim_cond)

        self.fn = fn
        self.norm = nn.RMSNorm(dim, elementwise_affine = not condition)

        self.accept_condition = condition

        if condition:
            self.to_gamma = LinearNoBias(dim_cond, dim)
            self.to_ada_norm_zero = nn.Linear(dim_cond, dim)

            nn.init.zeros_(self.to_gamma.weight)
            nn.init.zeros_(self.to_ada_norm_zero.weight)
            nn.init.constant_(self.to_ada_norm_zero.bias, -5.)

    def forward(
        self,
        x,
        *args,
        cond = None,
        cond_mask = None,
        **kwargs
    ):

        need_cond = self.accept_condition
        has_input_cond = need_cond and exists(cond)

        if exists(cond):
            assert self.accept_condition

        prenormed = self.norm(x)

        if has_input_cond:
            if cond.ndim == 2:
                cond = rearrange(cond, 'b d -> b 1 d')

            cond_scale = self.to_gamma(cond)

            conditioned = prenormed * cond_scale

            # handle a condition mask

            if exists(cond_mask):
                prenormed = einx.where('b n, b n d, b n d', cond_mask, conditioned, prenormed)
            else:
                prenormed = conditioned

        # the main block, either attention or feedforward or whatever

        all_fn_out = self.fn(prenormed, *args, **kwargs)

        if not has_input_cond:
            return all_fn_out

        # function may return multiple args

        (out, *rest), tree_spec = tree_flatten(all_fn_out)

        scale_out = self.to_ada_norm_zero(cond).sigmoid()

        if exists(cond_mask):
            is_cond = rearrange(cond_mask, '... -> ... 1')
            out = torch.where(is_cond, out * scale_out, out)
        else:
            out = out * scale_out

        # restore

        all_fn_out = tree_unflatten((out, *rest), tree_spec)

        return all_fn_out

# transformer-xl with ppo

class Attention(Module):
    def __init__(
        self,
        dim,
        window_size,
        dim_head = 64,
        heads = 8,
        fixed_window_size = False,
        accept_value_residual = False,
        max_mem_segments = 1
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

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

        assert max_mem_segments >= 1
        self.max_mem_segments = max_mem_segments

        self.fixed_window_size = fixed_window_size or max_mem_segments > 1
        self.window_size = window_size

        self.register_buffer('causal_mask', None, persistent = False)

    def forward(
        self,
        tokens,
        value_residual = None,
        kv_cache = None,
        past_segments = None,
        return_kv_cache = False,
        episode_ids: Tensor | None = None,
        past_episode_ids: Tensor | None = None
    ):
        seq_len = tokens.shape[-2]

        device = tokens.device

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

        if exists(past_segments):
            pk, pv = past_segments
            k = cat((pk, k), dim = -2)
            v = cat((pv, v), dim = -2)

        if exists(past_episode_ids) and exists(episode_ids):
            all_episode_ids = cat((past_episode_ids, episode_ids), dim = -1)
        else:
            all_episode_ids = episode_ids

        q, k = self.rotary_embed.rotate_queries_with_cached_keys(q, k)

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        i, j = sim.shape[-2:]

        if exists(flex_attention) and tokens.is_cuda:
            mask = create_xl_mask(i, j, self.window_size, episode_ids = all_episode_ids, lookback_blocks = self.max_mem_segments, device = device)
            out = flex_attention(q, k, v, block_mask = mask)
        else:
            if self.fixed_window_size:
                i_seq = arange(i, device = device)
                j_seq = arange(j, device = device) - (j - i)
                dist = einx.subtract('i, j -> i j', i_seq, j_seq)
                causal_mask = (dist < 0) | (dist > (self.max_mem_segments * self.window_size))
            else:
                if not exists(self.causal_mask) or self.causal_mask.shape != (i, j):
                    self.causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)

                causal_mask = self.causal_mask

            # handle intra-episodic attention if needed

            if exists(all_episode_ids):
                q_episode_ids = episode_ids
                k_episode_ids = all_episode_ids

                intra_episode_mask = einx.not_equal('b i, b j -> b i j', q_episode_ids, k_episode_ids)
                intra_episode_mask = rearrange(intra_episode_mask, 'b i j -> b 1 i j')
                causal_mask = causal_mask | intra_episode_mask

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
    @beartype
    def __init__(
        self,
        dim,
        depth,
        window_size,
        dim_head = 64,
        heads = 8,
        expansion_factor = 4.,
        dim_cond = None,
        final_norm = True,
        fixed_window_size = False,
        gru_layers = False,
        long_term_mem_layers: tuple[int, ...] = (),
        mem_kwargs: dict = dict(),
        max_mem_segments = 1,
        num_residual_streams = 1,
        mhc_sinkhorn_iters = 4
    ):
        super().__init__()
        self.dim = dim

        # memory

        long_term_mem_layers = set(long_term_mem_layers)

        assert all([1 <= l <= depth for l in long_term_mem_layers])

        self.long_term_mem_layers = long_term_mem_layers
        self.num_mem_mlps = len(long_term_mem_layers)
        self.has_mem = self.num_mem_mlps > 0
        self.max_mem_segments = max_mem_segments

        # hyper connections

        init_hyper_conn, self.expand_streams, self.reduce_streams = mc_get_init_and_expand_reduce_stream_functions(num_residual_streams, sinkhorn_iters = mhc_sinkhorn_iters)

        # condition

        condition = exists(dim_cond)

        self.to_cond_tokens = MLP(dim_cond, dim * 2, activate_last = True) if exists(dim_cond) else None

        norm_fn = partial(MaybeAdaRMSNormWrapper, dim = dim, dim_cond = (dim * 2) if condition else None) 

        # layers

        layers = ModuleList([])

        for i in range(depth):
            layer = i + 1
            is_first = layer == 1
            has_mem = layer in long_term_mem_layers

            gru = norm_fn(nn.GRU(dim, dim, batch_first = True)) if gru_layers else None

            mem = MemoryMLP(dim, **mem_kwargs) if has_mem else None

            attn = norm_fn(Attention(dim = dim, dim_head = dim_head, heads = heads, fixed_window_size = fixed_window_size, window_size = window_size, accept_value_residual = not is_first, max_mem_segments = max_mem_segments))

            ff = norm_fn(FeedForward(dim = dim, expansion_factor = expansion_factor))

            # maybe hyper connections

            mem_store_hc = init_hyper_conn(dim = dim, add_branch_out_to_residual = False)

            if num_residual_streams > 1:

                attn = init_hyper_conn(dim = dim, branch = attn)

                ff = init_hyper_conn(dim = dim, branch = ff)

                if gru_layers:
                    gru = init_hyper_conn(dim = dim, branch = gru)

                if has_mem:
                    mem = init_hyper_conn(dim = dim, branch = mem, forward_method_names = ('store',))

            layers.append(ModuleList([
                gru, mem, mem_store_hc, attn, ff
            ]))

        self.layers = layers
        self.norm = RMSNorm(dim) if final_norm else Identity()

        self.gru_layers = gru_layers

        # fixed window size

        self.fixed_window_size = fixed_window_size
        self.window_size = window_size

    def forward(
        self,
        x,
        cache: TransformerMemory | None = None,
        return_kv_cache = False,
        condition: Tensor | None = None,
        cond_mask: Tensor | None = None,
        episode_ids: Tensor | None = None
    ):
        curr_token_seq_len = x.shape[-2]

        # cache and residuals

        num_layers = len(self.layers)

        # extract variables from cache

        is_first_window = True
        total_tokens = 0
        kv_cache = gru_cache = mem_mlp_cache = mem_mlp_hidden_states = memory_segments = None

        if exists(cache):
            total_tokens, kv_cache, gru_cache, mem_mlp_cache, mem_mlp_hidden_states, memory_segments, episode_id_segments, curr_episode_ids, passed_episode_ids = cache
            is_first_window = total_tokens < self.window_size
        else:
            memory_segments = deque(maxlen = self.max_mem_segments)
            episode_id_segments = deque(maxlen = self.max_mem_segments)
            curr_episode_ids = None
            passed_episode_ids = None

        # handle memory segments

        past_segments = None
        past_episode_ids = None

        if len(memory_segments) > 0:
            past_segments = stack(list(memory_segments), dim = 0)
            past_segments = rearrange(past_segments, 'l depth kv b h n d -> depth kv b h (l n) d')

            episode_id_segments_list = [e for e in episode_id_segments if exists(e)]
            if len(episode_id_segments_list) > 0:
                past_episode_ids = cat(episode_id_segments_list, dim = -1)

        if exists(curr_episode_ids):
            if exists(past_episode_ids):
                past_episode_ids = cat((past_episode_ids, curr_episode_ids), dim = -1)
            else:
                past_episode_ids = curr_episode_ids

        kv_cache = default(kv_cache, (None,) * num_layers)
        gru_cache = default(gru_cache, (None,) * num_layers)
        mem_mlp_cache = default(mem_mlp_cache, (None,) * num_layers)
        mem_mlp_hidden_states = default(mem_mlp_hidden_states, (None,) * num_layers)

        # prepare next cache

        next_kv_caches = []
        next_gru_hiddens = [] if self.gru_layers else None
        next_mem_mlp_cache = [] if self.has_mem else None
        next_mem_mlp_hidden_states = [] if self.has_mem else None
        next_total_tokens = total_tokens + curr_token_seq_len

        is_window_boundary = divisible_by(next_total_tokens, self.window_size)

        value_residual = None

        # handle condition

        cond_tokens = None

        if exists(condition):
            assert exists(self.to_cond_tokens)
            cond_tokens = self.to_cond_tokens(condition)

        cond_kwargs = dict(cond = cond_tokens, cond_mask = cond_mask)

        # hc expand

        x = self.expand_streams(x)

        # layers

        for layer_index, ((maybe_gru, maybe_mem, maybe_mem_store_hc, attn, ff), layer_gru_cache, layer_mem_mlp, layer_kv_cache, layer_hidden_states) in enumerate(zip(self.layers, gru_cache, mem_mlp_cache, kv_cache, mem_mlp_hidden_states)):

            # handle maybe rnn

            if exists(maybe_gru):
                x, gru_hiddens = maybe_gru(x, layer_gru_cache, **cond_kwargs)

                next_gru_hiddens.append(gru_hiddens)

            # maybe handle retrieving

            is_mem_layer = exists(maybe_mem)

            if (
                not is_first_window and
                is_mem_layer
            ):
                x = maybe_mem(x, layer_mem_mlp)

            # attention

            layer_past_segments = None
            if exists(past_segments):
                layer_past_segments = past_segments[layer_index]

            x, (next_kv_cache, values) = attn(x, **cond_kwargs, value_residual = value_residual, kv_cache = layer_kv_cache, past_segments = layer_past_segments, return_kv_cache = True, episode_ids = episode_ids, past_episode_ids = past_episode_ids)

            # handle storing of memory

            if self.has_mem:
                next_mem_mlp = layer_mem_mlp
                next_layer_hidden_states = layer_hidden_states

                if is_mem_layer:
                    # accumulate hidden states
                    next_layer_hidden_states = safe_cat(layer_hidden_states, x, dim = -2)

                    if is_window_boundary:
                        mem_store_input, _ = maybe_mem_store_hc(next_layer_hidden_states)

                        next_mem_mlp = maybe_mem.store(mem_store_input, layer_mem_mlp)
                        next_layer_hidden_states = None

                next_mem_mlp_cache.append(next_mem_mlp)
                next_mem_mlp_hidden_states.append(next_layer_hidden_states)

            # feedforward

            x = ff(x, **cond_kwargs)

            next_kv_caches.append(next_kv_cache)
            value_residual = default(value_residual, values)

        # hc reduce

        x = self.reduce_streams(x)

        # norm

        embed = self.norm(x)

        next_kv_cache = stack(next_kv_caches)

        if exists(next_gru_hiddens):
            next_gru_hiddens = stack(next_gru_hiddens)

        next_cache = TransformerMemory(next_total_tokens, next_kv_cache, next_gru_hiddens, next_mem_mlp_cache, next_mem_mlp_hidden_states, memory_segments, episode_id_segments, curr_episode_ids, passed_episode_ids)

        return embed, next_cache

# simple 2 layer memory mlp
# following ttt/titans

from torch.func import functional_call, grad, vmap

class MemoryMLP(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 4.
    ):
        super().__init__()

        dim_hidden = int(dim * expansion_factor)

        self.norm = nn.RMSNorm(dim)

        # queries, keys, values

        self.to_queries = Linear(dim, dim, bias = False)

        self.to_key_values = nn.Sequential(
            Linear(dim, dim * 2, bias = False),
            nn.SiLU()
        )

        # memory mlp

        self.mlp = MLP(dim, dim_hidden, dim, activation = nn.SiLU())

        # initial params

        self.init_mlp_params = dict(self.mlp.named_parameters())

        # grad for storing

        def retrieve_fn(params, queries: Tensor):
            return functional_call(self.mlp, params, queries)

        def loss_fn(params, inputs: tuple[Tensor, Tensor, Tensor]):
            keys, values, learning_rate = inputs
            pred = functional_call(self.mlp, params, keys)
            loss = F.mse_loss(pred, values, reduction = 'none')
            loss = loss * learning_rate
            return loss.mean()

        self.grad_fn = vmap(grad(loss_fn), in_dims = (0, (0, 0, 0)))

        self.retrieve_fn = vmap(retrieve_fn, in_dims = (0, 0))

        # forgetting

        self.to_forget_gate = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, 1, bias = False),
            Rearrange('b 1 -> b'),
            nn.Sigmoid()
        )

        # loss weight / learning rate

        self.to_loss_weight = nn.Linear(dim, 1, bias = False)

    def get_init_mlp_params(
        self,
        batch_size
    ):
        return {name: repeat(params, '... -> b ...', b = batch_size) for name, params in self.init_mlp_params.items()}

    def store(
        self,
        tokens, # (b n d)
        memories: dict[str, Tensor] | None = None
    ):

        batch_size = tokens.shape[0]

        if not exists(memories):
            memories = self.get_init_mlp_params(batch_size)

        tokens = self.norm(tokens)

        keys, values = self.to_key_values(tokens).chunk(2, dim = -1)

        loss_weight = self.to_loss_weight(tokens)

        grad = self.grad_fn(memories, (keys, values, loss_weight))

        # prepare forget

        forget = self.to_forget_gate(tokens)

        # update memories

        next_memories = dict()

        for param_name, past_memory in memories.items():
            change = grad[param_name]

            past_memory = einx.multiply('b, b ...', forget, past_memory)

            next_memories[param_name] = past_memory - change

        return next_memories

    def forward(
        self,
        tokens, # (b n d)
        memories: dict[str, Tensor] | None = None
    ):
        batch_size = tokens.shape[0]

        if not exists(memories):
            memories = self.get_init_mlp_params(batch_size)

        tokens = self.norm(tokens)

        queries = self.to_queries(tokens)

        retrieved = self.retrieve_fn(memories, queries)

        return retrieved

# state embedder

class StateEmbedder(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_state: tuple[int, ...] | list[int] | int,
        num_internal_states: int | None = None,
        internal_states_selectors: list[list[int]] | None = None
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

        dim_states = (dim_state,) if not isinstance(dim_state, (tuple, list)) else dim_state

        self.dim_states = dim_states
        self.state_to_token = ModuleList([MLP(dim_state, dim, bias = False) for dim_state in dim_states])

        # internal state embeds for each robot

        self.internal_state_embedder = None

        if exists(num_internal_states) and exists(internal_states_selectors):
            self.internal_state_embedder = Embed(
                dim,
                num_continuous = num_internal_states,
                selectors = internal_states_selectors
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        state,
        state_type,
        state_id = 0,
        internal_state = None,
        internal_state_selector_id: int | None = None
    ):

        if state_type == 'image':
            token_embeds = self.image_to_token(state)
        elif state_type == 'raw':
            state_to_token = self.state_to_token[state_id]
            token_embeds = state_to_token(state)
        else:
            raise ValueError('invalid state type')

        if (
            exists(internal_state_selector_id) and
            exists(internal_state) and
            exists(self.internal_state_embedder)
        ):
            internal_state = internal_state.to(self.device)

            internal_state_embed = self.internal_state_embedder(internal_state, selector_index = internal_state_selector_id)

            token_embeds = token_embeds + internal_state_embed

        return token_embeds

# class

RewardShapingFn = Callable[..., float | Tensor]

RewardShapingInputFn = Callable[..., Any]

StateNamedConfig = tuple[tuple[str, int] | str]

RewardStorePath = str | tuple[str, int]

OneRewardShaper = RewardShapingFn | tuple[RewardShapingFn, RewardStorePath | None]

MaybeOneRewardShaper = OneRewardShaper | None

@beartype
def default_parse_env_reset_out(reset_out: tuple):
    assert len(reset_out) == 2
    return dict(zip(('state', 'info'), reset_out))

@beartype
def default_parse_env_step_out(step_out: tuple):
    assert len(step_out) in {4, 5}

    if len(step_out) == 5:
        data_dict = dict(zip(('state', 'reward', 'terminated', 'truncated', 'info'), step_out))
    elif len(step_out) == 4:
        data_dict = dict(zip(('state', 'reward', 'terminated', 'info'), step_out))
        data_dict['truncated'] = False

    return data_dict

class Locoformer(Module):
    def __init__(
        self,
        embedder: dict | Module,
        unembedder: dict | Readout,
        transformer: dict | TransformerXL,
        *,
        discount_factor = 0.999,
        gae_lam = 0.95,
        ppo_eps_clip = 0.2,
        ppo_entropy_weight = 0.01,
        ppo_value_clip = 0.4,
        ppo_soft_constrain_action_max = None,
        ppo_soft_constrain_action_loss_weight = 0.1,
        dim_value_input = None,                 # needs to be set for value network to be available
        value_network: Module = nn.Identity(),
        policy_network: Module = nn.Identity(),
        state_pred_network: Module | None = None,
        embed_past_action = False,
        state_pred_loss_weight = 0.05,
        reward_range: tuple[float, float] | None = None,
        reward_shaping_fns: (
            MaybeOneRewardShaper |
            list[MaybeOneRewardShaper] |
            list[list[MaybeOneRewardShaper]]
        ) = None,
        computed_reward_shaping_input_fns: dict[str, RewardShapingInputFn] = dict(),
        state_named_config: StateNamedConfig | None = None,
        num_reward_bins = 32,
        hl_gauss_loss_kwargs = dict(),
        value_loss_weight = 0.5,
        calc_gae_kwargs: dict = dict(),
        parse_env_reset_out: Callable | None = None,
        parse_env_step_out: Callable | None = None,
        recurrent_cache = True,
        use_spo = False,        # simple policy optimization https://arxiv.org/abs/2401.16025 - Levine's group (PI) verified it is more stable than PPO
        asymmetric_spo = False, # https://openreview.net/pdf?id=BA6n0nmagi
        max_mem_segments = 1
    ):
        super().__init__()

        if isinstance(transformer, dict):
            transformer = TransformerXL(max_mem_segments = max_mem_segments, **transformer)

        self.transformer = transformer

        # handle state embedder

        if isinstance(embedder, dict):
            embedder = StateEmbedder(**embedder)

        self.embedder = embedder

        # unembed state to actions or ssl predictions

        action_embedder = None
        if isinstance(unembedder, dict):
            action_embedder, unembedder = EmbedAndReadout(
                explicit_single_action_dim_given = True,
                **unembedder,
            )

        self.unembedder = unembedder

        # embedding past actions

        self.past_action_embedder = None
        self.embed_past_action = embed_past_action

        if embed_past_action and exists(action_embedder):
            self.past_action_embedder = action_embedder

        # attention window related

        self.fixed_window_size = transformer.fixed_window_size
        self.window_size = transformer.window_size
        self.max_mem_segments = max_mem_segments

        # policy network

        self.policy_network = policy_network

        # determine value network, using HL Gauss Layer

        self.to_value_pred = None

        if exists(dim_value_input):
            assert exists(reward_range)

            self.to_value_pred = nn.Sequential(
                value_network,
                LinearNoBias(dim_value_input, num_reward_bins)
            )

            reward_min, reward_max = reward_range

            self.hl_gauss_loss = HLGaussLoss(
                min_value = reward_min,
                max_value = reward_max,
                num_bins = num_reward_bins,
                **hl_gauss_loss_kwargs
            )

        # state prediction related

        self.can_pred_state = exists(state_pred_network)
        self.state_pred_network = state_pred_network

        if exists(state_pred_network):
            dim_states = self.embedder.dim_states
            total_dim_states = sum(dim_states)

            selectors = [t.tolist() for t in arange(total_dim_states).split(dim_states)]

            self.state_pred_head = Readout(transformer.dim, num_continuous = total_dim_states, selectors = selectors)

        self.has_state_pred_loss = state_pred_loss_weight > 0.
        self.state_pred_loss_weight = state_pred_loss_weight

        # ppo related

        self.discount_factor = discount_factor
        self.gae_lam = gae_lam
        self.ppo_eps_clip = ppo_eps_clip
        self.ppo_entropy_weight = ppo_entropy_weight
        self.ppo_value_clip = ppo_value_clip
        self.ppo_soft_constrain_action_max = ppo_soft_constrain_action_max
        self.ppo_soft_constrain_action_loss_weight = ppo_soft_constrain_action_loss_weight

        self.value_loss_weight = value_loss_weight

        self.calc_gae_kwargs = calc_gae_kwargs

        # maybe use spo

        self.use_spo = use_spo

        self.asymmetric_spo = asymmetric_spo

        # maybe recurrent kv cache, from Ding et al. https://arxiv.org/abs/2012.15688

        self.recurrent_cache = recurrent_cache

        # environment returns to dictionary

        self.parse_env_reset_out = default(parse_env_reset_out, default_parse_env_reset_out)
        self.parse_env_step_out = default(parse_env_step_out, default_parse_env_step_out)

        # reward shaping function

        self.has_reward_shaping = exists(reward_shaping_fns)

        if self.has_reward_shaping:
            if is_bearable(reward_shaping_fns, OneRewardShaper):
                reward_shaping_fns = [reward_shaping_fns]

            is_multiple = is_bearable(reward_shaping_fns, list[list[MaybeOneRewardShaper]])
            is_single = is_bearable(reward_shaping_fns, list[MaybeOneRewardShaper])

            assert is_multiple or is_single, 'reward shaping functions must be either list of shapers (2d) or list of list of shapers (3d)'

            self.reward_shaping_fns_multiple_envs = is_multiple

            # validation

            all_shaper_lists = reward_shaping_fns if is_multiple else [reward_shaping_fns]

            for shaper_list in all_shaper_lists:
                for shaper in shaper_list:
                    if not exists(shaper):
                        continue

                    has_store_field = isinstance(shaper, tuple) and len(shaper) == 2 and exists(shaper[1])

                    if not is_multiple:
                        if not has_store_field:
                            continue

                        store_path = shaper[1]
                        store_field_name, store_field_idx = (store_path, 0) if isinstance(store_path, str) else store_path

                        assert store_field_idx == 0, f'if reward shaping is 2d, store field index must be 0, but got {store_field_idx}'
                    else:
                        assert has_store_field, 'reward shaping with multiple envs (3d) must have a store field'

        if exists(state_named_config):
            assert 'state_named' not in computed_reward_shaping_input_fns, '`state_named` is already present in `computed_reward_shaping_input_fns`'
            computed_reward_shaping_input_fns['state_named'] = lambda state: tensor_to_dict(state, state_named_config)

        self.reward_shaping_fns = reward_shaping_fns
        self.computed_reward_shaping_input_fns = computed_reward_shaping_input_fns

        # loss related

        self.register_buffer('zero', tensor(0.), persistent = False)

    @property
    def device(self):
        return next(self.parameters()).device

    def actor_parameters(self):
        return [
            *self.policy_network.parameters(),
            *self.unembedder.parameters()
        ]

    def critic_parameters(self):
        if not exists(self.to_value_pred):
            return []

        return self.to_value_pred.parameters()

    @beartype
    def learn(
        self,
        optims,
        accelerator,
        replay,
        state_embed_kwargs: dict,
        action_select_kwargs: dict,
        state_id_kwarg: dict = dict(),
        batch_size = 16,
        epochs = 2,
        use_vision = False,
        compute_state_pred_loss = False,
        state_pred_loss_weight = None,
        maybe_construct_trial_from_buffer: Callable[[ReplayBuffer], Tensor] | None = None
    ):
        state_field = 'state_image' if use_vision else 'state'

        episode_mapping = None

        if exists(maybe_construct_trial_from_buffer):
            episode_mapping = maybe_construct_trial_from_buffer(replay)

        dataset = replay.dataset()

        if exists(episode_mapping):
            dataset = RemappedReplayDataset(dataset, episode_mapping)

        dl = replay.dataloader(
            batch_size = batch_size,
            dataset = dataset,
            shuffle = True
        )

        self.train()

        self, dl, *optims = accelerator.prepare(self, dl, *optims)

        for _ in range(epochs):
            for data in dl:

                data = SimpleNamespace(**data)

                actor_loss, critic_loss = self.ppo(
                    state = getattr(data, state_field),
                    internal_state = getattr(data, 'internal_state', None),
                    action = data.action,
                    action_log_prob = data.action_log_prob,
                    reward = data.reward,
                    value = data.value,
                    done = data.done,
                    condition = getattr(data, 'condition', None),
                    cond_mask = getattr(data, 'cond_mask', None),
                    episode_lens = data._lens,
                    optims = optims,
                    state_embed_kwargs = state_embed_kwargs,
                    action_select_kwargs = action_select_kwargs,
                    state_id_kwarg = state_id_kwarg,
                    compute_state_pred_loss = compute_state_pred_loss,
                    state_pred_loss_weight = state_pred_loss_weight,
                    accelerator = accelerator
                )

                accelerator.print(f'actor: {actor_loss.item():.3f} | critic: {critic_loss.item():.3f}')

    def evolve(
        self,
        environment,
        **kwargs
    ):
        evo_strat = EvoStrategy(self, environment = environment, **kwargs)
        evo_strat()

    def ppo(
        self,
        state,
        internal_state,
        action,
        action_log_prob,
        reward,
        value,
        done,
        episode_lens,
        condition: Tensor | None = None,
        cond_mask: Tensor | None = None,
        optims: list[Optimizer] | None = None,
        state_embed_kwargs: dict = dict(),
        action_select_kwargs: dict = dict(),
        state_id_kwarg: dict = dict(),
        compute_state_pred_loss = True,
        state_pred_loss_weight = None,
        accelerator = None,
        max_grad_norm = 0.5
    ):
        state_pred_loss_weight = default(state_pred_loss_weight, self.state_pred_loss_weight)

        window_size = self.window_size
        mask = ~done
        seq_len = state.shape[1]
        padding_mask = einx.less('j, i -> i j', arange(seq_len, device = self.device), episode_lens)
        gae_mask = padding_mask & mask

        total_learnable_tokens = gae_mask.sum().item()

        advantage, returns = calc_gae(reward, value, masks = gae_mask, lam = self.gae_lam, gamma = self.discount_factor, **self.calc_gae_kwargs)

        advantage = normalize(advantage, mask = gae_mask)

        advantage = rearrange(advantage, '... -> ... 1')

        past_action = pad_at_dim(action, (1, -1), dim = -2)

        data_dict = dict(
            state = state,
            internal_state = internal_state,
            action = action,
            past_action = past_action,
            old_action_log_prob = action_log_prob,
            reward = reward,
            mask = mask,
            advantage = advantage,
            returns = returns,
            windowed_gae_mask = gae_mask,
            condition = condition,
            cond_mask = cond_mask
        )

        num_windows = math.ceil(seq_len / window_size)

        windowed_data = dict()

        for name, tensor in data_dict.items():
            if exists(tensor):
                windowed_data[name] = tensor.split(window_size, dim = 1)
            else:
                windowed_data[name] = (None,) * num_windows

        mean_actor_loss = self.zero.clone()
        mean_critic_loss = self.zero.clone()

        # learn across windows

        cache = None

        for window_tensors in zip(*windowed_data.values()):

            data = SimpleNamespace(**dict(zip(windowed_data.keys(), window_tensors)))

            ((action_logits, maybe_state_pred), value_logits), cache = self.forward(data.state, past_action = data.past_action if self.embed_past_action else None, state_embed_kwargs = {**state_embed_kwargs, 'internal_state': data.internal_state}, action_select_kwargs = action_select_kwargs, state_id_kwarg = state_id_kwarg, condition = data.condition, cond_mask = data.cond_mask, cache = cache, detach_cache = True, return_values = True, return_raw_value_logits = True, return_state_pred = True)

            log_prob = self.unembedder.log_prob(action_logits, data.action, **action_select_kwargs)

            # update actor, classic clipped surrogate loss

            eps_clip = self.ppo_eps_clip
            ratio = (log_prob - data.old_action_log_prob).exp()

            calc_spo = lambda: -(ratio * data.advantage - (data.advantage.abs() * (ratio - 1.).square()) / (2 * eps_clip))

            calc_ppo = lambda: -torch.min(ratio * data.advantage, ratio.clamp(1. - eps_clip, 1. + eps_clip) * data.advantage)

            if self.asymmetric_spo:
                actor_loss = torch.where(data.advantage >= 0, calc_ppo(), calc_spo())
            elif self.use_spo:
                actor_loss = calc_spo()
            else:
                actor_loss = calc_ppo()

            # maybe entropy

            if self.ppo_entropy_weight > 0.:
                entropy = self.unembedder.entropy(action_logits, **action_select_kwargs)

                if exists(entropy):
                    actor_loss = actor_loss - self.ppo_entropy_weight * entropy

            windowed_actor_loss = actor_loss[data.windowed_gae_mask].sum() / total_learnable_tokens

            # maybe add state prediction

            if (
                exists(maybe_state_pred) and
                self.has_state_pred_loss and
                compute_state_pred_loss and
                data.windowed_gae_mask[:, :-1].any()
            ):
                state_pred = maybe_state_pred[:, :-1]
                state_labels = data.state[:, 1:]
                loss_mask = data.windowed_gae_mask[:, :-1]

                state_id = state_id_kwarg.get('state_id', 0)

                state_pred_loss = self.state_pred_head.calculate_loss(state_pred, state_labels, selector_index = state_id, return_unreduced_loss = True)

                state_pred_loss = state_pred_loss.mean(dim = -1) # average over state features

                windowed_state_pred_loss = state_pred_loss[loss_mask].sum() / total_learnable_tokens

                windowed_actor_loss = (
                    windowed_actor_loss +
                    windowed_state_pred_loss * state_pred_loss_weight
                )

            # maybe soft constrain continuous actions

            if (
                self.ppo_soft_constrain_action_max and
                self.unembedder.has_continuous
            ):
                loss_mask = data.windowed_gae_mask

                soft_constrain_loss = (action_logits[..., 0].abs() - self.ppo_soft_constrain_action_max).relu().pow(2)
                windowed_soft_constrain_loss = soft_constrain_loss[loss_mask].sum() / total_learnable_tokens

                windowed_actor_loss = (
                    windowed_actor_loss +
                    windowed_soft_constrain_loss * self.ppo_soft_constrain_action_loss_weight
                )

            # windowed loss

            windowed_actor_loss.backward(retain_graph = True)

            # update critic

            value_loss = self.hl_gauss_loss(value_logits, data.returns, reduction = 'none') * self.value_loss_weight

            windowed_critic_loss = value_loss[data.windowed_gae_mask].sum() / total_learnable_tokens
            windowed_critic_loss.backward(retain_graph = True)

            # accumulate

            mean_actor_loss.add_(windowed_actor_loss)
            mean_critic_loss.add_(windowed_critic_loss)

        # optimizer update

        if exists(optims):

            if exists(accelerator):
                accelerator.clip_grad_norm_(self.parameters(), max_grad_norm)
            else:
                nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

            for optim in optims:
                optim.step()
                optim.zero_grad()

        # return losses for logging

        return mean_actor_loss.detach(), mean_critic_loss.detach()

    def state_and_command_to_rewards(
        self,
        state,
        commands = None,
        action = None,
        replay_buffer = None,
        env_step_info = None,
        env_index: int | None = None
    ) -> Tensor:

        buffer = replay_buffer
        assert self.has_reward_shaping
        assert xnor(exists(env_index), self.reward_shaping_fns_multiple_envs), f'`env_index` must be passed in if multiple reward shaping functions are defined, and vice versa (not passed in if only single list of reward shaping functions)'

        rewards = []

        reward_shaping_fns = self.reward_shaping_fns[env_index] if exists(env_index) else self.reward_shaping_fns

        # ready a dictionary that contains all data possible for use for reward shaping

        reward_shaping_inputs = dict(
            state = lambda: state,
            commands = lambda: commands,
            command = lambda: commands,
            action = lambda: action,
            actions = lambda: buffer.data['action'][buffer.episode_index, :buffer.timestep_index],
            env_step_info = lambda: env_step_info,
            buffer = lambda: replay_buffer
        )

        def resolve_inputs(fn):
            sig = signature(fn)
            names = sig.parameters.keys()
            resolved = dict()

            for name in names:
                if name not in reward_shaping_inputs:
                    continue

                val = reward_shaping_inputs[name]
                if callable(val):
                    val = val()
                    reward_shaping_inputs[name] = val
                resolved[name] = val

            return resolved

        for key, input_fn in self.computed_reward_shaping_input_fns.items():
            assert key not in reward_shaping_inputs, f'reward shaping input key "{key}" already exists'

            reward_shaping_inputs[key] = lambda fn = input_fn: fn(**resolve_inputs(fn))

        valid_reward_shaping_param_names = set(reward_shaping_inputs.keys())

        # go through the functions

        for reward_shaping_info in reward_shaping_fns:

            if isinstance(reward_shaping_info, tuple):
                fn, reward_store_path = reward_shaping_info
            else:
                fn, reward_store_path = reward_shaping_info, None

            if isinstance(reward_store_path, tuple):
                store_field_name, store_field_index = reward_store_path
            else:
                store_field_name, store_field_index = reward_store_path, 0 # if not a tuple, assume it is a string field name, and we will store at 0th index (for scalar fields)

            assert callable(fn)

            sig = signature(fn)
            for name, p in sig.parameters.items():
                if p.default == Parameter.empty:
                    assert name in valid_reward_shaping_param_names, f'your reward shaping function needs to have param "{name}" within the set of {valid_reward_shaping_param_names}'

            fn_params = resolve_inputs(fn)

            # invoke reward function

            reward = fn(**fn_params)
            rewards.append(reward)

            # maybe store

            if exists(replay_buffer) and exists(reward_store_path):
                eps_index, time_index = buffer.episode_index, buffer.timestep_index

                buffer_field = buffer.data[store_field_name]

                if buffer_field.ndim == 2:
                    buffer_field[eps_index, time_index] = reward
                else:
                    buffer_field[eps_index, time_index, store_field_index] = reward

        # cast to Tensor if returns a float, just make it flexible for researcher

        rewards = [tensor(reward) if not is_tensor(reward) else reward for reward in rewards]

        assert all([r.numel() == 1 for r in rewards])

        if len(rewards) == 0:
            return None

        packed_rewards, _ = pack(rewards, '*')
        return packed_rewards

    @beartype
    def wrap_env_functions(
        self,
        env,
        env_output_transforms: dict[str, Callable] = dict(),
        state_transform: Callable = identity,
        reward_norm = 1.,
        command_generator: Callable = always(None)
    ):

        def transform_output(el):
            if isinstance(el, ndarray):
                return from_numpy(el)
            elif isinstance(el, (int, bool, float)):
                return tensor(el)
            else:
                return el

        def wrapped_reset(*args, **kwargs):
            env_reset_out =  env.reset(*args, **kwargs)

            env_reset_out_torch = tree_map(transform_output, env_reset_out)

            env_reset_out_dict = self.parse_env_reset_out(env_reset_out_torch)

            env_reset_out_dict['state'] = state_transform(env_reset_out_dict['state'])

            derived_states = dict()

            for derived_name, transform in env_output_transforms.items():
                derived_states[derived_name] = transform(env_reset_out_dict, env)

            env_reset_out_dict['derived_state'] = derived_states

            return env_reset_out_dict

        def wrapped_step(action, *args, command = None, replay_buffer = None, env_index = None, **kwargs):

            if is_tensor(action):
                if action.numel() == 1:
                    action = action.item()
                else:
                    action = action.tolist()

            env_step_out = env.step(action, *args, **kwargs)

            env_step_out_torch = tree_map(transform_output, env_step_out)

            env_step_out_dict = self.parse_env_step_out(env_step_out_torch)

            env_step_out_dict['state'] = state_transform(env_step_out_dict['state'])

            env_step_out_dict['reward'] = env_step_out_dict['reward'] / reward_norm

            if self.has_reward_shaping:

                shaped_rewards = self.state_and_command_to_rewards(
                    env_step_out_dict['state'],
                    commands = command,
                    action = action,
                    replay_buffer = replay_buffer,
                    env_step_info = env_step_out_dict['info'],
                    env_index = env_index
                )

                if exists(shaped_rewards):
                    env_step_out_dict['shaped_rewards'] = shaped_rewards

                    # add shaped rewards to main reward

                    env_step_out_dict['reward'] = env_step_out_dict['reward'] + shaped_rewards.sum()

            derived_states = dict()

            for derived_name, transform in env_output_transforms.items():
                derived_states[derived_name] = transform(env_step_out_dict, env)

            env_step_out_dict['derived_state'] = derived_states

            return env_step_out_dict

        return wrapped_reset, wrapped_step

    def get_stateful_forward(
        self,
        initial_states: Tensor | None = None,
        inference_mode = False,
        has_batch_dim = False,
        has_time_dim = False,
        state_time_dim = 1,
        **kwargs
    ):

        cache = None

        def stateful_forward(
            state: Tensor,
            condition: Tensor | None = None,
            cond_mask: Tensor | None = None,
            **override_kwargs
        ):
            nonlocal cache

            state = state.to(self.device)

            if exists(condition):
                condition = condition.to(self.device)

            if exists(cond_mask):
                cond_mask = cond_mask.to(self.device)

            # handle no batch or time, for easier time rolling out against envs

            if not has_batch_dim:
                state = rearrange(state, '... -> 1 ...')

                if exists(condition):
                    condition = rearrange(condition, '... -> 1 ...')

                if exists(cond_mask):
                    cond_mask = rearrange(cond_mask, '... -> 1 ...')

            if not has_time_dim:
                state = state.unsqueeze(state_time_dim)

                if exists(condition):
                    condition = rearrange(condition, '... d -> ... 1 d')

                if exists(cond_mask):
                    cond_mask = cond_mask.unsqueeze(state_time_dim)

            # handle episode_id

            episode_id = override_kwargs.get('episode_id')

            if exists(episode_id):
                episode_id = episode_id.to(self.device)

                if not has_batch_dim and episode_id.ndim == 0:
                    episode_id = rearrange(episode_id, '... -> 1 ...')

                if not has_time_dim and episode_id.ndim < (2 if has_batch_dim else 1):
                    episode_id = episode_id.unsqueeze(state_time_dim)

                override_kwargs['episode_id'] = episode_id

            # forwards

            out, cache = self.forward(
                state,
                condition = condition,
                cond_mask = cond_mask,
                cache = cache,
                **{**kwargs, **override_kwargs}
            )

            # maybe remove batch or time

            if not has_time_dim:
                out = tree_map_tensor(out, lambda t: t.squeeze(state_time_dim))

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

    @beartype
    def gather_experience_from_env_(
        self,
        wrapped_env_functions: tuple[Callable, Callable],
        replay: ReplayBuffer,
        embed_past_action = False,
        max_timesteps = None,
        use_vision = False,
        replay_buffer = None,
        action_select_kwargs: dict = dict(),
        state_embed_kwargs: dict = dict(),
        state_id_kwarg: dict = dict(),
        env_index: int | None = None,
        state_entropy_bonus_weight = 0.,
        action_rescale_range: tuple[float, float] | None = None,
        command_fn: Callable = always(None)
    ):

        self.eval()

        env_reset, env_step = wrapped_env_functions

        reset_out_dict = env_reset()
        derived, state = pick(reset_out_dict, ('derived_state', 'state'))

        state_image = derived.get('state_image', None)
        internal_state = derived.get('internal_state', None)

        timestep = 0

        max_timesteps = default(max_timesteps, replay.max_timesteps)

        stateful_forward = self.get_stateful_forward(
            has_batch_dim = False,
            has_time_dim = False,
            inference_mode = True
        )

        cum_rewards = 0.

        with replay.one_episode() as final_meta_data_store_dict:

            past_action = None

            while True:
                state_for_model = state_image if use_vision else state

                maybe_command = command_fn(state_for_model)

                # predict next action

                (action_logits, state_pred), value = stateful_forward(
                    state_for_model,
                    condition = maybe_command,
                    cond_mask = tensor(exists(maybe_command)),
                    state_embed_kwargs = {**state_embed_kwargs, 'internal_state': internal_state},
                    action_select_kwargs = action_select_kwargs,
                    state_id_kwarg = state_id_kwarg,
                    past_action = past_action if embed_past_action else None,
                    return_values = True,
                    return_state_pred = True
                )

                action = self.unembedder.sample(action_logits, **action_select_kwargs)

                # maybe clip

                if exists(action_rescale_range):
                    min_val, max_val = action_rescale_range
                    action = (action + 1.) * 0.5 * (max_val - min_val) + min_val

                # pass to environment

                step_dict = env_step(
                    action,
                    command = maybe_command,
                    replay_buffer = replay_buffer,
                    env_index = env_index
                )

                derived, next_state, reward, terminated, truncated = pick(step_dict, ('derived_state', 'state', 'reward', 'terminated', 'truncated'))

                next_state_image = derived.get('state_image', None)
                next_internal_state = derived.get('internal_state', None)

                # maybe state entropy bonus

                if state_entropy_bonus_weight > 0. and exists(state_pred):
                    state_id = state_id_kwarg.get('state_id', 0)
                    entropy = self.state_pred_head.entropy(state_pred, selector_index = state_id)

                    state_entropy_bonus = (entropy * state_entropy_bonus_weight).sum()

                    reward = reward + state_entropy_bonus.item() # the entropy is directly related to log variance

                # cum rewards

                cum_rewards += reward

                # increment counters
                # we will store the step with done=False, as only the bootstrap/boundary node is done=True

                exceeds_max_timesteps = max_timesteps >= 0 and timestep == (max_timesteps - 1)
                should_stop = truncated or terminated or tensor(exceeds_max_timesteps)

                # get log prob of action

                action_log_prob = self.unembedder.log_prob(action_logits, action, **action_select_kwargs)

                memory = replay.store(
                    state = state,
                    state_image = state_image,
                    action = action,
                    action_log_prob = action_log_prob,
                    internal_state = internal_state,
                    reward = reward,
                    value = value,
                    done = tensor(False),
                    condition = maybe_command,
                    cond_mask = tensor(exists(maybe_command))
                )

                timestep += 1

                # break if done or exceed max timestep
                if should_stop:

                    # handle bootstrap value, which is a non-learnable timestep added with the next value for GAE
                    # only if terminated signal not detected
                    if not terminated:
                        next_state_for_model = next_state_image if use_vision else next_state

                        _, next_value = stateful_forward(next_state_for_model, condition = maybe_command, cond_mask = tensor(exists(maybe_command)), return_values = True, state_embed_kwargs = {**state_embed_kwargs, 'internal_state': internal_state}, state_id_kwarg = state_id_kwarg, action_select_kwargs = action_select_kwargs)

                        terminal_node = dict(
                            state = next_state,
                            state_image = next_state_image,
                            internal_state = next_internal_state,
                            value = next_value,
                            reward = next_value,
                            done = True,
                            condition = maybe_command,
                            cond_mask = exists(maybe_command)
                        )

                    else:
                        # terminal node - store a step with 0 reward and value, and done=True, to stop GAE scan
                        terminal_node = dict(
                            state = next_state,
                            state_image = next_state_image,
                            internal_state = next_internal_state,
                            value = torch.zeros_like(value),
                            reward = torch.zeros_like(reward),
                            done = True,
                            condition = maybe_command,
                            cond_mask = exists(maybe_command)
                        )

                    terminal_node = {key: value for key, value in terminal_node.items() if key in memory._fields}

                    terminal_memory = memory._replace(**terminal_node)

                    replay.store(**terminal_memory._asdict())

                    # store the final cumulative reward into meta data

                    final_meta_data_store_dict.update(cum_rewards = cum_rewards)

                    break

                state = next_state
                state_image = next_state_image
                internal_state = next_internal_state

                past_action = action

        return cum_rewards

    def forward(
        self,
        state: Tensor,
        cache: TransformerMemory | None = None,
        condition: Tensor | None = None,
        cond_mask: Tensor | None = None,
        past_action: Tensor | None = None,
        state_embed_kwargs: dict = dict(),
        action_select_kwargs: dict = dict(),
        state_id_kwarg: dict = dict(),
        detach_cache = False,
        return_values = False,
        return_state_pred = False,
        return_raw_value_logits = False,
        episode_id: Tensor | None = None
    ):

        state = state.to(self.device)

        # move condition

        if exists(condition):
            condition = condition.to(self.device)

        # determine which function to invoke for state to token for transformer

        state_to_token = self.embedder

        # embed

        tokens = state_to_token(state, **state_embed_kwargs, **state_id_kwarg)

        # maybe add past action

        # determine if first window and start of sequence

        total_tokens = cache.total_tokens if exists(cache) else 0

        is_start_of_sequence = total_tokens == 0

        # maybe add past action

        if exists(past_action):
            assert self.embed_past_action
            past_action_embed = self.past_action_embedder(past_action, **action_select_kwargs)

            if is_start_of_sequence:
                past_action_embed = pad_at_dim(past_action_embed[..., 1:, :], (1, 0), dim = -2)

            tokens = tokens + past_action_embed

        # time

        time = tokens.shape[-2]

        # determine if we have episode ids

        has_episode_id = exists(episode_id)

        if is_start_of_sequence:
            passed_episode_ids = has_episode_id
        else:
            *_, passed_episode_ids = cache
            assert xnor(has_episode_id, passed_episode_ids), 'episode_id must be passed in if it was passed in the first window, and vice versa'

        # time

        assert ((total_tokens % self.window_size) + time) <= self.window_size

        # attention

        if not exists(cache):
            memory_segments = deque(maxlen = self.max_mem_segments)
            episode_id_segments = deque(maxlen = self.max_mem_segments)
            curr_episode_ids = None
        else:
            total_tokens, kv_cache, gru_cache, mem_mlp_cache, mem_mlp_hidden_states, memory_segments, episode_id_segments, curr_episode_ids, _ = cache

        attn_cache = cache

        embed, cache = self.transformer(
            tokens,
            condition = condition,
            cond_mask = cond_mask,
            cache = attn_cache,
            return_kv_cache = True,
            episode_ids = episode_id
        )

        # unembed to actions - in language models this would be the next state

        policy_embed = self.policy_network(embed)

        action_logits = self.unembedder(policy_embed, **action_select_kwargs)

        out = action_logits

        # maybe return state prediction

        if return_state_pred:
            state_pred = None

            if self.can_pred_state:
                state_id = state_id_kwarg.get('state_id', 0)
                state_pred_embed = self.state_pred_network(embed)
                state_pred = self.state_pred_head(state_pred_embed, selector_index = state_id)

            out = (out, state_pred)

        # maybe detach cache

        if detach_cache:
            cache = tree_map_tensor(cache, lambda t: t.detach())

        # handle returning of values

        if return_values:
            assert exists(self.to_value_pred)

            values = self.to_value_pred(embed)

            if not return_raw_value_logits:
                values = self.hl_gauss_loss(values) # converts the value logits to scalar values

            out = (out, values)

        # handle curtailing kv cache at the right intervals

        total_tokens, kv_cache, gru_cache, mem_mlp_cache, mem_mlp_hidden_states, memory_segments, episode_id_segments, curr_episode_ids, _ = cache

        # accumulate curr_episode_ids

        curr_episode_ids = safe_cat(curr_episode_ids, episode_id, dim = -1)

        if self.fixed_window_size or divisible_by(total_tokens, self.window_size * 2):
            kv_cache = kv_cache[..., -self.window_size:, :]
            curr_episode_ids = curr_episode_ids[..., -self.window_size:] if exists(curr_episode_ids) else None

        if self.recurrent_cache and divisible_by(total_tokens, self.window_size):
            kv_cache = torch.roll(kv_cache, shifts = -1, dims = 0)
            if exists(curr_episode_ids):
                curr_episode_ids = torch.roll(curr_episode_ids, shifts = -1, dims = 0)

            if exists(gru_cache):
                gru_cache = torch.roll(gru_cache, shifts = -1, dims = 0)

        if divisible_by(total_tokens, self.window_size):
            memory_segments.append(kv_cache[..., -self.window_size:, :].detach())
            episode_id_segments.append(curr_episode_ids[..., -self.window_size:].detach() if exists(curr_episode_ids) else None)

        cache = TransformerMemory(total_tokens, kv_cache, gru_cache, mem_mlp_cache, mem_mlp_hidden_states, memory_segments, episode_id_segments, curr_episode_ids, passed_episode_ids)

        return out, cache
