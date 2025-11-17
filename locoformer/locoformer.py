from __future__ import annotations
from functools import partial

from pathlib import Path
from contextlib import contextmanager
from collections import namedtuple

import numpy as np
from numpy import ndarray
from numpy.lib.format import open_memmap

from beartype import beartype
from beartype.door import is_bearable

import torch
from torch import nn, cat, stack, arange, Tensor, tensor, is_tensor, from_numpy
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, RMSNorm, Identity, Sequential
from torch.utils._pytree import tree_map
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer

import einx
from einops import rearrange, einsum
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding

from assoc_scan import AssocScan

# constants

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

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

def tree_map_tensor(x, fn):
    return tree_map(lambda t: t if not is_tensor(t) else fn(t), x)

def pad_at_dim(
    t,
    pad: tuple[int, int],
    dim = -1,
    value = 0.
):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def calc_entropy(logits):
    prob = logits.softmax(dim = -1)
    return -(prob * log(prob)).sum(dim = -1)

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

# data

def collate_var_time(data):

    datum = first(data)
    keys = datum.keys()

    all_tensors = zip(*[datum.values() for datum in data])

    collated_values = []

    for key, tensors in zip(keys, all_tensors):

        # the episode lens have zero dimension - think of a cleaner way to handle this later

        if key != '_lens':

            times = [t.shape[0] for t in tensors]
            max_time = max(times)
            tensors = [pad_at_dim(t, (0, max_time - t.shape[0]), dim = 0) for t in tensors]

        collated_values.append(stack(tensors))

    return dict(zip(keys, collated_values))

class ReplayDataset(Dataset):
    def __init__(
        self,
        folder: str | Path,
        fields: tuple[str, ...] | None = None
    ):
        if isinstance(folder, str):
            folder = Path(folder)

        episode_lens = folder / 'episode_lens.npy'
        self.episode_lens = open_memmap(str(episode_lens), mode = 'r')

        # get indices of non-zero lengthed episodes

        nonzero_episodes = self.episode_lens > 0
        self.indices = np.arange(self.episode_lens.shape[-1])[nonzero_episodes]

        # get all data files

        filepaths = [*folder.glob('*.data.npy')]
        assert len(filepaths) > 0

        fieldname_to_filepath = {path.name.split('.')[0]: path for path in filepaths}

        fieldnames_from_files = set(fieldname_to_filepath.keys())

        fields = default(fields, fieldnames_from_files)

        self.memmaps = dict()

        for field in fields:
            assert field in fieldnames_from_files, f'invalid field {field} - must be one of {fieldnames_from_files}'

            path = fieldname_to_filepath[field]

            self.memmaps[field] = open_memmap(str(path), mode = 'r')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        episode_index = self.indices[idx]

        episode_len = self.episode_lens[episode_index]

        data = {field: from_numpy(memmap[episode_index, :episode_len].copy()) for field, memmap in self.memmaps.items()}

        data['_lens'] = tensor(episode_len)

        return data

class ReplayBuffer:

    @beartype
    def __init__(
        self,
        folder: str | Path,
        max_episodes: int,
        max_timesteps: int,
        fields: dict[
            str,
            str | tuple[str, int | tuple[int, ...]]
        ]
    ):

        # folder for data

        if not isinstance(folder, Path):
            folder = Path(folder)
            folder.mkdir(exist_ok = True)

        self.folder = folder
        assert folder.is_dir()

        # keeping track of episode length

        episode_lens = folder / 'episode_lens.npy'

        self.episode_index = 0
        self.timestep_index = 0

        self.max_episodes = max_episodes
        self.max_timesteps= max_timesteps

        self.episode_lens = open_memmap(str(episode_lens), mode = 'w+', dtype = np.int32, shape = (max_episodes,))

        # create the memmap for individual data tracks

        self.shapes = dict()
        self.dtypes = dict()
        self.memmaps = dict()
        self.fieldnames = set(fields.keys())

        for field_name, field_info in fields.items():

            # some flexibility

            field_info = (field_info, ()) if isinstance(field_info, str) else field_info

            dtype_str, shape = field_info
            assert dtype_str in {'int', 'float', 'bool'}

            dtype = dict(int = np.int32, float = np.float32, bool = np.bool_)[dtype_str]

            # memmap file

            filepath = folder / f'{field_name}.data.npy'
            memmap = open_memmap(str(filepath), mode = 'w+', dtype = dtype, shape = (max_episodes, max_timesteps, *shape))

            self.memmaps[field_name] = memmap
            self.shapes[field_name] = shape
            self.dtypes[field_name] = dtype

        self.memory_namedtuple = namedtuple('Memory', list(fields.keys()))

    def reset_(self):
        self.episode_lens[:] = 0
        self.episode_index = 0
        self.timestep_index = 0

    def advance_episode(self):
        self.episode_index = (self.episode_index + 1) % self.max_episodes
        self.timestep_index = 0

    def flush(self):
        self.episode_lens[self.episode_index] = self.timestep_index

        for memmap in self.memmaps.values():
            memmap.flush()

        self.episode_lens.flush()

    @contextmanager
    def one_episode(self):

        yield

        self.flush()
        self.advance_episode()

    @beartype
    def store_datapoint(
        self,
        episode_index: int,
        timestep_index: int,
        name: str,
        datapoint: Tensor | ndarray
    ):
        assert 0 <= episode_index < self.max_episodes
        assert 0 <= timestep_index < self.max_timesteps

        if is_tensor(datapoint):
            datapoint = datapoint.detach().cpu().numpy()

        assert name in self.fieldnames, f'invalid field name {name} - must be one of {self.fieldnames}'

        assert datapoint.shape == self.shapes[name], f'invalid shape {datapoint.shape} - shape must be {self.shapes[name]}'

        self.memmaps[name][self.episode_index, self.timestep_index] = datapoint

    def store(
        self,
        **data
    ):
        assert is_bearable(data, dict[str, Tensor | ndarray])

        assert not self.timestep_index >= self.max_timesteps, 'you exceeded the `max_timesteps` set on the replay buffer'

        for name, datapoint in data.items():

            self.store_datapoint(self.episode_index, self.timestep_index, name, datapoint)

        self.timestep_index += 1

        return self.memory_namedtuple(**data)

    def dataset(self) -> Dataset:
        self.flush()

        return ReplayDataset(self.folder)

    def dataloader(self, batch_size, **kwargs) -> DataLoader:
        self.flush()

        return DataLoader(self.dataset(), batch_size = batch_size, collate_fn = collate_var_time, **kwargs)

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
        value_network: Module | None = None,
        discount_factor = 0.999,
        gae_lam = 0.95,
        ppo_eps_clip = 0.2,
        ppo_entropy_weight = 0.01,
        ppo_value_clip = 0.4,
        value_loss_weight = 0.5,
        calc_gae_kwargs: dict = dict()
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

        # ppo related

        self.discount_factor = discount_factor
        self.gae_lam = gae_lam
        self.ppo_eps_clip = ppo_eps_clip
        self.ppo_entropy_weight = ppo_entropy_weight
        self.ppo_value_clip = ppo_value_clip
        self.value_loss_weight = value_loss_weight

        self.calc_gae_kwargs = calc_gae_kwargs

        # loss related

        self.register_buffer('zero', tensor(0.), persistent = False)

    @property
    def device(self):
        return next(self.parameters()).device

    def actor_parameters(self):
        return self.unembedder.parameters()

    def critic_parameters(self):
        if not exists(self.value_network):
            return []

        return self.value_network.parameters()

    def ppo(
        self,
        state,
        action,
        old_action_log_prob,
        reward,
        old_value,
        mask,
        actor_optim: Optimizer | None = None,
        critic_optim: Optimizer | None = None
    ):
        window_size = self.window_size
        total_learnable_tokens = mask.sum().item()

        windowed_tensors = [
            t.split(window_size, dim = 1) for t in
            (
                state,
                action,
                old_action_log_prob,
                reward,
                old_value,
                mask
            )
        ]

        mean_actor_loss = self.zero.clone()
        mean_critic_loss = self.zero.clone()

        # learn across windows

        cache = None

        for (
            state,
            action,
            old_action_log_prob,
            reward,
            old_value,
            mask
        ) in zip(*windowed_tensors):

            (action_logits, value), cache = self.forward(state, cache = cache, detach_cache = True, return_values = True)
            entropy = calc_entropy(action_logits)

            action = rearrange(action, 'b t -> b t 1')
            log_prob = action_logits.gather(-1, action)
            log_prob = rearrange(log_prob, 'b t 1 -> b t')

            # update actor, classic clipped surrogate loss

            eps_clip = self.ppo_eps_clip
            ratio = (log_prob - old_action_log_prob).exp()

            advantage, returns = calc_gae(reward, old_value, lam = self.gae_lam, gamma = self.discount_factor, **self.calc_gae_kwargs)

            actor_loss = -torch.min(ratio * advantage, ratio.clamp(1. - eps_clip, 1. + eps_clip) * advantage)

            actor_loss = actor_loss - self.ppo_entropy_weight * entropy

            windowed_actor_loss = actor_loss[mask].sum() / total_learnable_tokens
            windowed_actor_loss.backward(retain_graph = True)

            # update critic

            value_loss = F.mse_loss(returns, value, reduction = 'none')

            value_clip = self.ppo_value_clip
            clipped_value = old_value + (value - old_value).clamp(-value_clip, value_clip)
            clipped_value_loss = F.mse_loss(returns, clipped_value, reduction = 'none')

            critic_loss = torch.maximum(value_loss, clipped_value_loss) * self.value_loss_weight

            windowed_critic_loss = critic_loss[mask].sum() / total_learnable_tokens
            windowed_critic_loss.backward(retain_graph = True)

            # accumulate

            mean_actor_loss.add_(windowed_actor_loss)
            mean_critic_loss.add_(windowed_critic_loss)

        # optimizer update

        if exists(actor_optim):
            actor_optim.step()
            actor_optim.zero_grad()

        if exists(critic_optim):
            critic_optim.step()
            critic_optim.zero_grad()

        # return losses for logging

        return mean_actor_loss.detach(), mean_critic_loss.detach()

    def wrap_env_functions(self, env):

        def wrapped_reset(*args, **kwargs):
            state, _ =  env.reset(*args, **kwargs)

            if isinstance(state, ndarray):
                state = from_numpy(state)

            return state, _

        def wrapped_step(action, *args, **kwargs):
            out = env.step(action.item(), *args, **kwargs)

            def transform_output(el):
                if isinstance(el, ndarray):
                    return from_numpy(el)
                elif isinstance(el, (int, bool, float)):
                    return tensor(el)
                else:
                    return el

            return tree_map(transform_output, out)

        return wrapped_reset, wrapped_step

    def get_stateful_forward(
        self,
        initial_states: Tensor | None = None,
        inference_mode = False,
        has_batch_dim = False,
        has_time_dim = False,
        **kwargs
    ):
        window_size = self.window_size

        cache = None

        def stateful_forward(state: Tensor, **override_kwargs):
            nonlocal cache

            # handle no batch or time, for easier time rolling out against envs

            if not has_batch_dim:
                state = rearrange(state, '... -> 1 ...')

            if not has_time_dim:
                state = rearrange(state, '... d -> ... 1 d')

            # forwards

            out, cache = self.forward(state, cache = cache, **{**kwargs, **override_kwargs})

            # handle cache

            cache_len = cache.shape[-2]

            if self.fixed_window_size or divisible_by(cache_len, window_size * 2):
                cache = cache[..., -window_size:, :]

            # maybe remove batch or time

            if not has_time_dim:
                out = tree_map_tensor(out, lambda t: rearrange(t, '... 1 d -> ... d'))

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

        state = state.to(self.device)

        tokens = self.embedder(state)

        embed, kv_cache = self.transformer(tokens, cache = cache, return_kv_cache = True)

        # unembed to actions - in language models this would be the next state

        action_logits = self.unembedder(embed)

        out = action_logits

        # maybe detach cache

        if detach_cache:
            kv_cache = kv_cache.detach()

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
