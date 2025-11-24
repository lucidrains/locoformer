from __future__ import annotations
from typing import Callable
from types import SimpleNamespace
from functools import partial

from pathlib import Path
from contextlib import contextmanager
from collections import namedtuple

from inspect import signature

import numpy as np
from numpy import ndarray
from numpy.lib.format import open_memmap

from beartype import beartype
from beartype.door import is_bearable

import torch
from torch import nn, cat, stack, arange, Tensor, tensor, is_tensor, from_numpy
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, RMSNorm, Identity, Sequential
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer

import einx
from einops import rearrange, einsum
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding

from hl_gauss_pytorch import HLGaussLoss

from assoc_scan import AssocScan

from x_mlps_pytorch import MLP

# constants

LinearNoBias = partial(Linear, bias = False)

Cache = namedtuple('Cache', ('curr_timestep', 'kv_cache')) # (int, Tensor)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(arr):
    return arr[0]

def xnor(x, y):
    return not (x ^ y)

def divisible_by(num, den):
    return (num % den) == 0

def num_parameters_on_function(fn):
    return len(signature(fn).parameters)

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

def is_empty(t):
    return t.numel() == 0

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

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / t.std().clamp_min(eps)

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

class RemappedReplayDataset(Dataset):
    def __init__(
        self,
        dataset: ReplayDataset,
        episode_mapping: Tensor | list[list[int]],
        shuffle_episodes = False
    ):
        assert len(dataset) > 0
        self.dataset = dataset

        if is_tensor(episode_mapping):
            assert episode_mapping.dtype in (torch.int, torch.long) and episode_mapping.ndim == 2
            episode_mapping = episode_mapping.tolist()

        self.episode_mapping = episode_mapping
        self.shuffle_episodes = shuffle_episodes

    def __len__(self):
        return len(self.episode_mapping)

    def __getitem__(self, idx):

        episode_indices = self.episode_mapping[idx]

        episode_indices = tensor(episode_indices)
        episode_indices = episode_indices[(episode_indices >= 0) & (episode_indices < len(self.dataset))]

        assert not is_empty(episode_indices)

        if self.shuffle_episodes and episode_indices.numel() > 1:
            num_episodes = len(episode_indices)
            episode_indices = episode_indices[torch.randperm(num_episodes)]

        episode_data = [self.dataset[i] for i in episode_indices.tolist()]

        episode_lens = stack([data.pop('_lens') for data in episode_data])

        keys = first(episode_data).keys()

        values = [list(data.values()) for data in episode_data]

        values = [cat(field_values) for field_values in zip(*values)] # concat across time

        multi_episode_data = dict(zip(keys, values))

        multi_episode_data['_lens'] = episode_lens.sum()

        multi_episode_data['_episode_indices'] = cat([torch.full((episode_len,), episode_index) for episode_len, episode_index in zip(episode_lens, episode_indices)])

        return multi_episode_data

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

    def __len__(self):
        return (self.episode_lens > 0).sum().item()

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

    def dataset(
        self,
        episode_mapping: Tensor | list[list[int]] | None = None,
    ) -> Dataset:
        self.flush()

        dataset = ReplayDataset(self.folder)

        if not exists(episode_mapping):
            return dataset

        return RemappedReplayDataset(dataset, episode_mapping)

    def dataloader(
        self,
        batch_size,
        episode_mapping: Tensor | list[list[int]] | None = None,
        **kwargs
    ) -> DataLoader:
        self.flush()

        return DataLoader(self.dataset(episode_mapping), batch_size = batch_size, collate_fn = collate_var_time, **kwargs)

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
        cond = None,
        **kwargs
    ):

        need_cond = self.accept_condition

        assert xnor(exists(cond), need_cond)

        prenormed = self.norm(x)

        if need_cond:
            if cond.ndim == 2:
                cond = rearrange(cond, 'b d -> b 1 d')

            scale_in = self.to_gamma(cond)
            prenormed = prenormed * (scale_in + 1.)

        all_fn_out = self.fn(prenormed, **kwargs)

        if not need_cond:
            return all_fn_out

        # function may return multiple args

        (out, *rest), tree_spec = tree_flatten(all_fn_out)

        if need_cond:
            scale_out = self.to_ada_norm_zero(cond).sigmoid()
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
        accept_value_residual = False
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
        dim_cond = None,
        final_norm = True,
        fixed_window_size = False,
    ):
        super().__init__()

        condition = exists(dim_cond)

        self.to_cond_tokens = MLP(dim_cond, dim * 2, activate_last = True) if exists(dim_cond) else None

        norm_fn = partial(MaybeAdaRMSNormWrapper, dim = dim, dim_cond = (dim * 2) if condition else None) 

        layers = ModuleList([])

        for i in range(depth):
            is_first = i == 0

            attn = norm_fn(Attention(dim = dim, dim_head = dim_head, heads = heads, fixed_window_size = fixed_window_size, window_size = window_size, accept_value_residual = not is_first))

            ff = norm_fn(FeedForward(dim = dim, expansion_factor = expansion_factor))

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
        return_kv_cache = False,
        condition: Tensor | None = None
    ):

        # cache and residuals

        cache = default(cache, (None,) * len(self.layers))

        next_kv_caches = []
        value_residual = None

        # handle condition

        cond_tokens = None
        if exists(condition):
            assert exists(self.to_cond_tokens)
            cond_tokens = self.to_cond_tokens(condition)

        # layers

        for (attn, ff), kv_cache in zip(self.layers, cache):

            attn_out, (next_kv_cache, values) = attn(x, cond = cond_tokens, value_residual = value_residual, kv_cache = kv_cache, return_kv_cache = True)

            x = attn_out + x
            x = ff(x, cond = cond_tokens) + x

            next_kv_caches.append(next_kv_cache)
            value_residual = default(value_residual, values)

        embed = self.norm(x)

        if not return_kv_cache:
            return embed

        next_kv_cache = stack(next_kv_caches)

        next_kv_cache = next_kv_cache[..., -self.window_size:, :]

        return embed, next_kv_cache

# class

StateToReward = Callable[[Tensor], float | Tensor]
StateCommandToReward = Callable[[Tensor, Tensor], float | Tensor]

class Locoformer(Module):
    def __init__(
        self,
        embedder: Module,
        unembedder: Module,
        transformer: dict | TransformerXL,
        discount_factor = 0.999,
        gae_lam = 0.95,
        ppo_eps_clip = 0.2,
        ppo_entropy_weight = 0.01,
        ppo_value_clip = 0.4,
        dim_value_input = None,                 # needs to be set for value network to be available
        value_network: Module = nn.Identity(),
        reward_range: tuple[float, float] | None = None,
        reward_shaping_fns: list[StateToReward | StateCommandToReward] | None = None,
        num_reward_bins = 32,
        hl_gauss_loss_kwargs = dict(),
        value_loss_weight = 0.5,
        calc_gae_kwargs: dict = dict(),
        recurrent_kv_cache = True,
        use_spo = False # simple policy optimization https://arxiv.org/abs/2401.16025 - Levine's group (PI) verified it is more stable than PPO
    ):
        super().__init__()

        if isinstance(transformer, dict):
            transformer = TransformerXL(**transformer)

        self.transformer = transformer

        self.embedder = embedder
        self.unembedder = unembedder

        self.fixed_window_size = transformer.fixed_window_size
        self.window_size = transformer.window_size

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

        # ppo related

        self.discount_factor = discount_factor
        self.gae_lam = gae_lam
        self.ppo_eps_clip = ppo_eps_clip
        self.ppo_entropy_weight = ppo_entropy_weight
        self.ppo_value_clip = ppo_value_clip
        self.value_loss_weight = value_loss_weight

        self.calc_gae_kwargs = calc_gae_kwargs

        # maybe use spo

        self.use_spo = use_spo

        # maybe recurrent kv cache, from Ding et al. https://arxiv.org/abs/2012.15688

        self.recurrent_kv_cache = recurrent_kv_cache

        # reward shaping function

        self.has_reward_shaping = exists(reward_shaping_fns)
        self.reward_shaping_fns = reward_shaping_fns

        # loss related

        self.register_buffer('zero', tensor(0.), persistent = False)

    @property
    def device(self):
        return next(self.parameters()).device

    def actor_parameters(self):
        return self.unembedder.parameters()

    def critic_parameters(self):
        if not exists(self.to_value_pred):
            return []

        return self.to_value_pred.parameters()

    def ppo(
        self,
        state,
        action,
        old_action_log_prob,
        reward,
        old_value,
        mask,
        episode_lens,
        actor_optim: Optimizer | None = None,
        critic_optim: Optimizer | None = None
    ):
        window_size = self.window_size
        total_learnable_tokens = mask.sum().item()

        seq_len = state.shape[1]
        gae_mask = einx.less('j, i -> i j', arange(seq_len, device = self.device), episode_lens)

        advantage, returns = calc_gae(reward, old_value, masks = gae_mask, lam = self.gae_lam, gamma = self.discount_factor, **self.calc_gae_kwargs)

        advantage = normalize(advantage)

        windowed_tensors = [
            t.split(window_size, dim = 1) for t in
            (
                state,
                action,
                old_action_log_prob,
                reward,
                old_value,
                mask,
                advantage,
                returns
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
            mask,
            advantage,
            returns
        ) in zip(*windowed_tensors):

            (action_logits, value_logits), cache = self.forward(state, cache = cache, detach_cache = True, return_values = True, return_raw_value_logits = True)
            entropy = calc_entropy(action_logits)

            action = rearrange(action, 'b t -> b t 1')
            log_prob = action_logits.gather(-1, action)
            log_prob = rearrange(log_prob, 'b t 1 -> b t')

            # update actor, classic clipped surrogate loss

            eps_clip = self.ppo_eps_clip
            ratio = (log_prob - old_action_log_prob).exp()

            if self.use_spo:
                actor_loss = -(ratio * advantage - (advantage.abs() * (ratio - 1.).square()) / (2 * eps_clip))
            else:
                actor_loss = -torch.min(ratio * advantage, ratio.clamp(1. - eps_clip, 1. + eps_clip) * advantage)

            actor_loss = actor_loss - self.ppo_entropy_weight * entropy

            windowed_actor_loss = actor_loss[mask].sum() / total_learnable_tokens
            windowed_actor_loss.backward(retain_graph = True)

            # update critic

            value_loss = self.hl_gauss_loss(value_logits, returns, reduction = 'none')

            value_clip = self.ppo_value_clip
            value = self.hl_gauss_loss(value_logits)

            clipped_value = old_value + (value - old_value).clamp(-value_clip, value_clip)
            clipped_value_loss = self.hl_gauss_loss(clipped_value, returns, reduction = 'none')

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

    def state_and_command_to_rewards(
        self,
        state,
        commands = None
    ) -> Tensor:

        assert self.has_reward_shaping

        rewards = []

        for fn in self.reward_shaping_fns:
            fn_num_params = num_parameters_on_function(fn)

            if fn_num_params == 1: # only state
                reward = fn(state)
            elif fn_num_params == 2: # state and command
                reward = fn(state, commands)
            else:
                raise ValueError('invalid number of arguments for reward shaping function')

            rewards.append(reward)

        # cast to Tensor if returns a float, just make it flexible for researcher

        rewards = [tensor(reward) if not is_tensor(reward) else reward for reward in rewards]

        return stack(rewards)

    def wrap_env_functions(self, env):

        def transform_output(el):
            if isinstance(el, ndarray):
                return from_numpy(el)
            elif isinstance(el, (int, bool, float)):
                return tensor(el)
            else:
                return el

        def wrapped_reset(*args, **kwargs):
            env_reset_out =  env.reset(*args, **kwargs)

            return tree_map(transform_output, env_reset_out)

        def wrapped_step(action, *args, **kwargs):

            if is_tensor(action):
                action = action.item()

            env_step_out = env.step(action, *args, **kwargs)

            env_step_out_torch = tree_map(transform_output, env_step_out)

            if not self.has_reward_shaping:
                return env_step_out_torch

            shaped_rewards = self.state_and_command_to_rewards(env_step_out_torch)

            return env_step_out_torch, shaped_rewards

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
        window_size = self.window_size

        cache = None

        def stateful_forward(
            state: Tensor,
            condition: Tensor | None = None,
            **override_kwargs
        ):
            nonlocal cache

            # handle no batch or time, for easier time rolling out against envs

            if not has_batch_dim:
                state = rearrange(state, '... -> 1 ...')

                if exists(command):
                    condition = rearrange(condition, '... -> 1 ...')

            if not has_time_dim:
                state = state.unsqueeze(state_time_dim)

                if exists(command):
                    condition = rearrange(condition, '... d -> ... 1 d')

            # forwards

            out, cache = self.forward(state, condition = condition, cache = cache, **{**kwargs, **override_kwargs})

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

    def forward(
        self,
        state: Tensor,
        cache: Cache | None = None,
        condition: Tensor | None = None,
        detach_cache = False,
        return_values = False,
        return_raw_value_logits = False
    ):

        state = state.to(self.device)

        tokens = self.embedder(state)

        # time

        time = tokens.shape[-2]

        # destruct the cache for the current timestep and the cache

        prev_kv_cache = None
        timestep_start = 0

        if exists(cache):
            timestep_start, prev_kv_cache = cache

        # an assert - make sure during training or inference, forward never gets anything that crosses the window segment boundary, to open up some possibilities with extending memory

        assert ((timestep_start % self.window_size) + time) <= self.window_size

        # attention

        embed, kv_cache = self.transformer(tokens, condition = condition, cache = prev_kv_cache, return_kv_cache = True)

        # unembed to actions - in language models this would be the next state

        action_logits = self.unembedder(embed)

        out = action_logits

        # maybe detach cache

        if detach_cache:
            kv_cache = kv_cache.detach()

        # handle returning of values

        if return_values:
            assert exists(self.to_value_pred)

            values = self.to_value_pred(embed)

            if not return_raw_value_logits:
                values = self.hl_gauss_loss(values) # converts the value logits to scalar values

            out = (out, values)

        # output and cache

        next_timestep = time + timestep_start

        # handle curtailing kv cache at the right intervals

        window_size = self.window_size

        if self.fixed_window_size or divisible_by(next_timestep, window_size * 2):
            kv_cache = kv_cache[..., -window_size:, :]

        # maybe recurrent cache - shift the kv cache from one layer above to the one below, for extending on receptive field of past

        if self.recurrent_kv_cache and divisible_by(next_timestep, window_size):
            kv_cache = torch.roll(kv_cache, shifts = -1, dims = 0)

        return out, (next_timestep, kv_cache)
