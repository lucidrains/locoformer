# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "discrete_continuous_embed_readout",
#     "einops",
#     "fire",
#     "locoformer",
#     "numpy",
#     "torch",
#     "torch-einops-utils",
# ]
#
# [tool.uv.sources]
# locoformer = { path = ".", editable = true }
# ///

import gzip
from math import ceil

import numpy as np

import torch
from torch import from_numpy
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from einops import rearrange
from accelerate import Accelerator
from fire import Fire
from torch_einops_utils.shape import size

from discrete_continuous_embed_readout import EmbedAndReadout

from locoformer.locoformer import Locoformer

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def divisible_by(num, den):
    return (num % den) == 0

def decode_token(token):
    return chr(max(32, token))

def decode_tokens(tokens):
    return ''.join(map(decode_token, tokens.cpu().tolist()))

def topk_logits_filter(logits, frac_num_tokens = 0.1):
    num_tokens = size(logits, '... [l]')
    k = ceil(frac_num_tokens * num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, segments):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.segments = segments
        self.total_len = seq_len * segments

    def __getitem__(self, index):
        rand_start = torch.randint(0, size(self.data, '[b] ...') - self.total_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.total_len + 1].long()
        return full_seq

    def __len__(self):
        return size(self.data, '[b] ...') // self.total_len

@torch.inference_mode()
def generate(model, prime, generate_length, topk_frac):
    model.eval()

    prime = prime.to(model.device)
    prime_len = size(prime, '[l]')
    out = prime

    stateful_forward, logits = model.get_stateful_forward(
        has_batch_dim = False,
        has_time_dim = True,
        initial_states = prime
    )

    while size(out, '... [l]') < prime_len + generate_length:
        filtered_logits = topk_logits_filter(logits[-1], topk_frac)
        sampled = model.unembedder.sample(filtered_logits)
        sampled = rearrange(sampled, '... -> ... 1')

        out = torch.cat((out, sampled), dim = -1)
        logits = stateful_forward(sampled)

    return prime, out[prime_len:]

# main

def main(
    data_path: str = './data/enwik8.gz',
    num_batches: int = int(1e5),
    batch_size: int = 16,
    grad_accum_every: int = 1,
    learning_rate: float = 2e-4,
    generate_every: int = 250,
    prime_length: int = 32,
    generate_length: int = 1024,
    topk_frac: float = 0.1,
    seq_len: int = 256,
    num_segments: int = 4,
    fixed_window_size: bool = False,
    dim_model: int = 512,
    depth: int = 6,
    max_mem_segments: int = 2,
    long_term_mem_layers: tuple[int, ...] = (3, 4),
    num_residual_streams: int = 1,
    seed: int = 42,
    cpu: bool = False
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # instantiate model

    embed, readout = EmbedAndReadout(dim_model, num_discrete = 256)

    model = Locoformer(
        embedder = embed,
        unembedder = readout,
        max_mem_segments = max_mem_segments,
        transformer = dict(
            dim = dim_model,
            depth = depth,
            fixed_window_size = fixed_window_size,
            window_size = seq_len,
            long_term_mem_layers = long_term_mem_layers,
            num_residual_streams = num_residual_streams
        )
    )

    # prepare enwik8 data

    with gzip.open(data_path) as file:
        data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
        train_data, valid_data = np.split(data, [int(90e6)])
        data_train, data_val = from_numpy(train_data), from_numpy(valid_data)

    train_dataset = TextSamplerDataset(data_train, seq_len, num_segments)
    val_dataset   = TextSamplerDataset(data_val, seq_len, num_segments)
    train_loader  = DataLoader(train_dataset, batch_size = batch_size)
    val_loader    = DataLoader(val_dataset, batch_size = batch_size)

    # optimizer

    optim = Adam(model.parameters(), lr = learning_rate)

    # prepare accelerate

    accelerator = Accelerator(cpu = cpu)

    accelerator.print(f'using device: {accelerator.device}')

    model, optim, train_loader, val_loader = accelerator.prepare(model, optim, train_loader, val_loader)

    # training loop

    train_loader_iter = cycle(train_loader)
    val_loader_iter = cycle(val_loader)

    loss_scale = 1. / (num_segments * grad_accum_every)

    for i in range(num_batches):
        model.train()

        total_loss = 0.

        for _ in range(grad_accum_every):
            seq = next(train_loader_iter)
            seq, labels = seq[:, :-1], seq[:, 1:]

            cache = None

            for segment_seq, segment_labels in zip(seq.chunk(num_segments, dim = -1), labels.chunk(num_segments, dim = -1)):

                logits, cache = model(
                    segment_seq,
                    cache = cache,
                    detach_cache = True
                )

                loss = F.cross_entropy(
                    rearrange(logits, 'b n l -> b l n'),
                    segment_labels
                )

                accelerator.backward(loss * loss_scale)
                total_loss += loss.item() * loss_scale

        optim.step()
        optim.zero_grad()

        accelerator.print(f'[{i}] loss: {total_loss:.3f}')

        if divisible_by(i, generate_every) and accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)

            val_seq = next(val_loader_iter)
            prime = val_seq[0, :prime_length]

            prime, generated = generate(unwrapped_model, prime, generate_length, topk_frac)

            decoded_prime = decode_tokens(prime)
            decoded_string = decode_tokens(generated)

            accelerator.print(f'\n\n[prime]: {decoded_prime}\n\n')
            accelerator.print('*' * 100)
            accelerator.print(f'\n\n [generated]: {decoded_string}\n\n')

if __name__ == '__main__':
    Fire(main)
