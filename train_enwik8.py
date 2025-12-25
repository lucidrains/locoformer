# /// script
# dependencies = [
#   'accelerate',
#   'discrete_continuous_embed_readout',
#   'locoformer',
#   'tqdm'
# ]
# ///

import tqdm
import gzip
from math import ceil
import numpy as np

import torch
from torch import nn
from torch import from_numpy
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from einops import rearrange
from accelerate import Accelerator

from discrete_continuous_embed_readout import EmbedAndReadout

from locoformer.locoformer import Locoformer

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100

GENERATE_EVERY  = 250
PRIME_LENGTH    = 32
GENERATE_LENGTH = 1024

SEQ_LEN = 256
NUM_SEGMENTS = 4
FIXED_WINDOW_SIZE = False

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def divisible_by(num, den):
    return (num % den) == 0

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def topk_logits_filter(logits, frac_num_tokens = 0.1):
    num_tokens = logits.shape[-1]
    k = ceil(frac_num_tokens * num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

# instantiate model

dim_model = 512

embed, readout = EmbedAndReadout(dim_model, num_discrete = 256)

model = Locoformer(
    embedder = embed,
    unembedder = readout,
    transformer = dict(
        dim = dim_model,
        depth = 6,
        fixed_window_size = FIXED_WINDOW_SIZE,
        window_size = SEQ_LEN,
        long_term_mem_layers = (3, 4)
    )
)

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
    train_data, valid_data = np.split(data, [int(90e6)])
    data_train, data_val = from_numpy(train_data), from_numpy(valid_data)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, segments):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.segments = segments
        self.total_len = seq_len * segments

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.total_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.total_len + 1].long()
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.total_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN, NUM_SEGMENTS)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN, NUM_SEGMENTS)
train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE)
val_loader    = DataLoader(val_dataset, batch_size = BATCH_SIZE)

# optimizer

optim = Adam(model.parameters(), lr = LEARNING_RATE)

# prepare accelerate

accelerate = Accelerator()

model, optim, train_loader = accelerate.prepare(model, optim, train_loader)

# training loop

train_loader_iter = cycle(train_loader)
val_loader_iter = cycle(val_loader)

for i in range(NUM_BATCHES):
    model.train()

    seq = next(train_loader_iter)
    seq, labels = seq[:, :-1], seq[:, 1:]

    cache = None

    for segment_seq, segment_labels in zip(seq.chunk(NUM_SEGMENTS, dim = -1), labels.chunk(NUM_SEGMENTS, dim = -1)):

        logits, cache = model(
            segment_seq,
            cache = cache,
            detach_cache = True
        )

        loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            segment_labels
        )

        accelerate.backward(loss / NUM_SEGMENTS)
        accelerate.print(f'[{i}] loss: {loss.item():.3f}')

        optim.step()
        optim.zero_grad()

    if divisible_by(i, GENERATE_EVERY):
        model.eval()

        val_seq = next(val_loader_iter)
        prime = val_seq[0, :PRIME_LENGTH]

        prime = prime.to(model.device)
        out = prime

        stateful_forward, logits = model.get_stateful_forward(has_batch_dim = False, has_time_dim = True, initial_states = prime, inference_mode = True)

        # sample

        while out.shape[-1] < GENERATE_LENGTH:
            filtered_logits = topk_logits_filter(logits[-1])

            sampled = model.unembedder.sample(filtered_logits)

            sampled = rearrange(sampled, '... -> ... 1')

            out = torch.cat((out, sampled), dim = -1)

            logits = stateful_forward(sampled)

        # decoded

        decoded_prime = decode_tokens(prime.cpu())
        decoded_string = decode_tokens(out[PRIME_LENGTH:].cpu())

        print(f'\n\n[prime]: {decoded_prime}\n\n')
        print('*' * 100)
        print(f'\n\n [generated]: {decoded_string}\n\n')
