# /// script
# dependencies = [
#   'accelerate',
#   'locoformer',
#   'tqdm'
# ]
# ///

import random
import tqdm
import gzip
import numpy as np

import torch
from torch import nn
from torch import from_numpy
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from einops import rearrange
from accelerate import Accelerator

from locoformer.locoformer import Locoformer

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100

GENERATE_EVERY  = 500
PRIME_LENGTH    = 512
GENERATE_LENGTH = 1024

SEQ_LEN = 512
NUM_SEGMENTS = 4

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate model

dim_model = 512

model = Locoformer(
    embedder = nn.Embedding(256, dim_model),
    unembedder = nn.Linear(dim_model, 256, bias = False),
    transformer = dict(
        dim = dim_model,
        depth = 6
    )
)

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    data = np.fromstring(file.read(int(95e6)), dtype = np.uint8)
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

for _ in range(NUM_BATCHES):

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
        accelerate.print(f'loss: {loss.item():.3f}')

        optim.step()
        optim.zero_grad()
