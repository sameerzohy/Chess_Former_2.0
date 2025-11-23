from torch.utils.data import random_split, DataLoader
import torch
from dataset import PGN_Moves

batch_size = 64
pgn_path = "pgn-datasets/final_800k_elite.pgn"

dataset = PGN_Moves(pgn_path, max_games=500, max_moves=100_000)

total_len = len(dataset)
train_len = int(0.9 * total_len)
val_len = total_len - train_len

train_set, val_set = random_split(
    dataset,
    [train_len, val_len],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)
