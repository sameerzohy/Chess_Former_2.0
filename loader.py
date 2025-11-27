# loader.py

import os
import glob

CHUNK_DIR = "pgn_chunks"   # where chunk_0001.pgn ... live

def get_train_val_files(block_size: int = 50, val_per_block: int = 9):
    """
    Split PGN chunk files into train and validation according to your rule:

      - Sort all files: chunk_0001.pgn, chunk_0002.pgn, ...
      - Walk in blocks of block_size files (default 50).
      - For each block:
          ~val_per_block files (default 9) are used for validation,
          taken from the END of the block.
          The rest are used for training.

      - For the last partial block (< block_size), the number of
        validation files is scaled proportionally:
            val_count ≈ round(len(block) * val_per_block / block_size)

    Returns:
      train_files: list[str]
      val_files:   list[str]
    """
    pattern = os.path.join(CHUNK_DIR, "chunk_*.pgn")
    all_chunk_files = sorted(glob.glob(pattern))
    n_files = len(all_chunk_files)

    if n_files == 0:
        raise RuntimeError(f"No PGN chunk files found at: {pattern}")

    print(f"Total chunk files found: {n_files}")

    train_files = []
    val_files = []

    for start in range(0, n_files, block_size):
        block = all_chunk_files[start:start + block_size]
        block_len = len(block)
        if block_len == 0:
            continue

        # For full blocks, use exactly val_per_block.
        # For the last partial block, scale proportionally.
        if block_len >= block_size:
            val_count = val_per_block
        else:
            val_count = int(round(block_len * val_per_block / block_size))
            if val_count == 0 and block_len > 1:
                val_count = 1

        # Don't take all files as val
        if block_len > 1:
            val_count = min(val_count, block_len - 1)
        else:
            val_count = 0

        train_count = block_len - val_count
        block_train = block[:train_count]
        block_val = block[train_count:]

        train_files.extend(block_train)
        val_files.extend(block_val)

    print(f"Using {len(train_files)} train files and {len(val_files)} val files")
    return train_files, val_files