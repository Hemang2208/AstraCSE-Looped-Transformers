"""AstraCSE-100M dataloader — Architecture.md §12–13.

Reads packed uint16 memmap shards from data/shards/{train,val}/.
Returns (input_ids, target_ids) tensors for next-token-prediction.

Tracks shard_path (stable ID) + intra-shard offset for checkpoint/resume
(§17.1). The state stores the shard *filename*, not its epoch-position, so
resuming with shuffle=True still lands on the correct shard even though the
epoch order is re-randomised.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset


SEQ_LEN = 1024


@dataclass
class DataloaderState:
    """Serialisable state for checkpoint/resume (Architecture.md §17.1).

    shard_name: filename of the shard being read (stable across shuffles).
    offset:     token offset within that shard where we left off.
    """
    shard_name: str = ""
    offset: int = 0

    def to_dict(self) -> dict:
        return {"shard_name": self.shard_name, "offset": self.offset}

    @classmethod
    def from_dict(cls, d: dict) -> DataloaderState:
        # Back-compat: old checkpoints stored shard_index (int) instead of
        # shard_name. If shard_name is absent we cannot resume precisely —
        # start of epoch is the safest fallback.
        return cls(
            shard_name=d.get("shard_name", ""),
            offset=d.get("offset", 0),
        )


class ShardedDataset(IterableDataset):
    """Iterates over packed uint16 shards, yielding (input_ids, target_ids) pairs.

    Shuffle behaviour: shard order is randomised each epoch (train split).
    Resume behaviour: the saved shard is found by *filename*, so the resume
    position is stable even when shuffle re-orders the epoch.
    """

    def __init__(
        self,
        shards_dir: Path,
        seq_len: int = SEQ_LEN,
        shuffle: bool = True,
        resume_state: Optional[DataloaderState] = None,
        seed: int = 0,
    ):
        self.shards_dir = Path(shards_dir)
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.seed = seed

        self.shard_paths = sorted(self.shards_dir.glob("*.bin"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No .bin shards found in {shards_dir}")

        self.state = resume_state or DataloaderState()

    def _iter_from_state(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        rng = random.Random(self.seed)
        shard_order = list(self.shard_paths)
        if self.shuffle:
            rng.shuffle(shard_order)

        # Locate resume shard by filename, not by epoch position
        resume_name = self.state.shard_name
        start_offset = self.state.offset if resume_name else 0
        resume_found = not resume_name  # if no resume, start immediately

        for shard_path in shard_order:
            if not resume_found:
                if shard_path.name == resume_name:
                    resume_found = True
                else:
                    continue  # skip shards before the resume point

            tokens = np.memmap(str(shard_path), dtype=np.uint16, mode="r")
            pos = start_offset if shard_path.name == resume_name else 0
            start_offset = 0  # only apply offset on the first (resume) shard

            while pos + self.seq_len + 1 <= len(tokens):
                chunk = tokens[pos: pos + self.seq_len + 1]
                ids = torch.from_numpy(chunk.astype(np.int64))
                input_ids = ids[:-1]
                target_ids = ids[1:]

                # Update state BEFORE yielding so a checkpoint taken mid-batch
                # captures where to continue from next.
                self.state.shard_name = shard_path.name
                self.state.offset = pos + self.seq_len

                yield input_ids, target_ids
                pos += self.seq_len

        # End of epoch — reset state so next epoch starts fresh
        self.state = DataloaderState()

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        yield from self._iter_from_state()


def make_dataloader(
    shards_dir: Path,
    batch_size: int,
    seq_len: int = SEQ_LEN,
    shuffle: bool = True,
    num_workers: int = 0,
    resume_state: Optional[DataloaderState] = None,
    seed: int = 0,
) -> tuple[torch.utils.data.DataLoader, ShardedDataset]:
    dataset = ShardedDataset(shards_dir, seq_len=seq_len, shuffle=shuffle,
                             resume_state=resume_state, seed=seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset


def count_tokens(shards_dir: Path) -> int:
    """Count total tokens across all shards in a directory."""
    total = 0
    for p in sorted(Path(shards_dir).glob("*.bin")):
        arr = np.memmap(str(p), dtype=np.uint16, mode="r")
        total += len(arr)
    return total
