from typing import List, Optional, Iterator
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class DynamicBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    A dynamic batch sampler that groups samples by audio duration (in seconds) into buckets
    and yields batches of indices such that samples within a batch have more uniform lengths.

    - Uses dataset.lengths (seconds) if available; expected to be effective duration capped at CHUNK_LENGTH.
    - Supports optional per-bucket batch sizes.
    - Supports distributed training (DDP) by partitioning batches across replicas.
    """

    def __init__(
        self,
        dataset: Dataset,
        default_batch_size: int,
        boundaries: Optional[List[float]] = None,
        bucket_batch_sizes: Optional[List[int]] = None,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.lengths: List[float] = getattr(dataset, "lengths", None)
        if self.lengths is None or len(self.lengths) != len(dataset):
            raise ValueError("DynamicBatchSampler requires dataset.lengths aligned with dataset size")

        # Bucket boundaries in seconds; right-open bins via bisect_right
        self.boundaries: List[float] = (
            boundaries if boundaries is not None else [2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0, 30.0]
        )
        self.num_buckets: int = len(self.boundaries) + 1

        # Per-bucket batch sizes; if not provided, use default for all buckets
        if bucket_batch_sizes is not None:
            if len(bucket_batch_sizes) != self.num_buckets:
                raise ValueError(
                    f"bucket_batch_sizes must have length {self.num_buckets}, got {len(bucket_batch_sizes)}"
                )
            self.bucket_batch_sizes = list(bucket_batch_sizes)
        else:
            self.bucket_batch_sizes = [int(default_batch_size) for _ in range(self.num_buckets)]

        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Distributed attributes
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

        # Cache for batches to avoid repeated computation
        self._cached_batches: Optional[List[List[int]]] = None
        self._cached_epoch: int = -1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        # Invalidate cache when epoch changes
        if self._cached_epoch != self.epoch:
            self._cached_batches = None

    def _build_batches_for_all_buckets(self) -> List[List[int]]:
        # Assign indices to buckets using vectorized numpy operations
        lengths_arr = np.asarray(self.lengths)
        bucket_ids = np.searchsorted(self.boundaries, lengths_arr, side='right')

        buckets: List[List[int]] = [
            np.nonzero(bucket_ids == b)[0].tolist() for b in range(self.num_buckets)
        ]

        rnd = random.Random(self.seed + self.epoch)

        # Build micro-batches per bucket
        all_micro_batches: List[List[int]] = []
        for b_idx, indices in enumerate(buckets):
            if len(indices) == 0:
                continue
            if self.shuffle:
                rnd.shuffle(indices)

            bsz = int(self.bucket_batch_sizes[b_idx])
            # Partition into chunks of size bsz
            micro_batches = [indices[i : i + bsz] for i in range(0, len(indices), bsz)]
            
            # Drop incomplete last batch for this bucket if needed
            if self.drop_last and len(micro_batches) > 0 and len(micro_batches[-1]) < bsz:
                micro_batches = micro_batches[:-1]

            # Ensure divisibility across replicas
            if self.num_replicas > 1:
                num_batches = len(micro_batches)
                if self.drop_last:
                    # drop tail so divisible
                    num_batches = (num_batches // self.num_replicas) * self.num_replicas
                    micro_batches = micro_batches[:num_batches]
                else:
                    # pad by cycling to make divisible
                    remainder = num_batches % self.num_replicas
                    if remainder != 0:
                        needed = self.num_replicas - remainder
                        # cycle from start
                        for i in range(needed):
                            micro_batches.append(micro_batches[i % len(micro_batches)])

            # Now select micro-batches for this rank
            if self.num_replicas > 1:
                micro_batches = micro_batches[self.rank :: self.num_replicas]

            all_micro_batches.extend(micro_batches)

        # Optionally shuffle the order of micro-batches across buckets
        if self.shuffle:
            rnd.shuffle(all_micro_batches)
        
        return all_micro_batches

    def _get_batches(self) -> List[List[int]]:
        """Get batches, using cache if available for current epoch."""
        if self._cached_batches is None or self._cached_epoch != self.epoch:
            self._cached_batches = self._build_batches_for_all_buckets()
            self._cached_epoch = self.epoch
        return self._cached_batches

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self._get_batches():
            yield batch

    def __len__(self) -> int:
        return len(self._get_batches())