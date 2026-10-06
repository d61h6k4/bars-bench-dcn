"""Training batches through a ``torch.utils.data.DataLoader`` (workers, prefetch, pinned memory).

The batch sampler owns the shuffle: it yields the row numbers of one batch, drawn with
``torch.randperm`` from a seeded generator in the main process, so the row order depends only on
the seed and never on ``num_workers``. The dataset just gathers those rows (``__getitems__``),
which is the work the workers run ahead of the training step. Device placement is left to
``Accelerator.prepare``, which keeps a batch sampler as it is.
"""

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

MIN_BATCH_ROWS = 2  # BatchNorm cannot train on a single row


class RowGather(Dataset):
    """Index block and labels; ``__getitems__`` gathers a whole batch of rows in one call."""

    def __init__(
        self, index: np.ndarray, target: np.ndarray, *, share_memory: bool = False
    ) -> None:
        self.index = torch.from_numpy(index)
        self.target = torch.from_numpy(target)
        if share_memory:  # worker processes then map the data instead of receiving a copy
            self.index.share_memory_()
            self.target.share_memory_()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.index[index], self.target[index]

    def __getitems__(self, rows: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        rows_tensor = torch.as_tensor(rows)
        return self.index[rows_tensor], self.target[rows_tensor]


class ShuffledBatches(Sampler[list[int]]):
    """A fresh seeded permutation per epoch, cut into batches; a trailing 1-row batch is dropped."""

    def __init__(self, n_rows: int, batch_size: int, generator: torch.Generator) -> None:
        super().__init__()
        self.n_rows = n_rows
        self.batch_size = batch_size
        self.generator = generator

    def __iter__(self) -> Iterator[list[int]]:
        order = torch.randperm(self.n_rows, generator=self.generator)
        for start in range(0, self.n_rows, self.batch_size):
            rows = order[start : start + self.batch_size]
            if len(rows) >= MIN_BATCH_ROWS:
                yield rows.tolist()

    def __len__(self) -> int:
        full, rest = divmod(self.n_rows, self.batch_size)
        return full + (rest >= MIN_BATCH_ROWS)


def _whole_batch(batch: Any) -> Any:
    return batch  # ``__getitems__`` already returns the collated batch


def make_loader(
    index: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    generator: torch.Generator,
    *,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
) -> DataLoader:
    """Shuffled training batches of ``(index_block, labels)`` as CPU tensors.

    ``num_workers=0`` gathers in the training process; with workers the gather runs
    ``prefetch_factor`` batches ahead of the step. ``pin_memory`` speeds up host-to-GPU copies
    (CUDA).
    """
    dataset = RowGather(index, target, share_memory=num_workers > 0)
    return DataLoader(
        dataset,
        batch_sampler=ShuffledBatches(len(dataset), batch_size, generator),
        collate_fn=_whole_batch,
        # Worker start-up draws a base seed; from a private generator it would otherwise come from
        # the global RNG and shift the dropout masks, making results depend on num_workers.
        generator=torch.Generator(),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
    )
