"""Training loop replicating FuxiCTR v2.2.0's ``BaseModel.fit`` (the BARS DCNv2 recipe).

Replicated: Adam, BCE, ``(lambda / 2) * ||W||^2`` on the embedding table added to every batch's
loss, gradient-norm clipping, shuffling every epoch, validation AUC after every epoch, learning
rate x``lr_reduce_factor`` (floored at ``min_lr``) on every non-improving epoch, early stopping
after ``patience`` consecutive non-improving epochs, and restoring the best epoch's weights.
Every piece is a parameter of :class:`TrainSettings`.
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from sklearn.metrics import log_loss, roc_auc_score
from torch.nn import functional

from bars_dcn.batches import make_loader
from bars_dcn.diagnostics import probe

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from bars_dcn.model import DCNv2

logger = logging.getLogger(__name__)

PREDICT_CHUNK = 100_000
PROBE_ROWS = 2048


@dataclass(frozen=True)
class TrainSettings:
    """Defaults are the BARS DCNv2 criteo_x4 recipe."""

    learning_rate: float = 1e-3
    batch_size: int = 10_000
    max_epochs: int = 100
    embedding_regularizer: float = 1e-5
    max_grad_norm: float = 10.0
    patience: int = 2
    lr_reduce_factor: float = 0.1
    min_lr: float = 1e-6
    min_delta: float = 1e-6
    device: str = "auto"
    num_workers: int = 0  # DataLoader worker processes gathering batches ahead of the step
    prefetch_factor: int = 2  # batches each worker keeps ready (only with num_workers > 0)
    lr_drop_epochs: tuple[int, ...] | None = None  # fixed schedule: scale the LR after these epochs
    log_every_steps: int = 500  # progress line (loss, steps/s) every this many steps; 0 disables
    log_dir: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class Block:
    """Encoded inputs of one split: field indices, float numerics (``(n, 0)`` if none), labels."""

    index: np.ndarray
    numeric: np.ndarray
    target: np.ndarray


class Decision(NamedTuple):
    improved: bool
    lr: float
    stop: bool


class PlateauStopper:
    """FuxiCTR's ``checkpoint_and_earlystop`` for a metric that is maximized.

    An epoch improves if ``metric >= best + min_delta``. Every non-improving epoch multiplies the
    learning rate by ``lr_reduce_factor`` (floored at ``min_lr``) and counts towards ``patience``;
    an improvement resets the count (the learning rate is not restored).
    """

    def __init__(
        self,
        patience: int = 2,
        lr_reduce_factor: float = 0.1,
        min_lr: float = 1e-6,
        min_delta: float = 1e-6,
    ) -> None:
        self.patience = patience
        self.lr_reduce_factor = lr_reduce_factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.best = -np.inf
        self.stalled = 0

    def update(self, metric: float, lr: float) -> Decision:
        if metric < self.best + self.min_delta:
            self.stalled += 1
            lr = max(lr * self.lr_reduce_factor, self.min_lr)
            return Decision(improved=False, lr=lr, stop=self.stalled >= self.patience)
        self.stalled = 0
        self.best = metric
        return Decision(improved=True, lr=lr, stop=False)


class TrainResult(NamedTuple):
    history: list[dict[str, float]]
    best_epoch: int | None


def embedding_penalty(model: DCNv2, regularizer: float) -> torch.Tensor:
    """``(lambda / 2) * ||W||^2`` over the embedding table (FuxiCTR's L2 regularizer)."""
    return 0.5 * regularizer * model.embedding.weight.pow(2).sum()


def make_accelerator(device: str) -> Accelerator:
    """Return a fresh ``Accelerator``; ``"auto"`` lets accelerate pick, ``"mps"`` demands MPS."""
    if device not in ("auto", "cpu", "mps"):
        msg = f"device must be 'auto', 'cpu' or 'mps', got {device!r}"
        raise ValueError(msg)
    AcceleratorState._reset_state(reset_partial_state=True)  # noqa: SLF001
    accelerator = Accelerator(cpu=device == "cpu")
    accelerator.dataloader_config.non_blocking = accelerator.device.type == "cuda"
    if device == "mps" and accelerator.device.type != "mps":
        msg = f"device='mps' requested but accelerate selected {accelerator.device}"
        raise RuntimeError(msg)
    return accelerator


def predict_logits(
    model: DCNv2, index: np.ndarray, numeric: np.ndarray | None = None
) -> np.ndarray:
    """Logits for an index block, in chunks, on the model's device (eval mode, no gradients)."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(index), PREDICT_CHUNK):
            rows = slice(start, start + PREDICT_CHUNK)
            batch = torch.from_numpy(index[rows]).long().to(device)
            extra = None if numeric is None else torch.from_numpy(numeric[rows]).to(device)
            chunks.append(model(batch, extra).cpu())
    model.train(was_training)
    return torch.cat(chunks).numpy() if chunks else np.empty(0, dtype=np.float32)


def _run_epoch(
    model: DCNv2,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    loader: DataLoader,
    settings: TrainSettings,
) -> float:
    """One pass over the (already shuffled and device-placed) batches; returns the mean loss."""
    model.train()
    total, steps = torch.zeros((), device=accelerator.device), 0
    started = time.perf_counter()
    for index, numeric, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(index.long(), numeric if numeric.shape[1] else None)
        loss = functional.binary_cross_entropy_with_logits(logits, labels)
        if settings.embedding_regularizer:
            loss = loss + embedding_penalty(model, settings.embedding_regularizer)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
        optimizer.step()
        total += loss.detach()
        steps += 1
        if settings.log_every_steps and steps % settings.log_every_steps == 0:
            elapsed = time.perf_counter() - started  # float() syncs the device, so this is honest
            logger.info(
                "step %d/%d: mean loss %.4f, %.1f steps/s",
                steps, len(loader), float(total) / steps, steps / elapsed,
            )  # fmt: skip
    return float(total) / max(steps, 1)


def _validate(model: DCNv2, valid: Block) -> tuple[float, float]:
    """Return validation AUC and log loss, computed in float64 like FuxiCTR."""
    numeric = valid.numeric if valid.numeric.shape[1] else None
    logits = predict_logits(model, valid.index, numeric)
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy().astype(np.float64)
    return (
        float(roc_auc_score(valid.target, probabilities)),
        float(log_loss(valid.target, probabilities, labels=[0, 1])),
    )


def _summary_writer(path: str) -> SummaryWriter:
    """Create a TensorBoard writer; ``tensorboard`` is a dev dependency, imported only when used."""
    from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

    return SummaryWriter(path)


class _TensorBoard:
    """Epoch scalars (``<log_dir>/metrics``) and signal-propagation runs (``spp/epoch_N``).

    The probe batch is the first ``PROBE_ROWS`` training rows, the same at every epoch; epoch 0 is
    the state before any update.
    """

    def __init__(self, log_dir: str, train: Block, device: torch.device) -> None:
        self.log_dir = log_dir
        self.writer = _summary_writer(f"{log_dir}/metrics")
        rows = slice(0, PROBE_ROWS)
        self.index = torch.from_numpy(train.index[rows]).long().to(device)
        self.target = torch.from_numpy(train.target[rows]).to(device)
        has_numeric = train.numeric.shape[1] > 0
        self.numeric = torch.from_numpy(train.numeric[rows]).to(device) if has_numeric else None

    def spp(self, model: DCNv2, epoch: int) -> None:
        with _summary_writer(f"{self.log_dir}/spp/epoch_{epoch:03d}") as writer:
            probe(model, self.index, self.target, writer, self.numeric)

    def epoch(self, model: DCNv2, record: dict[str, float]) -> None:
        epoch = int(record["epoch"])
        for key, value in record.items():
            if key != "epoch":
                self.writer.add_scalar(f"epoch/{key}", value, epoch)
        self.writer.flush()
        self.spp(model, epoch)


def _next_lr(settings: TrainSettings, epoch: int, lr: float, decision: Decision | None) -> float:
    """Return the next epoch's learning rate: plateau-triggered, or the fixed ``lr_drop_epochs``."""
    if settings.lr_drop_epochs is None:
        return lr if decision is None else decision.lr
    if epoch in settings.lr_drop_epochs:
        return max(lr * settings.lr_reduce_factor, settings.min_lr)
    return lr


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def fit_network(
    model: DCNv2,
    train: Block,
    settings: TrainSettings,
    valid: Block | None = None,
) -> tuple[DCNv2, TrainResult]:
    """Train ``model`` on ``train``; returns it on the CPU with the best weights.

    Without ``valid`` it trains exactly ``max_epochs`` (at a fixed learning rate unless
    ``lr_drop_epochs`` is set); with ``valid`` the LR drops on plateaus, or as scheduled.
    """
    accelerator = make_accelerator(settings.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    generator = torch.Generator()
    if settings.seed is not None:
        generator.manual_seed(settings.seed)
    loader = make_loader(
        train.index,
        train.numeric,
        train.target,
        settings.batch_size,
        generator,
        num_workers=settings.num_workers,
        prefetch_factor=settings.prefetch_factor,
        pin_memory=accelerator.device.type == "cuda",
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    stopper = PlateauStopper(
        settings.patience, settings.lr_reduce_factor, settings.min_lr, settings.min_delta
    )

    board = None
    if settings.log_dir:
        board = _TensorBoard(settings.log_dir, train, accelerator.device)
        board.spp(model, epoch=0)

    history: list[dict[str, float]] = []
    best_state, best_epoch = None, None
    lr = settings.learning_rate
    for epoch in range(1, settings.max_epochs + 1):
        loss = _run_epoch(model, optimizer, accelerator, loader, settings)
        record = {"epoch": float(epoch), "lr": lr, "train_loss": loss}
        stop, decision = False, None
        if valid is not None:
            record["val_auc"], record["val_logloss"] = _validate(model, valid)
            decision = stopper.update(record["val_auc"], lr)
            if decision.improved:
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stop = decision.stop
        lr = _next_lr(settings, epoch, lr, decision)
        _set_lr(optimizer, lr)
        history.append(record)
        if board is not None:
            board.epoch(model, record)
        logger.info("epoch %d: %s", epoch, {k: round(v, 6) for k, v in record.items()})
        if stop:
            break

    if board is not None:
        board.writer.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    model = accelerator.unwrap_model(model).cpu()
    accelerator.free_memory()
    return model, TrainResult(history, best_epoch)
