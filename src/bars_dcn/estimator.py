"""sklearn estimator around the DCNv2 torch module."""

from collections.abc import Sequence
from typing import Self

import numpy as np
import polars as pl
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted

from bars_dcn.model import DCNv2
from bars_dcn.model.dcn import Structure
from bars_dcn.training import PREDICT_CHUNK, TrainSettings, fit_network, predict_logits

Frame = pl.DataFrame | pl.LazyFrame | np.ndarray


def to_index_array(X: Frame) -> np.ndarray:
    """Collect an encoded index block into a 2-D integer array."""
    if isinstance(X, pl.LazyFrame):
        X = X.collect(engine="streaming")
    return X.to_numpy() if isinstance(X, pl.DataFrame) else np.asarray(X)


class DCNClassifier(ClassifierMixin, BaseEstimator):
    """Binary DCNv2 classifier over an already-encoded index block (see ``OrdinalEncoder``).

    The defaults are the BARS DCNv2 ``criteo_x4`` recipe (FuxiCTR v2.2.0): parallel structure,
    3 cross layers, 5 x 1000 MLP with BatchNorm and dropout 0.1, embedding dim 16, Adam 1e-3,
    batch 10000, embedding L2 1e-5, gradient clip 10, ReduceLROnPlateau-style schedule and early
    stopping on validation AUC (see :mod:`bars_dcn.training`).

    ``X`` holds one integer column per categorical field (index 0 conventionally OOV). With a
    polars frame, ``cat_columns`` picks (and orders) the fields, so the frame may carry other
    passthrough columns; ``None`` uses every column. Field cardinalities are inferred at ``fit``
    as ``max index + 1``.

    Input contract (enforced): integer dtype, non-negative, and at predict time below the
    per-field cardinality seen in ``fit`` (a larger index would silently read another field's
    embedding row, since all fields share one table). Floats, NaN and negatives are rejected.

    ``fit(X, y, eval_set=(X_valid, y_valid))`` validates after every epoch, reduces the learning
    rate on plateaus, stops early and restores the best epoch. Without ``eval_set`` it trains
    exactly ``max_epochs`` at a fixed rate. ``device="auto"`` lets accelerate pick cuda / mps /
    cpu for training; prediction always runs on the CPU. ``history_`` has one dict per epoch.
    ``log_dir`` writes TensorBoard epoch metrics and signal-propagation plots there.
    """

    def __init__(
        self,
        *,
        cat_columns: Sequence[str] | None = None,
        embedding_dim: int = 16,
        structure: Structure = "parallel",
        num_cross_layers: int = 3,
        use_low_rank_mixture: bool = False,
        low_rank: int = 32,
        num_experts: int = 4,
        stacked_hidden_units: Sequence[int] | None = None,
        parallel_hidden_units: Sequence[int] | None = (1000, 1000, 1000, 1000, 1000),
        batch_norm: bool = True,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 10_000,
        max_epochs: int = 100,
        embedding_regularizer: float = 1e-5,
        max_grad_norm: float = 10.0,
        early_stopping_patience: int = 2,
        lr_reduce_factor: float = 0.1,
        min_lr: float = 1e-6,
        device: str = "auto",
        log_dir: str | None = None,
        random_state: int | None = None,
    ) -> None:
        self.cat_columns = cat_columns
        self.embedding_dim = embedding_dim
        self.structure = structure
        self.num_cross_layers = num_cross_layers
        self.use_low_rank_mixture = use_low_rank_mixture
        self.low_rank = low_rank
        self.num_experts = num_experts
        self.stacked_hidden_units = stacked_hidden_units
        self.parallel_hidden_units = parallel_hidden_units
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.embedding_regularizer = embedding_regularizer
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        self.lr_reduce_factor = lr_reduce_factor
        self.min_lr = min_lr
        self.device = device
        self.log_dir = log_dir
        self.random_state = random_state

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.input_tags.positive_only = True
        return tags

    def _index_block(self, X: Frame) -> np.ndarray:
        if self.cat_columns is not None:
            if not isinstance(X, pl.DataFrame | pl.LazyFrame):
                msg = "cat_columns requires a polars DataFrame or LazyFrame"
                raise ValueError(msg)
            X = X.select(self.cat_columns)
        index = to_index_array(X)
        if index.ndim != 2:  # noqa: PLR2004
            msg = f"expected a 2-D index block, got {index.ndim} dimension(s)"
            raise ValueError(msg)
        if index.dtype.kind not in "iu":
            msg = f"expected an integer index block (encode with OrdinalEncoder), got {index.dtype}"
            raise ValueError(msg)
        if index.size and index.min() < 0:
            msg = "indices must be non-negative"
            raise ValueError(msg)
        return index

    def _checked_index(self, X: Frame) -> np.ndarray:
        """Index block validated against the fitted field count and cardinalities."""
        index = self._index_block(X)
        if index.shape[1] != self.n_features_in_:
            msg = f"X has {index.shape[1]} fields, expected {self.n_features_in_}"
            raise ValueError(msg)
        too_large = np.flatnonzero(index.max(axis=0, initial=0) >= self.cardinalities_)
        if too_large.size:
            msg = (
                f"index out of range for field(s) {too_large.tolist()} (cardinalities seen in fit)"
            )
            raise ValueError(msg)
        return index

    def _binary_target(self, y: object) -> np.ndarray:
        values = np.asarray(y)
        unknown = np.setdiff1d(values, self.classes_)
        if unknown.size:
            msg = f"y contains labels not seen in fit: {unknown.tolist()}"
            raise ValueError(msg)
        return (values == self.classes_[1]).astype(np.float32)

    def fit(self, X: Frame, y: object, eval_set: tuple[Frame, object] | None = None) -> Self:
        index = self._index_block(X)
        self.classes_ = np.unique(np.asarray(y))
        if len(self.classes_) != 2:  # noqa: PLR2004
            msg = f"only binary classification is supported, got {len(self.classes_)} classes"
            raise ValueError(msg)
        target = self._binary_target(y)
        if len(target) != len(index):
            msg = f"X has {len(index)} rows but y has {len(target)}"
            raise ValueError(msg)
        self.n_features_in_ = index.shape[1]
        self.cardinalities_ = (index.max(axis=0) + 1).tolist()
        valid = None
        if eval_set is not None:
            valid = (self._checked_index(eval_set[0]), self._binary_target(eval_set[1]))

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        model = DCNv2(
            self.cardinalities_,
            embedding_dim=self.embedding_dim,
            structure=self.structure,
            num_cross_layers=self.num_cross_layers,
            use_low_rank_mixture=self.use_low_rank_mixture,
            low_rank=self.low_rank,
            num_experts=self.num_experts,
            stacked_hidden_units=self.stacked_hidden_units,
            parallel_hidden_units=self.parallel_hidden_units,
            batch_norm=self.batch_norm,
            dropout=self.dropout,
        )
        settings = TrainSettings(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            embedding_regularizer=self.embedding_regularizer,
            max_grad_norm=self.max_grad_norm,
            patience=self.early_stopping_patience,
            lr_reduce_factor=self.lr_reduce_factor,
            min_lr=self.min_lr,
            device=self.device,
            log_dir=self.log_dir,
            seed=self.random_state,
        )
        self.model_, result = fit_network(model, index, target, settings, valid)
        self.history_ = result.history
        self.best_epoch_ = result.best_epoch
        return self

    def predict_proba(self, X: Frame) -> np.ndarray:
        check_is_fitted(self)
        index = self._checked_index(X)
        positive = torch.sigmoid(torch.from_numpy(predict_logits(self.model_, index))).numpy()
        return np.stack([1 - positive, positive], axis=1)

    def predict(self, X: Frame) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


__all__ = ["PREDICT_CHUNK", "DCNClassifier", "to_index_array"]
