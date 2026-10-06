"""Piecewise-linear encoding of numeric columns (Gorishniy et al., 2022)."""

from collections.abc import Sequence
from itertools import pairwise
from typing import Self, overload

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags


def ple_name(column: str, k: int) -> str:
    """Name of the ``k``-th bin column of ``column``."""
    return f"{column}_ple{k}"


class PiecewiseLinearEncoder(TransformerMixin, BaseEstimator):
    """Append float32 bin columns per numeric column: ``clip((x - lo_k) / (hi_k - lo_k), 0, 1)``.

    The bin edges are the ``n_bins + 1`` quantiles of the (filled) training column, de-duplicated,
    so a column with few distinct values gets fewer bins (``n_bins_``; a constant column gets
    none). Bin ``k`` is 0 below its lower edge, 1 above its upper edge and linear in between, so
    a value is encoded as a staircase of ones followed by one fractional bin.

    Missing values (null or NaN) are first replaced by ``fill_value``. The source columns are kept
    (additive, unlike the bucketizer), so one frame can feed both the bucketized embeddings and the
    float numeric block. Works on ``pl.DataFrame`` and ``pl.LazyFrame`` and returns the same kind.

    Parameters
    ----------
    columns : numeric columns to encode; ``None`` encodes every column of the ``fit`` frame.
    n_bins : number of quantile bins per column (at most).
    fill_value : replaces missing values before fitting and encoding.

    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        n_bins: int = 16,
        fill_value: float = 0,
    ) -> None:
        self.columns = columns
        self.n_bins = n_bins
        self.fill_value = fill_value

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.transformer_tags.preserves_dtype = []  # numbers in, float32 bins appended
        return tags

    def _filled(self, column: str) -> pl.Expr:
        return pl.col(column).cast(pl.Float64).fill_nan(None).fill_null(self.fill_value)

    def fit(self, X: pl.DataFrame | pl.LazyFrame, y: object = None) -> Self:
        if self.n_bins < 1:
            msg = f"n_bins must be at least 1, got {self.n_bins}"
            raise ValueError(msg)
        lazy = X.lazy()
        schema = lazy.collect_schema()
        columns = list(self.columns) if self.columns is not None else list(schema.names())
        not_numeric = [column for column in columns if not schema[column].is_numeric()]
        if not_numeric:
            msg = f"columns {not_numeric} are not numeric"
            raise ValueError(msg)
        levels = np.linspace(0, 1, self.n_bins + 1)
        quantiles = lazy.select(
            self._filled(column).quantile(float(q)).alias(f"{column}\0{i}")
            for column in columns
            for i, q in enumerate(levels)
        ).collect(engine="streaming")
        self.columns_ = columns
        self.edges_ = [
            np.unique([quantiles[f"{column}\0{i}"].item() for i in range(len(levels))])
            for column in columns
        ]
        self.n_bins_ = [max(len(edges) - 1, 0) for edges in self.edges_]
        self.feature_names_in_ = np.asarray(columns, dtype=object)
        return self

    @overload
    def transform(self, X: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame: ...

    def transform(self, X: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        bins = [
            ((self._filled(column) - lo) / (hi - lo))
            .clip(0, 1)
            .cast(pl.Float32)
            .alias(ple_name(column, k))
            for column, edges in zip(self.columns_, self.edges_, strict=True)
            for k, (lo, hi) in enumerate(pairwise(edges))
        ]
        return X.with_columns(bins)
