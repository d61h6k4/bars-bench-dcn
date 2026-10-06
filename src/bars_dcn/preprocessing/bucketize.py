"""Stateless numeric bucketing used by the BARS Criteo setup."""

from collections.abc import Sequence
from typing import Self, overload

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags


class LogSquaredBucketizer(TransformerMixin, BaseEstimator):
    """``x > threshold ? floor(ln(x)^2) : int(x)``, with missing values first set to ``fill_value``.

    This is the FuxiCTR ``convert_to_bucket`` used for the BARS Criteo numerics (the dataset
    README's ``floor(log2)`` description is wrong; see ARCHITECTURE.md). Output is ``int32``.

    Passthrough: only ``columns`` are replaced (in place); other columns are untouched. Works on
    ``pl.DataFrame`` and ``pl.LazyFrame`` and returns the same kind.

    Parameters
    ----------
    columns : numeric columns to bucket; ``None`` buckets every column of the ``fit`` frame.
    threshold : values above it are log-squared, the rest are truncated to an integer.
    fill_value : replaces missing values before bucketing.

    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        threshold: float = 2,
        fill_value: float = 0,
    ) -> None:
        self.columns = columns
        self.threshold = threshold
        self.fill_value = fill_value

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.transformer_tags.preserves_dtype = []  # numbers in, int32 out
        return tags

    def fit(self, X: pl.DataFrame | pl.LazyFrame, y: object = None) -> Self:
        schema = X.lazy().collect_schema()
        columns = list(self.columns) if self.columns is not None else list(schema.names())
        not_numeric = [column for column in columns if not schema[column].is_numeric()]
        if not_numeric:
            msg = f"columns {not_numeric} are not numeric"
            raise ValueError(msg)
        self.columns_ = columns
        self.feature_names_in_ = np.asarray(columns, dtype=object)
        return self

    @overload
    def transform(self, X: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame: ...

    def transform(self, X: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        def bucket(column: str) -> pl.Expr:
            value = pl.col(column).fill_null(self.fill_value)
            return (
                pl.when(value > self.threshold)
                .then((value.log() ** 2).floor())
                .otherwise(value)
                .cast(pl.Int32)
            )

        return X.with_columns(bucket(column) for column in self.columns_)
