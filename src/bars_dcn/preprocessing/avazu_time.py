"""Calendar features of the BARS Avazu setup."""

from typing import Self, overload

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags


class AvazuTimeFeatures(TransformerMixin, BaseEstimator):
    """Replace the ``YYMMDDHH`` string ``hour`` by hour-of-day and add ``weekday`` and ``weekend``.

    This is FuxiCTR's ``convert_hour`` / ``convert_weekday`` / ``convert_weekend`` for Avazu:
    ``hour`` becomes ``int(x[6:8])``, ``weekday`` is ``strftime('%w')`` (Sunday = 0) and
    ``weekend`` is 1 for Saturday and Sunday, else 0. All three are returned as strings, ready
    for a string :class:`~bars_dcn.preprocessing.OrdinalEncoder`. Stateless passthrough: every
    other column (``id`` included) is untouched. Works on ``pl.DataFrame`` and ``pl.LazyFrame``.

    Parameters
    ----------
    column : the ``YYMMDDHH`` string column.

    """

    def __init__(self, column: str = "hour") -> None:
        self.column = column

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.transformer_tags.preserves_dtype = []
        return tags

    def fit(self, X: pl.DataFrame | pl.LazyFrame, y: object = None) -> Self:
        dtype = X.lazy().collect_schema().get(self.column)
        if dtype != pl.String:
            msg = f"column {self.column!r} must be a String column, got {dtype}"
            raise ValueError(msg)
        self.feature_names_in_ = np.asarray([self.column], dtype=object)
        return self

    @overload
    def transform(self, X: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame: ...

    def transform(self, X: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        stamp = pl.col(self.column)
        day = stamp.str.slice(0, 6).str.strptime(pl.Date, "%y%m%d").dt.weekday()  # Monday = 1
        return X.with_columns(
            stamp.str.slice(6, 2).cast(pl.Int8).cast(pl.String).alias(self.column),
            (day % 7).cast(pl.String).alias("weekday"),  # Sunday = 0, as strftime('%w')
            (day >= 6).cast(pl.Int8).cast(pl.String).alias("weekend"),  # noqa: PLR2004
        )
