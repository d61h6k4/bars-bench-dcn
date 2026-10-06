"""Replace missing numeric values by a constant."""

from collections.abc import Sequence
from typing import Self, overload

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags


class MissingFiller(TransformerMixin, BaseEstimator):
    """Replace null and NaN in numeric columns by ``fill_value``; the columns become float64.

    Passthrough: only ``columns`` are replaced (in place). Works on ``pl.DataFrame`` and
    ``pl.LazyFrame`` and returns the same kind.

    Parameters
    ----------
    columns : numeric columns to fill; ``None`` fills every column of the ``fit`` frame.
    fill_value : the replacement.

    """

    def __init__(self, columns: Sequence[str] | None = None, fill_value: float = 0) -> None:
        self.columns = columns
        self.fill_value = fill_value

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.transformer_tags.preserves_dtype = []  # numbers in, float64 out
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
        return X.with_columns(
            pl.col(column).cast(pl.Float64).fill_nan(None).fill_null(self.fill_value)
            for column in self.columns_
        )
