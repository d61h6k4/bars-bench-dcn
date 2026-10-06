"""Ordinal encoding of string or integer categoricals with a min-count vocabulary."""

from collections.abc import Sequence
from typing import Self, overload

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags

OOV_INDEX = 0


class OrdinalEncoder(TransformerMixin, BaseEstimator):
    """Map categorical columns to integer indices, leaving every other column untouched.

    All encoded columns must be of one kind: strings or integers. Per column, the vocabulary is
    every non-missing value seen at least ``min_count`` times in ``fit``, ordered by descending
    count (ties by value) and indexed from 1. Missing values (null, ``""`` for strings, and any
    of ``na_values``), rare values and values unseen in ``fit`` all map to the single OOV index 0.

    Passthrough: ``transform`` replaces the encoded columns in place (as ``int32``) and keeps all
    other columns, so it composes in a plain ``Pipeline``. It works on ``pl.DataFrame`` and
    ``pl.LazyFrame`` and returns the same kind; on a ``LazyFrame`` nothing runs until ``collect``.
    The lookup is precompiled at ``fit`` (``dtypes_``), so calling ``transform`` on small batches
    is cheap.

    Parameters
    ----------
    columns : columns to encode; ``None`` encodes every column of the frame given to ``fit``.
    min_count : minimum number of occurrences for a value to enter the vocabulary.
    na_values : extra values to treat as missing (excluded from the vocabulary), of the columns'
        kind. E.g. ``[0]`` for bucketized numerics that use 0 as the fill value.

    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        min_count: int = 10,
        na_values: Sequence[int | str] = (),
    ) -> None:
        self.columns = columns
        self.min_count = min_count
        self.na_values = na_values

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        tags.transformer_tags.preserves_dtype = []  # strings or ints in, int32 out
        return tags

    def fit(self, X: pl.DataFrame | pl.LazyFrame, y: object = None) -> Self:
        lazy = X.lazy()
        schema = lazy.collect_schema()
        columns = list(self.columns) if self.columns is not None else list(schema.names())
        dtypes = {schema[column] for column in columns}
        if all(dtype == pl.String for dtype in dtypes):
            kind = "string"
        elif all(dtype.is_integer() for dtype in dtypes):
            kind = "integer"
        else:
            msg = f"columns must be all String or all integer, got {sorted(map(str, dtypes))}"
            raise ValueError(msg)
        na_type = str if kind == "string" else int
        if not all(isinstance(value, na_type) for value in self.na_values):
            msg = f"na_values must be {na_type.__name__} for {kind} columns, got {self.na_values}"
            raise ValueError(msg)

        self.kind_ = kind
        self.columns_ = columns
        self.categories_ = [self._vocabulary(lazy, column) for column in columns]
        # Built once: a pl.Enum carries its lookup table, so transform never rebuilds it.
        self.dtypes_ = [
            pl.Enum([str(value) for value in categories]) for categories in self.categories_
        ]
        self.feature_names_in_ = np.asarray(columns, dtype=object)
        return self

    def _vocabulary(self, lazy: pl.LazyFrame, column: str) -> list[str] | list[int]:
        values = pl.col(column)
        missing = values.is_null()
        if self.kind_ == "string":
            missing |= values == ""
        if self.na_values:
            missing |= values.is_in(list(self.na_values))
        return (
            lazy.filter(~missing)
            .group_by(column)
            .len()
            .filter(pl.col("len") >= self.min_count)
            .sort(["len", column], descending=[True, False])
            .select(column)
            .collect(engine="streaming")
            .get_column(column)
            .to_list()
        )

    @property
    def cardinalities_(self) -> list[int]:
        """Number of indices per column, OOV included."""
        return [len(categories) + 1 for categories in self.categories_]

    @overload
    def transform(self, X: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame: ...

    def transform(self, X: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        # Values outside the vocabulary (missing, rare, unseen) cast to null; index = position + 1.
        return X.with_columns(
            (self._as_text(column).cast(dtype, strict=False).to_physical() + 1)
            .fill_null(OOV_INDEX)
            .cast(pl.Int32)
            for column, dtype in zip(self.columns_, self.dtypes_, strict=True)
        )

    def _as_text(self, column: str) -> pl.Expr:
        values = pl.col(column)
        return values.cast(pl.String) if self.kind_ == "integer" else values
