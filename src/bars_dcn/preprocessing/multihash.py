"""Multi-hash trick over ordinal indices: several cheap integer hashes into one shared space."""

from collections.abc import Sequence
from typing import Self, overload

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags

MAX_MULTIPLIER = 2**31  # keeps ``multiplier * index`` inside int64 for any int32 index


def hash_name(column: str, k: int) -> str:
    """Name of the ``k``-th hash column of ``column``."""
    return f"{column}_h{k}"


class MultiHashEncoder(TransformerMixin, BaseEstimator):
    """Append ``n_hashes`` int32 columns per index column: ``(a * index + b) mod cardinality``.

    The input columns hold ordinal indices (see ``OrdinalEncoder``). Every column and hash gets its
    own random ``(a, b)``, drawn once at ``fit`` from ``random_state``, so two fields never collide
    in the same way. All outputs live in ``[0, cardinality)``, i.e. they index one embedding table
    shared by every position (``DCNClassifier(shared_embedding=True)``); a field is then
    represented by the concatenation of its ``n_hashes`` embeddings. The arithmetic is integer
    only, so it exports to ONNX as ``Mul``/``Add``/``Mod``.

    The source columns are kept (additive). Works on ``pl.DataFrame`` and ``pl.LazyFrame``.

    Parameters
    ----------
    columns : integer index columns to hash; ``None`` hashes every column of the ``fit`` frame.
    n_hashes : hash functions (probes) per column.
    cardinality : size of the shared hash space (the embedding table's rows).
    random_state : seed for the hash constants.

    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        n_hashes: int = 2,
        cardinality: int = 100_000,
        random_state: int = 0,
    ) -> None:
        self.columns = columns
        self.n_hashes = n_hashes
        self.cardinality = cardinality
        self.random_state = random_state

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.transformer_tags.preserves_dtype = []  # indices in, hashed int32 appended
        return tags

    def fit(self, X: pl.DataFrame | pl.LazyFrame, y: object = None) -> Self:
        if self.n_hashes < 1:
            msg = f"n_hashes must be at least 1, got {self.n_hashes}"
            raise ValueError(msg)
        if self.cardinality < 1:
            msg = f"cardinality must be at least 1, got {self.cardinality}"
            raise ValueError(msg)
        schema = X.lazy().collect_schema()
        columns = list(self.columns) if self.columns is not None else list(schema.names())
        not_integer = [column for column in columns if not schema[column].is_integer()]
        if not_integer:
            msg = f"columns {not_integer} are not integer indices"
            raise ValueError(msg)
        rng = np.random.default_rng(self.random_state)
        shape = (len(columns), self.n_hashes)
        self.columns_ = columns
        self.multipliers_ = rng.integers(1, MAX_MULTIPLIER, size=shape)
        self.offsets_ = rng.integers(0, self.cardinality, size=shape)
        self.feature_names_in_ = np.asarray(columns, dtype=object)
        return self

    @overload
    def transform(self, X: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame: ...

    def transform(self, X: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        hashed = [
            (
                (
                    pl.col(column).cast(pl.Int64) * int(self.multipliers_[i, k])
                    + int(self.offsets_[i, k])
                )
                % self.cardinality
            )
            .cast(pl.Int32)
            .alias(hash_name(column, k))
            for i, column in enumerate(self.columns_)
            for k in range(self.n_hashes)
        ]
        return X.with_columns(hashed)
