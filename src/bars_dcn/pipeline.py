"""Fitting a sklearn Pipeline whose last step takes a validation set."""

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

Frame = pl.DataFrame | pl.LazyFrame


def fit_pipeline(
    pipeline: Pipeline,
    X: Frame,
    y: object,
    eval_set: tuple[Frame, object] | None = None,
) -> Pipeline:
    """Fit ``pipeline`` and give its final estimator the *transformed* validation set.

    ``Pipeline.fit(..., model__eval_set=...)`` would pass ``eval_set`` untransformed. Here the
    preprocessing steps are fitted on the training data only, ``eval_set`` goes through the fitted
    steps, and the estimator is fitted with it. Frames are made lazy first, so the whole
    preprocessing stays one plan that the estimator collects once. Returns ``pipeline`` (fitted in
    place), still a plain sklearn ``Pipeline``.
    """
    *steps, (_, estimator) = pipeline.steps
    preprocessing = [step for _, step in steps if step not in (None, "passthrough")]

    train = X.lazy()
    for step in preprocessing:
        train = step.fit(train, y).transform(train)

    valid = None
    if eval_set is not None:
        valid_x, valid_y = eval_set
        valid_x = valid_x.lazy()
        for step in preprocessing:
            valid_x = step.transform(valid_x)
        valid = (valid_x, np.asarray(valid_y))

    estimator.fit(train, y, eval_set=valid)
    return pipeline
