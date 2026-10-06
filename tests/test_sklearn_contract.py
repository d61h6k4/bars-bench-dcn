"""Targeted sklearn contract for every sklearn component in the package.

Not sklearn's ``check_estimator``: these components take polars frames or encoded integer
blocks, which that suite cannot generate. Instead each component is checked against the parts of
the contract that pipelines, cloning and serving rely on. A guard fails if a new sklearn
component is added to the package without being registered here.
"""

import copy
import importlib
import pickle
import pkgutil
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils import get_tags
from sklearn.utils.validation import check_is_fitted

import bars_dcn
from bars_dcn.estimator import DCNClassifier
from bars_dcn.preprocessing import (
    AvazuTimeFeatures,
    LogSquaredBucketizer,
    MultiHashEncoder,
    OrdinalEncoder,
    PiecewiseLinearEncoder,
)


@dataclass
class Case:
    name: str
    make: Any  # () -> unfitted estimator
    data: Any  # () -> (X, y)
    output: Any  # (fitted, X) -> comparable output


def _strings():
    frame = pl.DataFrame({"a": ["x", "x", "y", "y", "z", None], "b": ["p", "q", "p", "q", "p", ""]})
    return frame, None


def _numbers():
    return pl.DataFrame({"n": [0, 1, 2, 3, 10, None], "m": [5, 5, 100, 1, 0, 7]}), None


def _indices():
    rng = np.random.default_rng(0)
    return rng.integers(0, [4, 6, 3], size=(40, 3)), rng.integers(0, 2, size=40)


def _hours():
    return pl.DataFrame({"hour": ["14102100", "14102523", "14102612"], "x": [1, 2, 3]}), None


CASES = [
    Case(
        "AvazuTimeFeatures",
        AvazuTimeFeatures,
        _hours,
        lambda est, x: est.transform(x).to_dict(as_series=False),
    ),
    Case(
        "OrdinalEncoder",
        lambda: OrdinalEncoder(min_count=1),
        _strings,
        lambda est, x: est.transform(x).to_dict(as_series=False),
    ),
    Case(
        "LogSquaredBucketizer",
        LogSquaredBucketizer,
        _numbers,
        lambda est, x: est.transform(x).to_dict(as_series=False),
    ),
    Case(
        "MultiHashEncoder",
        lambda: MultiHashEncoder(n_hashes=2, cardinality=7),
        lambda: (pl.DataFrame({"a": [0, 1, 2, 3, 4], "b": [4, 3, 2, 1, 0]}), None),
        lambda est, x: est.transform(x).to_dict(as_series=False),
    ),
    Case(
        "PiecewiseLinearEncoder",
        lambda: PiecewiseLinearEncoder(n_bins=3),
        _numbers,
        lambda est, x: est.transform(x).to_dict(as_series=False),
    ),
    Case(
        "DCNClassifier",
        lambda: DCNClassifier(
            embedding_dim=4,
            parallel_hidden_units=[8],
            max_epochs=2,
            batch_size=16,
            device="cpu",
            random_state=0,
        ),
        _indices,
        lambda est, x: est.predict_proba(x),
    ),
]


@pytest.fixture(params=CASES, ids=lambda case: case.name)
def case(request) -> Case:
    return request.param


def _assert_same(first, second):
    if isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    else:
        assert first == second


def test_every_sklearn_component_is_registered_here():
    for module in pkgutil.walk_packages(bars_dcn.__path__, "bars_dcn."):
        importlib.import_module(module.name)

    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    components = {
        sub.__name__ for sub in subclasses(BaseEstimator) if sub.__module__.startswith("bars_dcn")
    }
    assert components == {case.name for case in CASES}, (
        "add a Case for each new sklearn component (and remove stale ones)"
    )


def test_get_set_params_roundtrip_and_validation(case):
    est = case.make()
    params = est.get_params()
    assert est.set_params(**params) is est
    assert est.get_params() == params
    with pytest.raises(ValueError, match="Invalid parameter"):
        est.set_params(not_a_parameter=1)


def test_clone_keeps_params_and_drops_fitted_state(case):
    x, y = case.data()
    est = case.make().fit(x, y)
    cloned = clone(est)
    assert cloned.get_params() == est.get_params()
    with pytest.raises(NotFittedError):
        check_is_fitted(cloned)


def test_repr_and_metadata_routing_work(case):
    est = case.make()
    assert type(est).__name__ in repr(est)
    est.get_metadata_routing()  # inspects fit/predict/... signatures


def test_fit_does_not_mutate_params_and_only_adds_trailing_underscore_attributes(case):
    x, y = case.data()
    est = case.make()
    params_before = copy.deepcopy(est.get_params())
    attributes_before = set(vars(est))
    with pytest.raises(NotFittedError):
        check_is_fitted(est)
    est.fit(x, y)
    check_is_fitted(est)
    assert est.get_params() == params_before
    assert all(name.endswith("_") for name in set(vars(est)) - attributes_before)


def test_fit_is_idempotent(case):
    x, y = case.data()
    est = case.make()
    first = case.output(est.fit(x, y), x)
    _assert_same(case.output(est.fit(x, y), x), first)


def test_pickle_roundtrip_preserves_behavior(case):
    x, y = case.data()
    est = case.make().fit(x, y)
    restored = pickle.loads(pickle.dumps(est))  # noqa: S301
    _assert_same(case.output(restored, x), case.output(est, x))


def test_composes_in_a_pipeline_that_clones_and_pickles():
    x_frame, _ = _strings()
    pipeline = Pipeline([("encode", OrdinalEncoder(min_count=1))])
    assert clone(pipeline).get_params().keys() == pipeline.get_params().keys()
    pipeline.fit(x_frame)
    restored = pickle.loads(pickle.dumps(pipeline))  # noqa: S301
    assert restored.transform(x_frame).equals(pipeline.transform(x_frame))


class TestTags:
    def test_ordinal_encoder_declares_what_it_accepts(self):
        tags = get_tags(OrdinalEncoder())
        assert tags.input_tags.string
        assert tags.input_tags.categorical
        assert tags.input_tags.allow_nan
        assert tags.transformer_tags is not None
        assert tags.transformer_tags.preserves_dtype == []

    def test_bucketizer_declares_what_it_accepts(self):
        tags = get_tags(LogSquaredBucketizer())
        assert tags.input_tags.allow_nan
        assert not tags.input_tags.string
        assert tags.transformer_tags is not None
        assert tags.transformer_tags.preserves_dtype == []

    def test_dcn_classifier_declares_what_it_accepts(self):
        tags = get_tags(DCNClassifier())
        assert tags.classifier_tags is not None
        assert not tags.classifier_tags.multi_class
        assert tags.input_tags.positive_only
        assert not tags.input_tags.allow_nan
        assert not tags.input_tags.string
