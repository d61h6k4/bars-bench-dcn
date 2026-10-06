"""The real BARS pipelines (passthrough steps, integer and string encoders, column selection)."""

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import polars as pl
import pytest
import torch
from sklearn.pipeline import Pipeline

from bars_dcn.bench.data import AVAZU_FIELDS, AVAZU_RAW, CATEGORICAL, NUMERIC
from bars_dcn.estimator import DCNClassifier
from bars_dcn.onnx import to_onnx
from bars_dcn.preprocessing import (
    AvazuTimeFeatures,
    LogSquaredBucketizer,
    OrdinalEncoder,
)

DATA = Path(__file__).parents[1] / "data"
CRITEO_INPUTS = {**dict.fromkeys(NUMERIC, "double"), **dict.fromkeys(CATEGORICAL, "string")}
AVAZU_INPUTS = dict.fromkeys(["hour", *AVAZU_RAW], "string")


def _model(**kwargs) -> DCNClassifier:
    return DCNClassifier(
        embedding_dim=8, num_cross_layers=2, parallel_hidden_units=[32, 16], batch_norm=True,
        dropout=0.1, max_epochs=1, batch_size=2048, device="cpu", random_state=0, **kwargs,
    )  # fmt: skip


def _randomize(model) -> None:
    """Untrained weights give ~0.5 everywhere; make outputs vary so parity is meaningful."""
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        model.embedding.weight.normal_(0, 0.3, generator=gen)
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                module.running_mean.normal_(0, 0.5, generator=gen)
                module.running_var.uniform_(0.5, 2.0, generator=gen)


def _feed(frame: pl.DataFrame, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    """What a serving process builds: strings with '' for missing, float64 with NaN for missing."""
    feed = {}
    for column, dtype in inputs.items():
        values = frame[column].to_list()
        if dtype == "double":
            feed[column] = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
        else:
            feed[column] = np.array(["" if v is None else v for v in values], dtype=object)
        feed[column] = feed[column].reshape(-1, 1)
    return feed


def _assert_parity(pipeline: Pipeline, inputs: dict[str, Any], frame: pl.DataFrame) -> np.ndarray:
    session = ort.InferenceSession(to_onnx(pipeline, inputs).SerializeToString())
    assert [i.name for i in session.get_inputs()] == list(inputs)  # exactly the columns it reads
    label, probabilities = (np.asarray(out) for out in session.run(None, _feed(frame, inputs)))
    expected = pipeline.predict_proba(frame)
    np.testing.assert_allclose(probabilities, expected, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(label, pipeline.predict(frame))
    return probabilities[:, 1]


# --- criteo: bucketizer (double) -> integer encoder -> string encoder -> estimator -------------


@pytest.fixture(scope="module")
def criteo():
    train = pl.read_parquet(DATA / "criteo_x4_sample" / "train.parquet")
    valid = pl.read_parquet(DATA / "criteo_x4_sample" / "valid.parquet").drop("Label")
    pipeline = Pipeline(
        [
            ("bucket", LogSquaredBucketizer(columns=NUMERIC)),
            ("encode_numeric", OrdinalEncoder(columns=NUMERIC, min_count=3, na_values=[0])),
            ("encode_categorical", OrdinalEncoder(columns=CATEGORICAL, min_count=3)),
            ("model", _model()),
        ]
    )
    pipeline.fit(train.drop("Label"), train["Label"])
    _randomize(pipeline["model"].model_)
    edge = pl.DataFrame(
        {
            **{c: [None, 0, -3, 1, 2, 3, 7, 100, 2_147_483_647] for c in NUMERIC},
            **{c: [None, "", "zzz-unseen", None, "", "x", None, "", "y"] for c in CATEGORICAL},
        },
        schema={**dict.fromkeys(NUMERIC, pl.Int32), **dict.fromkeys(CATEGORICAL, pl.String)},
    )
    return pipeline, pl.concat([valid.head(1000), edge])


def test_criteo_pipeline_matches_on_real_and_edge_rows(criteo):
    pipeline, requests = criteo
    positive = _assert_parity(pipeline, CRITEO_INPUTS, requests)
    assert positive.std() > 0.02  # the comparison is not between constants


def test_criteo_graph_has_exactly_the_raw_inputs_and_the_two_outputs(criteo):
    pipeline, _ = criteo
    graph = to_onnx(pipeline, CRITEO_INPUTS).graph
    assert [i.name for i in graph.input] == list(CRITEO_INPUTS)
    assert [o.name for o in graph.output] == ["label", "probabilities"]


def test_criteo_numeric_columns_drive_the_prediction(criteo):
    pipeline, requests = criteo
    session = ort.InferenceSession(to_onnx(pipeline, CRITEO_INPUTS).SerializeToString())
    feed = _feed(requests.head(200), CRITEO_INPUTS)
    original = np.asarray(session.run(None, feed)[1])
    changed = {**feed, "I4": np.full((200, 1), 5000.0)}
    assert np.abs(np.asarray(session.run(None, changed)[1]) - original).max() > 1e-4


def _only_input(session: ort.InferenceSession) -> str:
    """A stand-alone step whose output has its input's name gets that input renamed by skl2onnx."""
    (graph_input,) = session.get_inputs()
    return graph_input.name


# --- the bucketizer alone: exact agreement with polars -----------------------------------------


def test_bucketizer_matches_polars_exactly_for_every_integer_up_to_four_million_and_edges():
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [np.arange(-5, 4_000_000), rng.integers(4_000_000, 2**31 - 1, 2_000_000), [2**31 - 1]]
    ).astype(np.int64)
    frame = pl.DataFrame({"x": values}, schema={"x": pl.Int32})
    step = LogSquaredBucketizer().fit(frame)
    session = ort.InferenceSession(to_onnx(step, {"x": "double"}).SerializeToString())
    (bucket,) = session.run(None, {_only_input(session): values.astype(np.float64).reshape(-1, 1)})
    np.testing.assert_array_equal(np.asarray(bucket).ravel(), step.transform(frame)["x"].to_numpy())


def test_bucketizer_treats_nan_as_the_fill_value():
    frame = pl.DataFrame({"x": [None, 5]}, schema={"x": pl.Int32})
    step = LogSquaredBucketizer().fit(frame)
    session = ort.InferenceSession(to_onnx(step, {"x": "double"}).SerializeToString())
    (bucket,) = session.run(None, {_only_input(session): np.array([[np.nan], [5.0]])})
    assert np.asarray(bucket).ravel().tolist() == step.transform(frame)["x"].to_list() == [0, 2]


# --- avazu: time features (string arithmetic) -> string encoder -> estimator with cat_columns --


@pytest.fixture(scope="module")
def avazu():
    train = pl.read_parquet(DATA / "avazu_x4_sample" / "train.parquet")
    valid = pl.read_parquet(DATA / "avazu_x4_sample" / "valid.parquet")
    pipeline = Pipeline(
        [
            ("time", AvazuTimeFeatures()),
            ("encode", OrdinalEncoder(columns=AVAZU_FIELDS, min_count=2)),
            ("model", _model(cat_columns=AVAZU_FIELDS)),
        ]
    )
    pipeline.fit(train.drop("click"), train["click"])
    _randomize(pipeline["model"].model_)
    edge = valid.head(3).with_columns(
        pl.Series("hour", ["14102123", "14103000", "14102500"]),
        *[pl.lit("zzz-unseen").alias(c) for c in AVAZU_RAW[2:6]],
    )
    return pipeline, pl.concat([valid.head(1000), edge])


def test_avazu_pipeline_matches_on_real_and_edge_rows(avazu):
    pipeline, requests = avazu
    positive = _assert_parity(pipeline, AVAZU_INPUTS, requests)
    assert positive.std() > 0.02


def test_avazu_graph_does_not_read_the_id_column(avazu):
    pipeline, _ = avazu
    names = [i.name for i in to_onnx(pipeline, AVAZU_INPUTS).graph.input]
    assert "id" not in names
    assert names == list(AVAZU_INPUTS)


def test_avazu_time_features_match_polars_for_every_day_2000_to_2068():
    days = pl.date_range(pl.date(2000, 1, 1), pl.date(2068, 12, 31), eager=True)
    stamps = (
        pl.DataFrame({"d": days}).select(pl.col("d").dt.strftime("%y%m%d")).to_series().to_list()
    )
    hours = ["00", "07", "23"]
    frame = pl.DataFrame({"hour": [d + h for d in stamps for h in hours]})
    step = AvazuTimeFeatures().fit(frame)
    session = ort.InferenceSession(to_onnx(step, {"hour": "string"}).SerializeToString())
    feed = {_only_input(session): frame["hour"].to_numpy().astype(object).reshape(-1, 1)}
    outputs = dict(
        zip([o.name for o in session.get_outputs()], session.run(None, feed), strict=True)
    )
    expected = step.transform(frame)
    for column in ("hour", "weekday", "weekend"):
        assert np.asarray(outputs[column]).ravel().tolist() == expected[column].to_list(), column


# --- errors are explicit ------------------------------------------------------------------------


def test_missing_input_column_is_reported():
    pipeline = Pipeline([("encode", OrdinalEncoder(min_count=1)), ("model", _model())])
    frame = pl.DataFrame({"a": ["x", "y"] * 20, "b": ["p", "q"] * 20})
    pipeline.fit(frame, [0, 1] * 20)
    with pytest.raises(ValueError, match="needs column 'b'"):
        to_onnx(pipeline, {"a": "string"})


def test_wrong_input_type_is_reported():
    step = LogSquaredBucketizer().fit(pl.DataFrame({"x": [1, 2]}))
    with pytest.raises(ValueError, match="as double, got string"):
        to_onnx(step, {"x": "string"})


def test_estimator_fit_on_an_array_cannot_be_exported():
    rng = np.random.default_rng(0)
    model = _model().fit(rng.integers(0, 4, size=(60, 3)), rng.integers(0, 2, 60))
    with pytest.raises(NotImplementedError, match="polars frame"):
        to_onnx(Pipeline([("model", model)]), dict.fromkeys("abc", "string"))
