"""Multi-hash encoder, the shared embedding table, and the ONNX parity of the whole path."""

import numpy as np
import onnxruntime as ort
import polars as pl
import pytest
import torch
from sklearn.pipeline import Pipeline

from bars_dcn.estimator import DCNClassifier
from bars_dcn.model import DCNv2
from bars_dcn.onnx import to_onnx
from bars_dcn.preprocessing import MultiHashEncoder, OrdinalEncoder, hash_name


def _frame(n: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "a": rng.choice(["x", "y", "z", "w"], n).tolist(),
            "b": rng.choice(["p", "q", "r"], n).tolist(),
            "y": rng.integers(0, 2, n),
        }
    )


def test_hashes_are_in_range_and_use_the_documented_formula():
    frame = pl.DataFrame({"a": [0, 1, 2, 10_000_000], "b": [3, 3, 3, 3]})
    encoder = MultiHashEncoder(n_hashes=3, cardinality=101, random_state=5).fit(frame)
    out = encoder.transform(frame)
    assert out.columns[:2] == ["a", "b"]
    for i, column in enumerate(["a", "b"]):
        for k in range(3):
            expected = (
                frame[column].to_numpy().astype(np.int64) * int(encoder.multipliers_[i, k])
                + int(encoder.offsets_[i, k])
            ) % 101
            assert out[hash_name(column, k)].to_list() == expected.tolist()
    assert all(0 <= v < 101 for name in out.columns[2:] for v in out[name].to_list())


def test_fields_do_not_collide_identically():
    frame = pl.DataFrame({"a": list(range(50)), "b": list(range(50))})
    out = MultiHashEncoder(n_hashes=1, cardinality=1009).fit_transform(frame)
    assert out[hash_name("a", 0)].to_list() != out[hash_name("b", 0)].to_list()


def test_non_integer_columns_are_rejected():
    with pytest.raises(ValueError, match="not integer"):
        MultiHashEncoder().fit(pl.DataFrame({"s": ["a"]}))


def test_shared_embedding_has_one_table_and_no_offsets():
    model = DCNv2([5, 5, 5], embedding_dim=4, parallel_hidden_units=[8], shared_embedding=True)
    assert model.embedding.num_embeddings == 5
    assert model.offsets.tolist() == [0, 0, 0]
    assert model(torch.tensor([[4, 0, 4]])).shape == (1,)


def test_onnx_matches_the_pipeline_with_hashed_shared_embeddings():
    frame = _frame()
    columns = ["a", "b"]
    hashed = [hash_name(c, k) for c in columns for k in range(2)]
    pipeline = Pipeline(
        [
            ("encode", OrdinalEncoder(columns=columns, min_count=1)),
            ("hash", MultiHashEncoder(columns=columns, n_hashes=2, cardinality=50)),
            (
                "model",
                DCNClassifier(
                    cat_columns=hashed,
                    shared_embedding=True,
                    embedding_dim=4,
                    num_cross_layers=2,
                    parallel_hidden_units=[16],
                    max_epochs=2,
                    batch_size=64,
                    device="cpu",
                    random_state=0,
                ),
            ),
        ]
    )
    pipeline.fit(frame, frame["y"].to_numpy())
    assert pipeline[-1].model_.embedding.num_embeddings <= 50
    requests = frame.select(columns).head(100).vstack(pl.DataFrame({"a": ["unseen"], "b": [""]}))
    session = ort.InferenceSession(
        to_onnx(pipeline, dict.fromkeys(columns, "string")).SerializeToString()
    )
    feed = {c: np.array(requests[c].to_list(), dtype=object).reshape(-1, 1) for c in columns}
    _, probabilities = session.run(None, feed)
    np.testing.assert_allclose(
        np.asarray(probabilities), pipeline.predict_proba(requests), atol=1e-5
    )
