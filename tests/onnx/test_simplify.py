import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from bars_dcn.onnx._simplify import cancel_split_concat


def _model(nodes: list, outputs: list[str]) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in outputs],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)


def _run(model: onnx.ModelProto) -> list[np.ndarray]:
    session = ort.InferenceSession(model.SerializeToString())
    feed = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    return [np.asarray(out) for out in session.run(None, feed)]


def _split() -> onnx.NodeProto:
    return helper.make_node("Split", ["x"], ["a", "b", "c"], axis=1, num_outputs=3)


def test_a_concat_of_all_split_pieces_in_order_is_removed():
    model = _model(
        [_split(), helper.make_node("Concat", ["a", "b", "c"], ["j"], axis=1),
         helper.make_node("Relu", ["j"], ["y"])],
        ["y"],
    )  # fmt: skip
    expected = _run(model)
    cancel_split_concat(model)
    assert [n.op_type for n in model.graph.node] == ["Relu"]
    np.testing.assert_array_equal(_run(model)[0], expected[0])


def test_reordered_or_partial_concats_are_kept():
    for pieces in (["b", "a", "c"], ["a", "b"]):
        model = _model([_split(), helper.make_node("Concat", pieces, ["y"], axis=1)], ["y"])
        cancel_split_concat(model)
        assert [n.op_type for n in model.graph.node] == ["Split", "Concat"]


def test_split_pieces_used_elsewhere_are_kept():
    model = _model(
        [_split(), helper.make_node("Concat", ["a", "b", "c"], ["j"], axis=1),
         helper.make_node("Relu", ["a"], ["y"])],
        ["y", "j"],
    )  # fmt: skip
    cancel_split_concat(model)
    assert [n.op_type for n in model.graph.node] == ["Split", "Concat", "Relu"]
