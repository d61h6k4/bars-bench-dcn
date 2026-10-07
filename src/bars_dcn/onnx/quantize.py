"""Post-training int8 quantization of the large dense layers of an exported pipeline.

At batch 1 the MLP and cross layers are weight-streaming bound, so int8 weights (a quarter of
the bytes) are the lever. Each large ``Gemm`` becomes ``DynamicQuantizeLinear`` (uint8
activations, scale computed per call) -> ``MatMulInteger`` with per-output-channel symmetric int8
weights -> rescale -> bias. Only layers with at least ``min_weights`` weights are rewritten: the
small ScalarLens projections would gain nodes for no saving.
"""

from typing import TYPE_CHECKING

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

if TYPE_CHECKING:
    from pathlib import Path

_INT8_MAX = 127
_BIAS = 2  # position of the optional bias among a Gemm's inputs


def _quantized_gemm(
    node: onnx.NodeProto, weight: np.ndarray, bias: str | None
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    """Nodes and initializers computing ``x @ weight`` (``weight`` is ``(K, N)``) in int8."""
    name = node.name or node.output[0]
    scale = np.maximum(np.abs(weight).max(axis=0), 1e-12) / _INT8_MAX  # per output channel
    quantized = np.clip(np.rint(weight / scale), -_INT8_MAX, _INT8_MAX).astype(np.int8)

    def tag(suffix: str) -> str:
        return f"{name}_{suffix}"

    inits = [
        numpy_helper.from_array(quantized, tag("w_int8")),
        numpy_helper.from_array(scale.astype(np.float32), tag("w_scale")),
    ]
    nodes = [
        helper.make_node(
            "DynamicQuantizeLinear",
            [node.input[0]],
            [tag("xq"), tag("x_scale"), tag("x_zero")],
            name=tag("quantize"),
        ),
        helper.make_node(
            "MatMulInteger", [tag("xq"), tag("w_int8"), tag("x_zero")], [tag("acc")], name=tag("mm")
        ),
        helper.make_node("Cast", [tag("acc")], [tag("acc_f")], to=TensorProto.FLOAT, name=tag("c")),
        helper.make_node("Mul", [tag("x_scale"), tag("w_scale")], [tag("scale")], name=tag("s")),
        helper.make_node(
            "Mul", [tag("acc_f"), tag("scale")], [tag("out") if bias else node.output[0]],
            name=tag("rescale"),
        ),
    ]  # fmt: skip
    if bias:
        nodes.append(
            helper.make_node("Add", [tag("out"), bias], [node.output[0]], name=tag("bias"))
        )
    return nodes, inits


def quantize_dense(source: Path, target: Path, min_weights: int = 100_000) -> list[str]:
    """Write ``target``: ``source`` with its large ``Gemm`` layers in int8. Returns their names."""
    model = onnx.load(source)
    weights = {i.name: i for i in model.graph.initializer}
    replaced: list[str] = []
    new_nodes: list[onnx.NodeProto] = []
    drop: set[str] = set()
    for node in model.graph.node:
        weight = weights.get(node.input[1]) if node.op_type == "Gemm" else None
        if weight is None or int(np.prod(weight.dims)) < min_weights:
            new_nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if (
            attrs.get("transA", 0)
            or attrs.get("alpha", 1.0) != 1.0
            or attrs.get("beta", 1.0) != 1.0
        ):
            msg = f"Gemm {node.name!r} uses transA/alpha/beta, which is not supported"
            raise NotImplementedError(msg)
        matrix = numpy_helper.to_array(weight)
        matrix = matrix.T if attrs.get("transB", 0) else matrix
        nodes, inits = _quantized_gemm(
            node, matrix, node.input[_BIAS] if len(node.input) > _BIAS else None
        )
        new_nodes.extend(nodes)
        model.graph.initializer.extend(inits)
        drop.add(weight.name)
        replaced.append(node.name)
    if not replaced:
        msg = f"no Gemm with at least {min_weights} weights in {source}"
        raise ValueError(msg)
    used = {i for node in new_nodes for i in node.input}
    keep = [i for i in model.graph.initializer if i.name not in drop or i.name in used]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    onnx.save(model, target)
    return replaced
