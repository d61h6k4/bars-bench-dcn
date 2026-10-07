"""skl2onnx converter for ``DCNClassifier``.

Selects its fields from the frame, inlines the exported torch graph and adds the head.
"""

import copy
from typing import TYPE_CHECKING

import numpy as np
import onnx
import torch
from onnx import helper
from skl2onnx.common.data_types import DoubleTensorType, FloatTensorType, Int64TensorType
from torch.export import Dim
from torch.nn.utils.fusion import fuse_linear_bn_eval

from bars_dcn.model.mlp import MLPBlock

if TYPE_CHECKING:
    from bars_dcn.estimator import DCNClassifier

ONNX_OPSET = 20  # StringConcat (the fused string lookup) needs 20


def fold_batch_norm(module: torch.nn.Module) -> torch.nn.Module:
    """Fold each ``Linear -> BatchNorm1d`` of the MLP blocks into one ``Linear`` (eval, exact)."""
    for block in module.modules():
        if not isinstance(block, MLPBlock):
            continue
        layers = list(block.layers)
        folded: list[torch.nn.Module] = []
        for layer in layers:
            if isinstance(layer, torch.nn.BatchNorm1d) and isinstance(folded[-1], torch.nn.Linear):
                folded[-1] = fuse_linear_bn_eval(folded[-1], layer)
            else:
                folded.append(layer)
        block.layers = torch.nn.Sequential(*folded)
    return module


def export_logit_graph(
    module: torch.nn.Module, n_fields: int, n_numeric: int = 0
) -> onnx.ModelProto:
    """Export ``x_cat (n, n_fields) int64 [, x_num (n, n_numeric) float32] -> logit (n,)``."""
    module = fold_batch_norm(copy.deepcopy(module).cpu().eval())
    batch = Dim("batch", min=1)
    example = [torch.zeros(2, n_fields, dtype=torch.int64)]
    names = ["x_cat"]
    if n_numeric:
        example.append(torch.zeros(2, n_numeric, dtype=torch.float32))
        names.append("x_num")
    program = torch.onnx.export(
        module,
        tuple(example),
        dynamo=True,
        opset_version=ONNX_OPSET,
        input_names=names,
        output_names=["logit"],
        dynamic_shapes={name: {0: batch} for name in names},
        report=False,
        verbose=False,
    )
    if program is None:
        msg = "torch.onnx.export returned no program"
        raise RuntimeError(msg)
    return program.model_proto


def shape_calculator(operator) -> None:  # noqa: ANN001
    """Set the output types: the label and both class probabilities."""
    batch = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = Int64TensorType([batch])
    operator.outputs[1].type = FloatTensorType([batch, 2])


def _inline(scope, container, graph: onnx.GraphProto, input_names: list[str]) -> str:  # noqa: ANN001
    """Copy ``graph`` into the container, its inputs bound in order; return the renamed output."""
    names = {i.name: name for i, name in zip(graph.input, input_names, strict=True)}
    for init in graph.initializer:
        names[init.name] = scope.get_unique_variable_name(init.name)
        container.add_initializer(names[init.name], init.data_type, list(init.dims), init)
    for node in graph.node:
        for out in node.output:
            names[out] = scope.get_unique_variable_name(out)
        container.add_node(
            node.op_type,
            [names[i] if i else "" for i in node.input],
            [names[o] for o in node.output],
            op_domain=node.domain,
            op_version=ONNX_OPSET,
            name=scope.get_unique_operator_name(node.name or node.op_type),
            **{a.name: helper.get_attribute_value(a) for a in node.attribute},
        )
    return names[graph.output[0].name]


def _as_float(scope, container, variable) -> str:  # noqa: ANN001
    """Return the variable as float32 (the numeric block is cast, as at fit time)."""
    if not isinstance(variable.type, DoubleTensorType):
        return variable.full_name
    out = scope.get_unique_variable_name("float")
    container.add_node(
        "Cast",
        variable.full_name,
        out,
        name=scope.get_unique_operator_name("Cast"),
        to=onnx.TensorProto.FLOAT,
    )
    return out


def _block(
    scope,  # noqa: ANN001
    container,  # noqa: ANN001
    frame: dict,
    names: object,
    kinds: tuple[type, ...],
    label: str,
    prefix: str,
) -> str:
    """Concatenate the named frame columns into one ``(batch, n)`` block."""
    names = [str(name) for name in names]  # ty: ignore[not-iterable]
    missing = [name for name in names if name not in frame]
    if missing:
        msg = f"the estimator needs columns {missing}, which are not available at this step"
        raise ValueError(msg)
    wrong = [name for name in names if not isinstance(frame[name].type, kinds)]
    if wrong:
        msg = f"columns {wrong} reach the estimator without being encoded to {label}"
        raise ValueError(msg)
    out = scope.get_unique_variable_name(prefix)
    columns = [frame[name] for name in names]
    if all(isinstance(c.type, DoubleTensorType) for c in columns):  # one Cast for the block
        block = scope.get_unique_variable_name("block")
        container.add_node(
            "Concat",
            [c.full_name for c in columns],
            block,
            name=scope.get_unique_operator_name("Concat"),
            axis=1,
        )
        container.add_node(
            "Cast",
            block,
            out,
            name=scope.get_unique_operator_name("Cast"),
            to=onnx.TensorProto.FLOAT,
        )
        return out
    container.add_node(
        "Concat",
        [_as_float(scope, container, c) for c in columns],
        out,
        name=scope.get_unique_operator_name("Concat"),
        axis=1,
    )
    return out


def converter(scope, operator, container) -> None:  # noqa: ANN001
    estimator: DCNClassifier = operator.raw_operator
    if estimator.classes_.dtype.kind not in "iu":
        msg = f"only integer class labels are supported, got {estimator.classes_.dtype}"
        raise NotImplementedError(msg)
    if not hasattr(estimator, "feature_names_in_"):
        msg = "ONNX export needs the field order: fit the estimator on a polars frame"
        raise NotImplementedError(msg)

    # the model's fields, in fit order, from the frame the previous steps produced
    frame = {variable.raw_name: variable for variable in operator.inputs}

    inputs = [
        _block(
            scope, container, frame, estimator.feature_names_in_,
            (Int64TensorType,), "integer indices", "x_cat",
        )
    ]  # fmt: skip
    if estimator.n_numeric_in_:
        inputs.append(
            _block(
                scope, container, frame, estimator.numeric_names_in_,
                (FloatTensorType, DoubleTensorType), "floats", "x_num",
            )
        )  # fmt: skip

    graph = export_logit_graph(
        estimator.model_, estimator.n_features_in_, estimator.n_numeric_in_
    ).graph
    logit = _inline(scope, container, graph, inputs)

    def name(prefix: str) -> str:
        return scope.get_unique_variable_name(prefix)

    def constant(prefix: str, value: np.ndarray) -> str:
        out = name(prefix)
        container.add_initializer(
            out, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape, value
        )
        return out

    label, probabilities = operator.outputs
    positive, positive_col, negative = name("positive"), name("positive_col"), name("negative")
    container.add_node("Sigmoid", logit, positive, name=scope.get_unique_operator_name("Sigmoid"))
    container.add_node(
        "Unsqueeze",
        [positive, constant("axis1", np.array([1], dtype=np.int64))],
        positive_col,
        name=scope.get_unique_operator_name("Unsqueeze"),
    )
    container.add_node(
        "Sub",
        [constant("one", np.array(1, dtype=np.float32)), positive_col],
        negative,
        name=scope.get_unique_operator_name("Sub"),
    )
    container.add_node(
        "Concat",
        [negative, positive_col],
        probabilities.full_name,
        name=scope.get_unique_operator_name("Concat"),
        axis=1,
    )
    best = name("best")
    container.add_node(
        "ArgMax",
        probabilities.full_name,
        best,
        name=scope.get_unique_operator_name("ArgMax"),
        axis=1,
        keepdims=0,
    )
    classes = constant("classes", estimator.classes_.astype(np.int64))
    container.add_node(
        "Gather", [classes, best], label.full_name, name=scope.get_unique_operator_name("Gather")
    )
