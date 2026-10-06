"""skl2onnx converter for ``DCNClassifier``: inlines the exported torch graph, adds the head."""

import copy
from typing import TYPE_CHECKING

import numpy as np
import onnx
import torch
from onnx import helper
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from torch.export import Dim

if TYPE_CHECKING:
    from bars_dcn.estimator import DCNClassifier

ONNX_OPSET = 18


def export_logit_graph(module: torch.nn.Module, n_fields: int) -> onnx.ModelProto:
    """Export ``x_cat (n, n_fields) int64 -> logit (n,)`` with a dynamic batch dimension."""
    module = copy.deepcopy(module).cpu().eval()
    example = torch.zeros(2, n_fields, dtype=torch.int64)
    program = torch.onnx.export(
        module,
        (example,),
        dynamo=True,
        opset_version=ONNX_OPSET,
        input_names=["x_cat"],
        output_names=["logit"],
        dynamic_shapes={"x_cat": {0: Dim("batch", min=1)}},
        report=False,
        verbose=False,
    )
    if program is None:
        msg = "torch.onnx.export returned no program"
        raise RuntimeError(msg)
    return program.model_proto


def shape_calculator(operator) -> None:  # noqa: ANN001
    calculate_linear_classifier_output_shapes(operator)


def _inline(scope, container, graph: onnx.GraphProto, input_name: str) -> str:  # noqa: ANN001
    """Copy ``graph`` into the container; return the renamed single output."""
    names = {graph.input[0].name: input_name}
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


def converter(scope, operator, container) -> None:  # noqa: ANN001
    estimator: DCNClassifier = operator.raw_operator
    if estimator.cat_columns is not None:
        msg = "ONNX conversion with cat_columns (column selection) is not implemented yet"
        raise NotImplementedError(msg)
    if estimator.classes_.dtype.kind not in "iu":
        msg = f"only integer class labels are supported, got {estimator.classes_.dtype}"
        raise NotImplementedError(msg)

    graph = export_logit_graph(estimator.model_, estimator.n_features_in_).graph
    logit = _inline(scope, container, graph, operator.inputs[0].full_name)

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
