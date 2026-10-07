"""Small helpers over the skl2onnx container: unique names, constants and nodes."""

from typing import Literal

import numpy as np
from onnx import TensorProto, helper
from skl2onnx.common.data_types import (
    DoubleTensorType,
    FloatTensorType,
    Int64TensorType,
    StringTensorType,
)

DType = Literal["string", "double", "float", "int64"]
ML_DOMAIN = "ai.onnx.ml"
TENSOR_TYPES = {
    "string": StringTensorType,
    "double": DoubleTensorType,
    "float": FloatTensorType,
    "int64": Int64TensorType,
}


def dtype_of(variable) -> DType:  # noqa: ANN001
    """Return the element type of a ``(batch, 1)`` skl2onnx variable."""
    for name, tensor_type in TENSOR_TYPES.items():
        if isinstance(variable.type, tensor_type):
            return name  # ty: ignore[invalid-return-type]
    msg = f"unsupported variable type {variable.type} for {variable.raw_name!r}"
    raise TypeError(msg)


class Builder:
    """Adds nodes and constants to a skl2onnx ``container``, with unique names from ``scope``."""

    def __init__(self, scope, container) -> None:  # noqa: ANN001
        self.scope = scope
        self.container = container

    def constant(self, value: object, dtype: type = np.int64) -> str:
        array = np.asarray(value, dtype=dtype)
        name = self.scope.get_unique_variable_name("const")
        self.container.add_initializer(
            name, helper.np_dtype_to_tensor_dtype(array.dtype), array.shape, array
        )
        return name

    def string_constant(self, values: list[str]) -> str:
        """Add a ``(1, len(values))`` string initializer; return its name."""
        name = self.scope.get_unique_variable_name("const")
        self.container.add_initializer(
            name, TensorProto.STRING, [1, len(values)], [v.encode() for v in values]
        )
        return name

    def op(
        self,
        op_type: str,
        inputs: list[str],
        *,
        domain: str = "",
        version: int | None = None,
        **attrs: object,
    ) -> str:
        """Add a node with one output; return that output's name."""
        output = self.scope.get_unique_variable_name(op_type.lower())
        self.container.add_node(
            op_type,
            inputs,
            output,
            name=self.scope.get_unique_operator_name(op_type),
            op_domain=domain,
            op_version=version,
            **attrs,
        )
        return output

    def ops(self, op_type: str, inputs: list[str], n_outputs: int, **attrs: object) -> list[str]:
        """Add a node with ``n_outputs`` outputs; return their names."""
        outputs = [self.scope.get_unique_variable_name(op_type.lower()) for _ in range(n_outputs)]
        self.container.add_node(
            op_type,
            inputs,
            outputs,
            name=self.scope.get_unique_operator_name(op_type),
            **attrs,
        )
        return outputs
