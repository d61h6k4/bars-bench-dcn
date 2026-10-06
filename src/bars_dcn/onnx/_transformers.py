"""skl2onnx parsers and converters for the closed set of ``bars_dcn`` transformers.

Our transformers are passthrough: a step takes the whole frame and replaces or adds some named
columns. skl2onnx models that with one ``(batch, 1)`` variable per column. The parser of a step
returns the frame it produces: new variables for the columns the step changes or adds and the
*same* variables for every other column, so untouched columns cost nothing and later steps can
still select by name. The converter emits the nodes for the changed columns only.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from onnx import TensorProto
from skl2onnx.common.data_types import DoubleTensorType, Int64TensorType, StringTensorType

from bars_dcn.onnx._builder import ML_DOMAIN, Builder, DType, dtype_of
from bars_dcn.preprocessing import (
    OOV_INDEX,
    AvazuTimeFeatures,
    LogSquaredBucketizer,
    OrdinalEncoder,
)

if TYPE_CHECKING:
    from collections.abc import Callable

TENSOR_TYPES = {"string": StringTensorType, "double": DoubleTensorType, "int64": Int64TensorType}
LABEL_ENCODER_VERSION = 2
# Sakamoto's day-of-week table: (y + y/4 - y/100 + y/400 + T[m-1] + d) % 7 with Sunday = 0
_WEEKDAY_TABLE = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]


@dataclass(frozen=True)
class Column:
    """A ``(batch, 1)`` tensor of the frame: its graph name and element type."""

    name: str
    dtype: DType


Frame = dict[str, Column]


def _need(frame: Frame, name: str, dtype: DType, who: str) -> str:
    if name not in frame:
        msg = f"{who} needs column {name!r}, which is not in the ONNX inputs or earlier steps"
        raise ValueError(msg)
    if frame[name].dtype != dtype:
        msg = f"{who} needs column {name!r} as {dtype}, got {frame[name].dtype}"
        raise ValueError(msg)
    return frame[name].name


def _bucketizer_outputs(step: LogSquaredBucketizer) -> dict[str, DType]:
    return dict.fromkeys(step.columns_, "int64")


def _emit_bucketizer(graph: Builder, frame: Frame, step: LogSquaredBucketizer) -> Frame:
    """``x > t ? floor(ln(x)^2) : x`` on ``double`` (NaN = missing -> ``fill_value``), as int64."""
    fill = graph.constant(step.fill_value, np.float64)
    threshold = graph.constant(step.threshold, np.float64)
    out: Frame = {}
    for column in step.columns_:
        value = _need(frame, column, "double", "LogSquaredBucketizer")
        value = graph.op("Where", [graph.op("IsNaN", [value]), fill, value])
        log = graph.op("Log", [value])
        floor = graph.op("Floor", [graph.op("Mul", [log, log])])
        bucket = graph.op("Where", [graph.op("Greater", [value, threshold]), floor, value])
        out[column] = Column(graph.op("Cast", [bucket], to=TensorProto.INT64), "int64")
    return out


def _ordinal_outputs(step: OrdinalEncoder) -> dict[str, DType]:
    return dict.fromkeys(step.columns_, "int64")


def _emit_ordinal(graph: Builder, frame: Frame, step: OrdinalEncoder) -> Frame:
    """One ``LabelEncoder`` per column; every value outside the vocabulary maps to OOV 0."""
    string = step.kind_ == "string"
    key_type: DType = "string" if string else "int64"
    key_attribute = "keys_strings" if string else "keys_int64s"
    out: Frame = {}
    for column, categories in zip(step.columns_, step.categories_, strict=True):
        index = graph.op(
            "LabelEncoder",
            [_need(frame, column, key_type, "OrdinalEncoder")],
            domain=ML_DOMAIN,
            version=LABEL_ENCODER_VERSION,
            **{key_attribute: list(categories)},
            values_int64s=list(range(1, len(categories) + 1)),
            default_int64=OOV_INDEX,
        )
        out[column] = Column(index, "int64")
    return out


def _avazu_outputs(step: AvazuTimeFeatures) -> dict[str, DType]:
    return {step.column: "string", "weekday": "string", "weekend": "string"}


def _emit_avazu_time(graph: Builder, frame: Frame, step: AvazuTimeFeatures) -> Frame:
    """Hour-of-day, weekday and weekend (as strings) from the ``YYMMDDHH`` string, in integers.

    ONNX has no substring operator, so the string is cast to an integer and the calendar is
    computed arithmetically. The hour must be a well-formed ``YYMMDDHH`` string (not missing).
    """
    stamp = graph.op(
        "Cast", [_need(frame, step.column, "string", "AvazuTimeFeatures")], to=TensorProto.INT64
    )

    def const(value: int) -> str:
        return graph.constant([value])

    def div(a: str, b: int) -> str:
        return graph.op("Div", [a, const(b)])

    def mod(a: str, b: int) -> str:
        return graph.op("Mod", [a, const(b)])

    def add(a: str, b: str) -> str:
        return graph.op("Add", [a, b])

    def sub(a: str, b: str) -> str:
        return graph.op("Sub", [a, b])

    def as_int(boolean: str) -> str:
        return graph.op("Cast", [boolean], to=TensorProto.INT64)

    date = div(stamp, 100)  # YYMMDD
    day, month = mod(date, 100), mod(div(date, 100), 100)
    year = add(div(date, 10_000), const(2000))
    year = sub(year, as_int(graph.op("Less", [month, const(3)])))  # January and February count
    table = graph.op("Gather", [graph.constant(_WEEKDAY_TABLE), sub(month, const(1))], axis=0)
    leaps = sub(add(div(year, 4), div(year, 400)), div(year, 100))
    weekday = mod(add(add(year, leaps), add(table, day)), 7)  # Sunday = 0
    saturday_or_sunday = graph.op(
        "Or", [graph.op("Equal", [weekday, const(0)]), graph.op("Equal", [weekday, const(6)])]
    )

    def text(value: str) -> Column:
        return Column(graph.op("Cast", [value], to=TensorProto.STRING), "string")

    return {
        step.column: text(mod(stamp, 100)),
        "weekday": text(weekday),
        "weekend": text(as_int(saturday_or_sunday)),
    }


@dataclass(frozen=True)
class Spec:
    alias: str
    outputs: Callable[[Any], dict[str, DType]]  # changed or added columns and their types
    emit: Callable[[Builder, Frame, Any], Frame]


SPECS: dict[type, Spec] = {
    LogSquaredBucketizer: Spec(
        "BarsDcnLogSquaredBucketizer", _bucketizer_outputs, _emit_bucketizer
    ),
    OrdinalEncoder: Spec("BarsDcnOrdinalEncoder", _ordinal_outputs, _emit_ordinal),
    AvazuTimeFeatures: Spec("BarsDcnAvazuTimeFeatures", _avazu_outputs, _emit_avazu_time),
}


def parse(scope, model, inputs, custom_parsers=None):  # noqa: ANN001, ANN202, ARG001
    """Frame in, frame out: new variables for changed or added columns, the rest passes through."""
    spec = SPECS[type(model)]
    operator = scope.declare_local_operator(spec.alias, model)
    operator.inputs = list(inputs)
    batch = inputs[0].get_first_dimension()
    changed = {
        name: scope.declare_local_variable(name, TENSOR_TYPES[dtype]([batch, 1]))
        for name, dtype in spec.outputs(model).items()
    }
    operator.outputs = list(changed.values())
    present = {variable.raw_name for variable in inputs}
    return [changed.get(variable.raw_name, variable) for variable in inputs] + [
        variable for name, variable in changed.items() if name not in present
    ]


def shape_calculator(operator) -> None:  # noqa: ANN001
    """Output types are declared by the parser."""


def converter(scope, operator, container) -> None:  # noqa: ANN001
    step = operator.raw_operator
    spec = SPECS[type(step)]
    frame = {v.raw_name: Column(v.full_name, dtype_of(v)) for v in operator.inputs}
    produced = spec.emit(Builder(scope, container), frame, step)
    for variable in operator.outputs:
        container.add_node(
            "Identity",
            produced[variable.raw_name].name,
            variable.full_name,
            name=scope.get_unique_operator_name("Identity"),
        )
