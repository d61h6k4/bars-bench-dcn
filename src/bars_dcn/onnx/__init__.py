"""ONNX export of whole sklearn pipelines built from ``bars_dcn`` components, via skl2onnx.

Importing this package registers the skl2onnx parsers and converters. This is training-side
tooling; serving needs only the ``.onnx`` file, ``onnxruntime`` and numpy.

Serving contract: one input per column the pipeline reads, named after the column, of shape
``(n, 1)``: ``"string"`` columns as strings where a missing value is ``""``, ``"double"`` columns
as float64 where a missing value is NaN. Outputs are ``label`` (``(n,)``, int64) and
``probabilities`` (``(n, 2)``, float32). Columns the pipeline never reads (an ``id``, the label)
are simply not inputs.
"""

from typing import TYPE_CHECKING

from skl2onnx import convert_sklearn, update_registered_converter

from bars_dcn.estimator import DCNClassifier
from bars_dcn.onnx import _estimator, _transformers
from bars_dcn.onnx._builder import ML_DOMAIN, TENSOR_TYPES
from bars_dcn.onnx._estimator import ONNX_OPSET

if TYPE_CHECKING:
    from collections.abc import Mapping

    import onnx

ML_OPSET = 3

for _cls, _spec in _transformers.SPECS.items():
    update_registered_converter(
        _cls,
        _spec.alias,
        _transformers.shape_calculator,
        _transformers.converter,
        parser=_transformers.parse,
    )
update_registered_converter(
    DCNClassifier,
    "BarsDcnClassifier",
    _estimator.shape_calculator,
    _estimator.converter,
    options={"zipmap": [True, False, "columns"], "nocl": [True, False]},
)


def to_onnx(model: object, inputs: Mapping[str, str]) -> onnx.ModelProto:
    """Convert a fitted pipeline; ``inputs`` maps each column it reads to its element type."""
    unknown = {name: dtype for name, dtype in inputs.items() if dtype not in TENSOR_TYPES}
    if unknown:
        msg = f"input types must be one of {sorted(TENSOR_TYPES)}, got {unknown}"
        raise ValueError(msg)
    return convert_sklearn(
        model,
        initial_types=[(name, TENSOR_TYPES[dtype]([None, 1])) for name, dtype in inputs.items()],
        target_opset={"": ONNX_OPSET, ML_DOMAIN: ML_OPSET},
        options={DCNClassifier: {"zipmap": False}},
    )
