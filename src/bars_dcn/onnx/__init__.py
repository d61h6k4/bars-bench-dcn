"""ONNX export of whole sklearn pipelines built from ``bars_dcn`` components.

Importing this package registers the skl2onnx converters. This is training-side tooling; serving
needs only the ``.onnx`` file, ``onnxruntime`` and numpy. The serving contract: one input per
column, named after the column, a string tensor of shape ``(n, 1)`` where a missing value is
``""``; outputs are ``label`` (``(n,)``, int64) and ``probabilities`` (``(n, 2)``, float32).
"""

from typing import TYPE_CHECKING

from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import StringTensorType

from bars_dcn.estimator import DCNClassifier
from bars_dcn.onnx import _estimator, _ordinal
from bars_dcn.onnx._estimator import ONNX_OPSET
from bars_dcn.onnx._ordinal import ML_DOMAIN
from bars_dcn.preprocessing import OrdinalEncoder

if TYPE_CHECKING:
    from collections.abc import Sequence

    import onnx

ML_OPSET = 3

update_registered_converter(
    OrdinalEncoder, "BarsDcnOrdinalEncoder", _ordinal.shape_calculator, _ordinal.converter
)
update_registered_converter(
    DCNClassifier,
    "BarsDcnClassifier",
    _estimator.shape_calculator,
    _estimator.converter,
    options={"zipmap": [True, False, "columns"], "nocl": [True, False]},
)


def to_onnx(model: object, columns: Sequence[str]) -> onnx.ModelProto:
    """Convert a fitted pipeline whose inputs are string columns ``columns``."""
    return convert_sklearn(
        model,
        initial_types=[(column, StringTensorType([None, 1])) for column in columns],
        target_opset={"": ONNX_OPSET, ML_DOMAIN: ML_OPSET},
        options={DCNClassifier: {"zipmap": False}},
    )
