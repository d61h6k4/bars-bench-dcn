"""skl2onnx converter for ``OrdinalEncoder``: one ``LabelEncoder`` per string column."""

from skl2onnx.common.data_types import Int64TensorType

from bars_dcn.preprocessing import OOV_INDEX, OrdinalEncoder

ML_DOMAIN = "ai.onnx.ml"
LABEL_ENCODER_VERSION = 2


def shape_calculator(operator) -> None:  # noqa: ANN001
    n_rows = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = Int64TensorType([n_rows, len(operator.inputs)])


def converter(scope, operator, container) -> None:  # noqa: ANN001
    encoder: OrdinalEncoder = operator.raw_operator
    if encoder.kind_ != "string":
        msg = "ONNX conversion of integer-column OrdinalEncoder is not implemented yet"
        raise NotImplementedError(msg)
    if [variable.raw_name for variable in operator.inputs] != encoder.columns_:
        msg = (
            "ONNX conversion of passthrough pipelines (encoder columns differ from the graph "
            "inputs) is not implemented yet"
        )
        raise NotImplementedError(msg)
    encoded = []
    for variable, categories in zip(operator.inputs, encoder.categories_, strict=True):
        out = scope.get_unique_variable_name(f"{variable.raw_name}_index")
        container.add_node(
            "LabelEncoder",
            variable.full_name,
            out,
            op_domain=ML_DOMAIN,
            op_version=LABEL_ENCODER_VERSION,
            name=scope.get_unique_operator_name("LabelEncoder"),
            keys_strings=categories,
            values_int64s=list(range(1, len(categories) + 1)),
            default_int64=OOV_INDEX,
        )
        encoded.append(out)
    container.add_node(
        "Concat",
        encoded,
        operator.outputs[0].full_name,
        name=scope.get_unique_operator_name("Concat"),
        axis=1,
    )
