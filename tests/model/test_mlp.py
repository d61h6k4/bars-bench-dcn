from torch import nn

from bars_dcn.model import MLPBlock


def _types(block: MLPBlock) -> list[type]:
    return [type(layer) for layer in block.layers]


def test_layer_order_with_batch_norm_and_dropout():
    block = MLPBlock(8, [16, 4], batch_norm=True, dropout=0.1)
    linear, bn, relu, drop = nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout
    assert _types(block) == [linear, bn, relu, drop, linear, bn, relu, drop]
    assert block.output_dim == 4


def test_plain_block_has_only_linear_and_relu():
    block = MLPBlock(8, [16, 4])
    assert _types(block) == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU]
