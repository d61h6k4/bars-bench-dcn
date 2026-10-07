import copy

import torch
from torch import nn

from bars_dcn.model.dcn import DCNv2
from bars_dcn.onnx._estimator import fold_batch_norm


def test_folding_removes_batch_norm_and_keeps_the_outputs():
    torch.manual_seed(0)
    model = DCNv2([5, 7, 9], embedding_dim=4, parallel_hidden_units=[8, 6], batch_norm=True)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            assert module.running_mean is not None
            assert module.running_var is not None
            module.running_mean.normal_(0, 0.5)
            module.running_var.uniform_(0.5, 2.0)
            module.weight.data.uniform_(0.5, 1.5)
            module.bias.data.normal_(0, 0.2)
    model.eval()
    x = torch.stack([torch.randint(0, c, (32,)) for c in (5, 7, 9)], dim=1)
    expected = model(x)
    folded = fold_batch_norm(copy.deepcopy(model))
    assert not any(isinstance(m, nn.BatchNorm1d) for m in folded.modules())
    torch.testing.assert_close(folded(x), expected, atol=1e-5, rtol=1e-5)
    assert any(isinstance(m, nn.BatchNorm1d) for m in model.modules())  # the original is untouched
