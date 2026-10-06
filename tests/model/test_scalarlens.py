import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch.export import Dim

from bars_dcn.model.scalarlens import ScalarLens

N, C, D = 3, 4, 8


def _module(**kwargs) -> ScalarLens:
    torch.manual_seed(0)
    module = ScalarLens(N, C, D, intervals=5, rank=4, head_width=16, **kwargs)
    module.set_ranges(torch.tensor([0.0, -10.0, 5.0]), torch.tensor([100.0, 10.0, 5.0]))
    with torch.no_grad():  # make the learned parts non-trivial
        module.width_logits.normal_(0, 1)
        module.knots.normal_(0, 1)
    return module


def test_boundaries_are_strictly_increasing_from_low_to_high():
    bounds = _module().boundaries()
    assert bounds.shape == (N, 6)
    assert (bounds.diff(dim=-1) > 0).all()
    torch.testing.assert_close(bounds[:, 0], torch.tensor([0.0, -10.0, 5.0]))
    torch.testing.assert_close(bounds[:2, -1], torch.tensor([100.0, 10.0]))
    assert bounds[2, -1] == 6.0  # a constant field gets width 1


def test_the_range_endpoints_map_to_the_first_and_last_vectors():
    module = _module()
    knots = module.knots[0]
    coordinate = module.coordinate(torch.tensor([[0.0, 0.0, 5.0], [100.0, 0.0, 5.0]]))
    torch.testing.assert_close(coordinate[0, 0], knots[0])
    torch.testing.assert_close(coordinate[1, 0], knots[-1])


def test_coordinate_is_exactly_the_midpoint_in_the_middle_of_an_interval():
    module = _module()
    bounds, knots = module.boundaries()[0], module.knots[0]
    middle = ((bounds[2] + bounds[3]) / 2).item()
    x = torch.tensor([[middle, 0.0, 0.0]])
    torch.testing.assert_close(module.coordinate(x)[0, 0], 0.5 * (knots[2] + knots[3]))


def test_values_outside_the_training_range_are_clipped():
    module = _module()
    low = module.coordinate(torch.tensor([[0.0, -10.0, 5.0]]))
    high = module.coordinate(module.high.unsqueeze(0))
    far = module.coordinate(torch.tensor([[-1e9, -1e9, -1e9]]))
    torch.testing.assert_close(far, low)
    torch.testing.assert_close(module.coordinate(torch.tensor([[1e9, 1e9, 1e9]])), high)


def test_the_coordinate_ignores_the_categorical_context_but_the_response_does_not():
    module = _module()
    x = torch.tensor([[10.0, 0.0, 5.0], [10.0, 0.0, 5.0]])
    cat = torch.randn(2, C, D)
    out = module(x, cat)
    assert out.shape == (2, N, D)
    assert not torch.allclose(out[0], out[1])  # same value, different context: different token
    same = module(x, cat[:1].expand(2, -1, -1))
    torch.testing.assert_close(same[0], same[1])  # identical value and context: identical token
    torch.testing.assert_close(module.coordinate(x)[0], module.coordinate(x)[1])


def test_gradients_reach_the_boundaries_knots_and_the_shared_operator():
    module = _module()
    x = torch.rand(16, N) * 50
    module(x, torch.randn(16, C, D)).square().sum().backward()
    for name in ("width_logits", "knots", "head_weight", "down", "up", "step_logits", "gain"):
        grad = getattr(module, name).grad
        assert grad is not None
        assert torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, name


def test_onnx_export_matches_torch_for_any_batch_size():
    module = _module().eval()
    program = torch.onnx.export(
        module,
        (torch.zeros(2, N), torch.zeros(2, C, D)),
        dynamo=True,
        opset_version=18,
        input_names=["x_num", "cat_in"],
        dynamic_shapes={"x_num": {0: Dim("b", min=1)}, "cat_embeddings": {0: Dim("b", min=1)}},
        report=False,
        verbose=False,
    )
    assert program is not None
    session = ort.InferenceSession(program.model_proto.SerializeToString())
    for batch in (1, 7):
        x = torch.cat(
            [torch.rand(batch, 1) * 120 - 10, torch.randn(batch, 1) * 20, torch.rand(batch, 1)], 1
        )
        cat = torch.randn(batch, C, D)
        (got,) = session.run(None, {"x_num": x.numpy(), "cat_in": cat.numpy()})
        np.testing.assert_allclose(np.asarray(got), module(x, cat).detach().numpy(), atol=1e-5)


def test_boundaries_that_collapse_in_float32_give_finite_outputs_and_gradients():
    module = ScalarLens(2, C, D, intervals=16)
    module.set_ranges(
        torch.full((2,), 5e8), torch.full((2,), 5e8)
    )  # span 1 is below float32 spacing
    x = torch.tensor([[5e8, 5e8 - 100], [5e8 + 100, 5e8]])
    out = module(x, torch.randn(2, C, D))
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


def test_degenerate_interval_count_is_rejected():
    with pytest.raises(ValueError, match="intervals"):
        ScalarLens(N, C, D, intervals=0)
