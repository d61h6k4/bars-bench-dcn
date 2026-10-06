"""Signal-propagation diagnostics: per-layer activation and gradient statistics in TensorBoard.

Everything is observed through PyTorch hooks, attached for one forward/backward pass and removed
afterwards; the model is not modified. Observation points, in execution order:

- ``embedding``: output of the embedding lookup.
- ``cross``: the residual stream ``x0 .. xL`` (``x_{l+1} = x_l + x0 * (W_l x_l + b_l)``). ``x_l`` is
  the input of cross layer ``l`` and ``x_L`` the output of the cross network, so no value is
  recomputed. Only ``x0`` and ``xL`` are available for the low-rank mixture cross network.
- ``stacked`` / ``parallel``: the output of every ReLU of that MLP (post-activation), with the
  fraction of exact zeros.
- ``head``: the final logit.

Backward statistics are of the gradient of the loss with respect to the same tensors. Branches
are independent depth axes, so each gets its own tags with the depth inside the branch as step;
the whole network is also written under the unbranched tags with the global order as step.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from torch import Tensor, nn
from torch.nn import functional

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle
    from torch.utils.tensorboard import SummaryWriter

    from bars_dcn.model import DCNv2

FORWARD = "Signal_Prop_Forward"
BACKWARD = "Signal_Prop_Backward"


@dataclass
class Point:
    branch: str
    label: str
    depth: int  # position inside the branch
    forward: Tensor
    backward: Tensor | None = None


class SignalPropagation:
    """Context manager recording the observation points of a ``DCNv2`` during one pass."""

    def __init__(self, model: DCNv2) -> None:
        self.points: list[Point] = []
        self._model = model
        self._handles: list[RemovableHandle] = []
        self._depth: dict[str, int] = {}

    def __enter__(self) -> Self:
        model = self._model
        self._on_output(model.embedding, "embedding", "embedding")
        self._watch_cross(model.cross)
        for branch in ("stacked", "parallel"):
            block = getattr(model, branch)
            if block is not None:
                relus = (m for m in block.modules() if isinstance(m, nn.ReLU))
                for k, relu in enumerate(relus, start=1):
                    self._on_output(relu, branch, f"{branch}.relu{k}")
        self._on_output(model.fc, "head", "logit")
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _record(self, branch: str, label: str, tensor: Tensor) -> None:
        depth = self._depth.get(branch, 0)
        self._depth[branch] = depth + 1
        point = Point(branch, label, depth, tensor.detach().float().cpu())
        self.points.append(point)
        if tensor.requires_grad:  # a tensor hook sees d(loss)/d(tensor), summed over all uses
            tensor.register_hook(
                lambda grad: setattr(point, "backward", grad.detach().float().cpu())
            )

    def _on_output(self, module: nn.Module, branch: str, label: str) -> None:
        self._handles.append(
            module.register_forward_hook(lambda _m, _i, output: self._record(branch, label, output))
        )

    def _on_input(self, module: nn.Module, branch: str, label: str) -> None:
        self._handles.append(
            module.register_forward_pre_hook(lambda _m, args: self._record(branch, label, args[0]))
        )

    def _watch_cross(self, cross: nn.Module) -> None:
        layers = getattr(cross, "layers", None)  # CrossNetV2 has one Linear per layer
        if layers is None:
            self._on_input(cross, "cross", "x0")
        else:
            for i, layer in enumerate(layers):
                self._on_input(layer, "cross", f"x{i}")
        self._on_output(cross, "cross", f"x{len(layers) if layers is not None else 'L'}")

    def write(self, writer: SummaryWriter) -> None:
        """Std and raw values per point; the global order or the in-branch depth is the step."""
        for index, point in enumerate(self.points):
            self._write_point(writer, "", index, point)
            self._write_point(writer, f"{point.branch}/", point.depth, point)
        layout = "\n".join(
            f"{i}: {p.branch}[{p.depth}] {p.label}" for i, p in enumerate(self.points)
        )
        writer.add_text("Signal_Prop/Layer_Index", layout)

    @staticmethod
    def _write_point(writer: SummaryWriter, scope: str, step: int, point: Point) -> None:
        for prefix, values in ((FORWARD, point.forward), (BACKWARD, point.backward)):
            if values is None:
                continue
            writer.add_scalar(f"{prefix}/{scope}Layer_Std", values.std().item(), step)
            writer.add_histogram(f"{prefix}/{scope}Layer_Dist", values.flatten(), step)
        if ".relu" in point.label:
            zeros = (point.forward == 0).float().mean().item()
            writer.add_scalar(f"{FORWARD}/{scope}Zero_Fraction", zeros, step)


def probe(
    model: DCNv2,
    index: Tensor,
    target: Tensor,
    writer: SummaryWriter,
    numeric: Tensor | None = None,
) -> None:
    """One eval-mode forward/backward pass on a fixed batch, written with ``writer``.

    Eval mode keeps BatchNorm running statistics and dropout masks out of the picture; the
    training state is restored and parameter gradients are cleared.
    """
    was_training = model.training
    model.eval()
    try:
        with SignalPropagation(model) as spp:
            loss = functional.binary_cross_entropy_with_logits(model(index, numeric), target)
            loss.backward()
        spp.write(writer)
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
