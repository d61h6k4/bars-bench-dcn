"""Graph clean-up after conversion: drop a ``Split`` whose pieces are re-joined by a ``Concat``.

The converters emit per-column variables (a ``Split`` after a vectorized step) and the estimator
joins the columns again; when the ``Concat`` takes exactly all outputs of one ``Split``, in order,
along the same axis, the pair is the identity and both nodes go.
"""

from collections import Counter

import onnx
from onnx import helper


def _axis(node: onnx.NodeProto) -> int:
    return next((helper.get_attribute_value(a) for a in node.attribute if a.name == "axis"), 0)


def cancel_split_concat(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove every ``Concat(Split(x))`` identity in place and return the model."""
    graph = model.graph
    uses = Counter(name for node in graph.node for name in node.input)
    uses.update(output.name for output in graph.output)
    splits = {
        out: i
        for i, node in enumerate(graph.node)
        if node.op_type == "Split"
        for out in node.output
    }
    rename: dict[str, str] = {}
    dead: set[int] = set()
    for i, node in enumerate(graph.node):
        if node.op_type != "Concat" or not node.input or node.input[0] not in splits:
            continue
        split_index = splits[node.input[0]]
        split = graph.node[split_index]
        identity = (
            list(node.input) == list(split.output)
            and _axis(node) == _axis(split)
            and len(split.input) == 1  # equal pieces: no explicit ``split`` sizes
            and all(uses[name] == 1 for name in split.output)
        )
        if identity:
            rename[node.output[0]] = split.input[0]
            dead.update((i, split_index))
    if not rename:
        return model
    kept = [node for i, node in enumerate(graph.node) if i not in dead]
    for node in kept:
        for i, name in enumerate(node.input):
            node.input[i] = rename.get(name, name)
    del graph.node[:]
    graph.node.extend(kept)
    return model
