#!/usr/bin/env python3
"""Export the competition GlobalContextCNN policy from weights.npz to ONNX.

The graph accepts the already-normalized 45-channel observation, the two
512-turn opponent-stat histories, and the flattened legality penalty. It emits
the exact 4,410 policy logits used by the NumPy competition submission.
"""

from pathlib import Path
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


P = 21
E = 192
GROUPS = 8


class Graph:
    """Small typed-on-paper ONNX graph builder for the fixed g08 architecture."""

    def __init__(self, weights: np.lib.npyio.NpzFile) -> None:
        self.weights = weights
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.index = 0

    def name(self, prefix: str) -> str:
        """Return a unique ONNX value name."""
        self.index += 1
        return f"{prefix}_{self.index}"

    def const(self, name: str, value: np.ndarray) -> str:
        """Register one constant tensor and return its name."""
        self.initializers.append(numpy_helper.from_array(value, name))
        return name

    def weight(self, key: str) -> str:
        """Register one submission weight under an ONNX-safe name."""
        name = key.replace(".", "_")
        value = np.asarray(self.weights[key], dtype=np.float32)
        return self.const(name, value)

    def conv_bias(self, key: str) -> str:
        """Register a convolution bias as the one-dimensional ONNX form."""
        name = key.replace(".", "_")
        value = np.asarray(self.weights[key], dtype=np.float32).reshape(-1)
        return self.const(name, value)

    def op(self, kind: str, inputs: list[str], prefix: str, **attrs: object) -> str:
        """Append a single-output ONNX node."""
        output = self.name(prefix)
        self.nodes.append(helper.make_node(kind, inputs, [output], **attrs))
        return output

    def silu(self, x: str, prefix: str) -> str:
        """SiLU expressed as x * sigmoid(x)."""
        sigmoid = self.op("Sigmoid", [x], f"{prefix}_sigmoid")
        return self.op("Mul", [x, sigmoid], prefix)

    def linear(self, x: str, key: str) -> str:
        """Apply y = x W^T + b to a batch-major matrix."""
        weight_t = self.const(
            f"{key.replace('.', '_')}_T",
            np.asarray(self.weights[f"{key}.weight"], dtype=np.float32).T,
        )
        product = self.op("MatMul", [x, weight_t], f"{key}_matmul")
        return self.op("Add", [product, self.weight(f"{key}.bias")], f"{key}_add")

    def group_norm(self, x: str, key: str) -> str:
        """Match Equinox GroupNorm over each group's channels and spatial cells."""
        shape_group = self.const(f"{key}_group_shape", np.array([1, GROUPS, -1], np.int64))
        grouped = self.op("Reshape", [x, shape_group], f"{key}_grouped")
        axes = self.const(f"{key}_axes", np.array([2], np.int64))
        mean = self.op("ReduceMean", [grouped, axes], f"{key}_mean", keepdims=1)
        centered = self.op("Sub", [grouped, mean], f"{key}_centered")
        square = self.op("Mul", [centered, centered], f"{key}_square")
        variance = self.op("ReduceMean", [square, axes], f"{key}_variance", keepdims=1)
        eps = self.const(f"{key}_eps", np.array(1e-5, np.float32))
        denom = self.op("Sqrt", [self.op("Add", [variance, eps], f"{key}_var_eps")], f"{key}_sqrt")
        normalized = self.op("Div", [centered, denom], f"{key}_normalized")
        shape_map = self.const(f"{key}_map_shape", np.array([1, E, P, P], np.int64))
        mapped = self.op("Reshape", [normalized, shape_map], f"{key}_mapped")
        scale = self.const(
            f"{key}_scale",
            np.asarray(self.weights[f"{key}.weight"], np.float32).reshape(1, E, 1, 1),
        )
        bias = self.const(
            f"{key}_bias",
            np.asarray(self.weights[f"{key}.bias"], np.float32).reshape(1, E, 1, 1),
        )
        return self.op("Add", [self.op("Mul", [mapped, scale], f"{key}_scaled"), bias], f"{key}_shifted")

    def inject(self, x: str, index: int) -> str:
        """Mean+max pool, MLP, and broadcast-add one global context vector."""
        axes = self.const(f"inject_{index}_axes", np.array([2, 3], np.int64))
        mean = self.op("ReduceMean", [x, axes], f"inject_{index}_mean", keepdims=0)
        maximum = self.op("ReduceMax", [x, axes], f"inject_{index}_max", keepdims=0)
        pooled = self.op("Concat", [mean, maximum], f"inject_{index}_pool", axis=1)
        hidden = self.silu(self.linear(pooled, f"injects.{index}.l1"), f"inject_{index}_silu")
        vector = self.linear(hidden, f"injects.{index}.l2")
        unsqueeze_axes = self.const(f"inject_{index}_unsqueeze_axes", np.array([2, 3], np.int64))
        broadcast = self.op("Unsqueeze", [vector, unsqueeze_axes], f"inject_{index}_broadcast")
        return self.op("Add", [x, broadcast], f"inject_{index}_add")


def export(weights_path: Path, output_path: Path) -> None:
    """Build and save the fixed-shape g08 policy graph."""
    weights = np.load(weights_path)
    graph = Graph(weights)

    x = graph.op(
        "Conv",
        ["obs", graph.weight("stem.weight"), graph.conv_bias("stem.bias")],
        "stem",
        kernel_shape=[1, 1],
    )

    temporal_scale = graph.const("temporal_scale", np.array(50.0, np.float32))
    army_input = graph.op("Div", ["army_history", temporal_scale], "army_scaled")
    land_input = graph.op("Div", ["land_history", temporal_scale], "land_scaled")
    army = graph.silu(graph.linear(army_input, "temporal.army_l1"), "army_silu")
    army = graph.linear(army, "temporal.army_l2")
    land = graph.silu(graph.linear(land_input, "temporal.land_l1"), "land_silu")
    land = graph.linear(land, "temporal.land_l2")
    temporal = graph.op("Add", [army, land], "temporal_sum")
    temporal_axes = graph.const("temporal_axes", np.array([2, 3], np.int64))
    temporal_map = graph.op("Unsqueeze", [temporal, temporal_axes], "temporal_map")
    x = graph.op("Add", [x, temporal_map], "temporal_add")

    rows, cols = np.meshgrid(
        np.linspace(-1, 1, P, dtype=np.float32),
        np.linspace(-1, 1, P, dtype=np.float32),
        indexing="ij",
    )
    coords = graph.const("coords", np.stack([rows, cols])[None])
    x = graph.op("Concat", [x, coords], "coord_concat", axis=1)
    x = graph.op(
        "Conv",
        [x, graph.weight("coord_proj.weight"), graph.conv_bias("coord_proj.bias")],
        "coord_proj",
        kernel_shape=[1, 1],
    )

    inject_index = 0
    inject_after = set(int(value) for value in weights["_inject_after"])
    for index in range(int(weights["_depth"])):
        residual = x
        x = graph.silu(graph.group_norm(x, f"blocks.{index}.norm"), f"block_{index}_silu")
        x = graph.op(
            "Conv",
            [x, graph.weight(f"blocks.{index}.conv.weight"), graph.conv_bias(f"blocks.{index}.conv.bias")],
            f"block_{index}_conv",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
        x = graph.op("Add", [residual, x], f"block_{index}_residual")
        if index in inject_after:
            x = graph.inject(x, inject_index)
            inject_index += 1

    x = graph.silu(graph.group_norm(x, "norm_out"), "norm_out_silu")
    policy_weight = graph.const(
        "policy_weight",
        np.asarray(weights["policy_head.weight"], np.float32).reshape(10, E, 1, 1),
    )
    policy_bias = graph.weight("policy_head.bias")
    logits_map = graph.op("Conv", [x, policy_weight, policy_bias], "policy", kernel_shape=[1, 1])
    flat_shape = graph.const("flat_shape", np.array([1, 10 * P * P], np.int64))
    flat_logits = graph.op("Reshape", [logits_map, flat_shape], "flat_logits")
    logits = graph.op("Add", [flat_logits, "penalty"], "logits")

    model_graph = helper.make_graph(
        graph.nodes,
        "g08_global_context_policy",
        [
            helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 45, P, P]),
            helper.make_tensor_value_info("army_history", TensorProto.FLOAT, [1, 512]),
            helper.make_tensor_value_info("land_history", TensorProto.FLOAT, [1, 512]),
            helper.make_tensor_value_info("penalty", TensorProto.FLOAT, [1, 10 * P * P]),
        ],
        [helper.make_tensor_value_info(logits, TensorProto.FLOAT, [1, 10 * P * P])],
        graph.initializers,
    )
    model = helper.make_model(model_graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)


def main() -> None:
    """Parse command-line paths and export the model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.weights, args.output)


if __name__ == "__main__":
    main()
