"""Keras 3.x split methods. Includes HGQ2
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable, Union, Literal, Sequence

import keras
import math
import hgq

LayerId = Union[int, str]

def _is_input_layer(layer: keras.layers.Layer) -> bool:
    # Keras 3 InputLayer type exists; keep it soft in case backends differ
    return layer.__class__.__name__ == "InputLayer"


def _sanitize_io_name(name: str) -> str:
    # Keras names can contain "/" and ":"; make them stable as Input names.
    return name.replace("/", "_").replace(":", "_")


def _resolve_group_layers(model: keras.Model, group: list[LayerId]) -> list[keras.layers.Layer]:
    if not group:
        raise ValueError("Empty list cannot be a sub model")

    resolved: list[keras.layers.Layer] = []
    for item in group:
        if isinstance(item, int):
            if item < 0 or item >= len(model.layers):
                raise IndexError(f"Layer index out of range: {item}")
            layer = model.layers[item]
        elif isinstance(item, str):
            layer = model.get_layer(item)
        else:
            raise TypeError(f"Layer id must be int or str, got {type(item)!r}")

        if _is_input_layer(layer):
            raise ValueError("Do not include InputLayer in groups; use real layers only.")
        resolved.append(layer)

    # preserve original model order (important for deterministic node replay)
    order = {layer: i for i, layer in enumerate(model.layers)}
    resolved.sort(key=lambda l: order.get(l, 10**9))
    return resolved


def _flatten_tensors(x):
    # Keras 3 has keras.tree utilities; keep it minimal.
    return keras.tree.flatten(x)


def sequential_split(
    model: keras.Model,
    layer_groups: list[list[LayerId]],
    names: list[str] | None = None,
) -> list[keras.Model]:
    """
    Split a Functional/Sequential model into sequential submodels based on layer groupings.

    Each returned submodel is a Functional model that:
      - takes as inputs the tensors entering the group from outside
      - outputs the tensors produced inside the group that are consumed outside (or are model outputs)

    Supports branched graphs, multi-input, multi-output, and skip connections.

    Limitations:
      - does NOT support shared/reused layers (same layer called multiple times) reliably.
      - requires groups to be topologically ordered.

    Args:
        model: Keras 3 Model (Sequential or Functional).
        layer_groups: list of groups; each group is a list of layer indices or layer names.
        names: optional list of submodel names (same length as layer_groups). Defaults to "split.{i}".

    Returns:
        list[keras.Model]
    """
    if names is None:
        names = [f"split.{i}" for i in range(len(layer_groups))]
    if len(names) != len(layer_groups):
        raise ValueError("names must have the same length as layer_groups")

    # Resolve layers per group
    groups_layers: list[list[keras.layers.Layer]] = [
        _resolve_group_layers(model, group) for group in layer_groups
    ]

    # Validate disjointness
    all_layers = [l for g in groups_layers for l in g]
    if len(set(all_layers)) != len(all_layers):
        raise ValueError("A layer appears in more than one group (groups must be disjoint).")

    # Reject shared/reused layers (layer called multiple times => multiple inbound nodes)
    # This is the most common “graph replay ambiguity”.
    for layer in all_layers:
        inbound = getattr(layer, "_inbound_nodes", None)
        if inbound is not None and len(inbound) > 1:
            raise NotImplementedError(
                f"Layer {layer.name!r} appears to be reused/shared in the graph "
                f"(len(_inbound_nodes)={len(inbound)}). This splitter currently "
                "does not support shared layers."
            )

    # Build consumer map: tensor -> set(consumer layers)
    consumers: dict[object, set[keras.layers.Layer]] = defaultdict(set)
    for layer in model.layers:
        for node in getattr(layer, "_inbound_nodes", []):
            for t in getattr(node, "input_tensors", []) or []:
                consumers[t].add(layer)

    # Helper: tensor producer layer
    def producer_layer(t) -> keras.layers.Layer | None:
        hist = getattr(t, "_keras_history", None)
        if hist is None:
            return None
        # In Keras 3, _keras_history.operation is the producing layer/operation
        return getattr(hist, "operation", None)

    model_inputs = list(model.inputs)
    model_outputs = list(model.outputs)
    model_output_set = set(model_outputs)

    submodels: list[keras.Model] = []

    # For ordering validation: keep track of which layers are already placed “earlier”
    earlier_layers: set[keras.layers.Layer] = set(l for l in model.layers if _is_input_layer(l))

    for gi, group_layers in enumerate(groups_layers):
        group_set = set(group_layers)

        # Identify boundary input tensors for this group
        boundary_inputs: list[object] = []
        seen_in = set()

        # Prefer model.inputs first (stable ordering)
        for t in model_inputs:
            # If a model input is consumed by any layer in the group, it is a boundary input
            if any(consumer in group_set for consumer in consumers.get(t, set())):
                if t not in seen_in:
                    boundary_inputs.append(t)
                    seen_in.add(t)

        # Then other external tensors feeding into the group
        for layer in group_layers:
            for node in getattr(layer, "_inbound_nodes", []):
                for t in getattr(node, "input_tensors", []) or []:
                    prod = producer_layer(t)
                    if prod is None:
                        continue
                    if prod not in group_set:
                        if t not in seen_in:
                            boundary_inputs.append(t)
                            seen_in.add(t)

                        # Topology check: external producer must be from earlier groups or be an InputLayer
                        if not _is_input_layer(prod) and prod not in earlier_layers:
                            raise ValueError(
                                f"Groups are not topologically ordered. Group {gi} "
                                f"({names[gi]!r}) needs tensor produced by layer {prod.name!r} "
                                "which is not in any earlier group."
                            )

        if not boundary_inputs:
            raise ValueError(
                f"Could not find any boundary inputs for group {gi} ({names[gi]!r}). "
                "This usually means the group is disconnected or groups are not ordered."
            )

        # Identify boundary output tensors: produced in group and used outside group OR are model outputs
        boundary_outputs: list[object] = []
        seen_out = set()

        # First: any model outputs produced by this group (keep model output order)
        for t in model_outputs:
            prod = producer_layer(t)
            if prod in group_set and t not in seen_out:
                boundary_outputs.append(t)
                seen_out.add(t)

        # Then: tensors that leave the group to later layers
        for layer in group_layers:
            for node in getattr(layer, "_inbound_nodes", []):
                for t in getattr(node, "output_tensors", []) or []:
                    if t in seen_out:
                        continue
                    # if any consumer is outside group, it's a boundary output
                    cons = consumers.get(t, set())
                    if any(c not in group_set for c in cons) or (t in model_output_set):
                        boundary_outputs.append(t)
                        seen_out.add(t)

        if not boundary_outputs:
            raise ValueError(
                f"Could not find any boundary outputs for group {gi} ({names[gi]!r}). "
                "This usually means the group does not feed anything outside itself."
            )

        # Build a new subgraph for this group by replaying nodes
        tensor_map: dict[object, object] = {}

        # Create Inputs for boundary inputs
        new_inputs = []
        for t in boundary_inputs:
            shape = tuple(getattr(t, "shape", ())[1:])  # drop batch dim
            dtype = getattr(t, "dtype", None)
            inp = keras.Input(
                shape=shape,
                dtype=dtype,
                name=_sanitize_io_name(getattr(t, "name", f"group{gi}_in")),
            )
            tensor_map[t] = inp
            new_inputs.append(inp)

        # Replay computation inside group:
        # Iterate layers in model order; for each node, if its inputs are available, compute its outputs.
        progress = True
        while progress:
            progress = False
            for layer in group_layers:
                for node in getattr(layer, "_inbound_nodes", []):
                    in_ts = list(getattr(node, "input_tensors", []) or [])
                    out_ts = list(getattr(node, "output_tensors", []) or [])
                    if not in_ts or not out_ts:
                        continue

                    # already computed?
                    if all(t in tensor_map for t in out_ts):
                        continue

                    # ready to compute?
                    if not all(t in tensor_map for t in in_ts):
                        continue

                    new_in = [tensor_map[t] for t in in_ts]
                    call_arg = new_in[0] if len(new_in) == 1 else new_in

                    new_out = layer(call_arg)
                    new_out_flat = _flatten_tensors(new_out)

                    if len(new_out_flat) != len(out_ts):
                        raise RuntimeError(
                            f"Mismatch replaying layer {layer.name!r}: "
                            f"expected {len(out_ts)} outputs, got {len(new_out_flat)}."
                        )

                    for ot, nt in zip(out_ts, new_out_flat):
                        tensor_map[ot] = nt

                    progress = True

        # Materialize boundary outputs
        new_outputs = []
        for t in boundary_outputs:
            if t not in tensor_map:
                prod = producer_layer(t)
                raise RuntimeError(
                    f"Failed to build boundary output tensor {getattr(t, 'name', t)!r} "
                    f"for group {gi} ({names[gi]!r}). Producer={getattr(prod, 'name', prod)!r}. "
                    "This may indicate shared layers, missing layers in groups, or ordering issues."
                )
            new_outputs.append(tensor_map[t])

        inputs_arg = new_inputs[0] if len(new_inputs) == 1 else new_inputs
        outputs_arg = new_outputs[0] if len(new_outputs) == 1 else new_outputs
        submodels.append(keras.Model(inputs=inputs_arg, outputs=outputs_arg, name=names[gi]))

        # Update earlier_layers for topology validation
        earlier_layers.update(group_set)

    return submodels


def dense_split(
    layer: keras.layers.Dense,
    axis: int,
    number_of_partitions: int,
) -> list[keras.Model]:
    """Keras 3.0 Dense layer parallel split function

    Args:
        model (keras.layers.Dense): original dense layer
        axis (int): type of split to perform 0 is input, 1 is neuron
        number_of_partitions (int): number of sub_models to create for this layer

    Raises:
        ValueError: Keras Dense layer only allow for axis 0 or 1

    Returns:
        list[keras.Model]: list of the parallel sub models + the combiner model
    """

    if axis > 1:
        raise ValueError(
            "Dense layers only support axis 0 or 1 for input and neurons splits respectively"
        )

    # get the og layer weights, biases
    original_weights, original_biases = layer.get_weights()

    sub_models = []
    partition_size = original_weights.shape[axis] // number_of_partitions
    for partition in range(number_of_partitions):
        # split the weights & biases if there's a remainder, it's added to the last partition.
        idx = [slice(None)] * original_weights.ndim
        start = partition * partition_size
        end = (
            original_weights.shape[axis]
            if partition == number_of_partitions - 1
            else start + partition_size
        )
        idx[axis] = slice(start, end)
        weights = original_weights[tuple(idx)]

        cfg = layer.get_config()
        if axis == 0:
            # input split
            input_dim = end - start
            cfg["activation"] = None
            if partition == 0:
                cfg["use_bias"] = True
                new_layer = keras.layers.Dense.from_config(cfg)
                new_layer.build((None, input_dim))
                new_layer.set_weights([weights, original_biases])
            else:
                cfg["use_bias"] = False
                new_layer = keras.layers.Dense.from_config(cfg)
                new_layer.build((None, input_dim))
                new_layer.set_weights([weights])
            inputs = keras.Input(shape=(input_dim,))
            output = new_layer(inputs)

        elif axis == 1:
            # neuron split
            biases = original_biases[idx[axis]]
            cfg["units"] = biases.size
            new_layer = keras.layers.Dense.from_config(cfg)
            input_dim = layer.input.shape[1]  # type: ignore
            new_layer.build((None, input_dim))
            new_layer.set_weights([weights, biases])
            inputs = keras.Input(shape=(input_dim,))
            output = new_layer(inputs)

        sub_models.append(keras.Model(inputs=inputs, outputs=output))

    # create a last model that will combine (concat/add) all the outputs together and
    #   should be either added to the next submodel or saved as an intermediate one.
    if axis == 0:
        inputs = [
            keras.Input(shape=tuple(
                sm.output.shape[1:]), name=f"{layer.name}.{i}")
            for i, sm in enumerate(sub_models)
        ]
        combine = keras.layers.Add(name=f"{layer.name}.add")(inputs)
        activation_name = layer.get_config()["activation"]
        output = (
            keras.layers.Activation(activation_name)(combine)
            if activation_name not in ["linear", None]
            else combine
        )

    elif axis == 1:
        inputs = [
            keras.Input(shape=tuple(
                sm.output.shape[1:]), name=f"{layer.name}.{i}")
            for i, sm in enumerate(sub_models)
        ]
        output = keras.layers.Concatenate(name=f"{layer.name}.concat")(inputs)

    combiner = keras.Model(inputs=inputs, outputs=output)
    sub_models.append(combiner)

    return sub_models


def qdense_split(
    layer: hgq.layers.QDense,
    axis: int,
    number_of_partitions: int,
) -> list[keras.Model]:
    """Keras 3.0 Dense layer parallel split function

    Args:
        model (keras.layers.Dense): original dense layer
        axis (bool): type of split to perform 0 is input, 1 is neuron
        number_of_partitions (int): number of sub_models to create for this layer

    Raises:
        ValueError: Keras Dense layer only allow for axis 0 or 1

    Returns:
        list[keras.Model]: list of the parallel sub models + the combiner model


    HGQ2 QDense layer weights layout:
        Idx	Shape	    Meaning ?              
        0	(In, Out)	Kernel weights
        1	(Out)	    Biases
        2	()	        beta
        3	()	        ebops
        4	(1, In)	    f
        5	(1, In)	    k
        6	(1, In)	    i
        7	()	        i_decay_speed
        8	(In, Out)	b
        9	(In, Out)	i
        10	(In, Out)	k
        11	(Out,)	    b
        12	(Out,)	    i
        13	(Out,)	    k
        14	()	        i_decay_speed

    Where the In and Out are the input and output dimensions of the layer to be split.
    """
    og_weights = layer.get_weights()
    input_dim = og_weights[0].shape[0]
    output_dim = og_weights[0].shape[1]
    sub_models = []


    for w in layer.weights:
        print(f"og {w.shape}, {w.name}")

    if axis > 1:
        raise ValueError(
            "QDense layers only support axis 0 or 1 for input and neurons splits respectively"
        )
    
    if axis == 0:
        # input split
        partition_size = input_dim // number_of_partitions

        for partition in range(number_of_partitions):
            new_weights = []
            start = partition * partition_size
            end = input_dim if partition == number_of_partitions - 1 else start + partition_size

            for w in og_weights:
                if w.ndim == 0:
                    new_weights.append(w)
                    continue

                # get the correct dimension to slice
                dim_idx = [i for i, d in enumerate(w.shape) if d == input_dim]
                
                if dim_idx is None or len(dim_idx) == 0:
                    new_weights.append(w)
                
                else:
                    idx = [slice(None)] * w.ndim
                    idx[dim_idx[0]] = slice(start, end)
                    new_weights.append(w[tuple(idx)])
            
            for w in new_weights:
                print(f"partition {partition} {w.shape}, og {len(og_weights)}")

            cfg = layer.get_config()
            cfg['activation'] = None # activation applied after combining
            
            if partition == 0:
                cfg['use_bias'] = True
                new_layer = hgq.layers.QDense.from_config(cfg)
                new_layer.build((None, end-start))
                new_layer.set_weights(new_weights)

            else:
                cfg['use_bias'] = False
                new_layer = hgq.layers.QDense.from_config(cfg)
                new_layer.build((None, end-start))
                for i in [14, 13, 12, 11, 1]: # remove bias related weights. FIXME: better way?
                    if i < len(new_weights):
                        new_weights.pop(i)
                new_layer.set_weights(new_weights)

            new_input = keras.Input(shape=(end-start,))
            new_output = new_layer(new_input)
            sub_models.append(keras.Model(inputs=new_input, outputs=new_output))
           

    elif axis == 1:
        # neuron split
        partition_size = output_dim // number_of_partitions

        for partition in range(number_of_partitions):
            new_weights = []
            start = partition * partition_size
            end = output_dim if partition == number_of_partitions - 1 else start + partition_size

            for w in og_weights:
                if w.ndim == 0:
                    new_weights.append(w)
                    continue

                # get the correct dimension to slice
                dim_idx = [i for i, d in enumerate(w.shape) if d == output_dim]
                
                if dim_idx is None or len(dim_idx) == 0:
                    new_weights.append(w)
                
                else:
                    idx = [slice(None)] * w.ndim
                    idx[dim_idx[0]] = slice(start, end)
                    new_weights.append(w[tuple(idx)])

            cfg = layer.get_config()
            cfg['units'] = end - start
            new_layer = hgq.layers.QDense.from_config(cfg)
            new_layer.build((None, input_dim))
            new_layer.set_weights(new_weights)
            new_input = keras.Input(shape=(input_dim,))
            new_output = new_layer(new_input)
            sub_models.append(keras.Model(inputs=new_input, outputs=new_output))
        
    # create a last model that will combine (concat/add) all the outputs together and
    #   should be either added to the next submodel or saved as an intermediate one.
    if axis == 0:
        inputs = [
            keras.Input(shape=tuple(
                sm.output.shape[1:]), name=f"{layer.name}.{i}")
            for i, sm in enumerate(sub_models)
        ]
        combine = keras.layers.Add(name=f"{layer.name}.add")(inputs)
        activation_name = layer.get_config()["activation"]
        output = (
            keras.layers.Activation(activation_name)(combine)
            if activation_name not in ["linear", None]
            else combine
        )
    elif axis == 1:
        inputs = [
            keras.Input(shape=tuple(
                sm.output.shape[1:]), name=f"{layer.name}.{i}")
            for i, sm in enumerate(sub_models)
        ]
        output = keras.layers.Concatenate(name=f"{layer.name}.concat")(inputs)

    combiner = keras.Model(inputs=inputs, outputs=output, name=f"{layer.name}_combiner")
    sub_models.append(combiner)

    return sub_models


SplitType = Literal["filters", "in_channels", "spatial_h", "spatial_w", "spatial_hw"]


def _partition_1d(n: int, parts: int) -> list[tuple[int, int]]:
    if parts <= 0:
        raise ValueError("parts must be >= 1")
    if n <= 0:
        raise ValueError("n must be >= 1")
    if parts > n:
        raise ValueError(f"Cannot partition length {n} into {parts} non-empty parts")

    base = n // parts
    rem = n % parts
    ranges = []
    start = 0
    for i in range(parts):
        size = base + (1 if i < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def _conv2d_output_dim(
    in_dim: int, k_eff: int, stride: int, padding: str
) -> int:
    if padding == "valid":
        return (in_dim - k_eff) // stride + 1
    if padding == "same":
        return math.ceil(in_dim / stride)
    raise NotImplementedError(f"Unsupported padding: {padding!r}")


@dataclass(frozen=True)
class _SamePadding:
    top: int
    bottom: int
    left: int
    right: int


def _compute_same_padding_2d(
    h_in: int, w_in: int, k_eff_h: int, k_eff_w: int, sh: int, sw: int
) -> _SamePadding:
    h_out = math.ceil(h_in / sh)
    w_out = math.ceil(w_in / sw)

    pad_h = max(0, (h_out - 1) * sh + k_eff_h - h_in)
    pad_w = max(0, (w_out - 1) * sw + k_eff_w - w_in)

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return _SamePadding(top, bottom, left, right)


def _infer_input_shape_for_layer(layer: keras.layers.Layer, input_shape=None) -> tuple[int, ...]:
    if input_shape is not None:
        return tuple(input_shape)

    # Try connected layer first
    try:
        x = layer.input
        if x is not None and getattr(x, "shape", None) is not None:
            shp = tuple(x.shape)[1:]
            if all(s is not None for s in shp):
                return shp
            return shp  # may include None; caller can decide
    except Exception:
        pass

    # Try input_shape property (may exist)
    shp = getattr(layer, "input_shape", None)
    if shp is not None:
        if isinstance(shp, (list, tuple)) and shp and isinstance(shp[0], (list, tuple)):
            # multi-input layer; not supported here
            raise ValueError("conv2d_split expects a single-input Conv2D layer")
        return tuple(shp)[1:] if len(shp) and shp[0] is None else tuple(shp)

    raise ValueError(
        "Could not infer input shape. Pass input_shape=(H, W, C) (channels_last) "
        "or (C, H, W) (channels_first)."
    )


def _clone_conv2d(layer: keras.layers.Conv2D, **overrides) -> keras.layers.Conv2D:
    cfg = layer.get_config()
    cfg.update(overrides)
    return keras.layers.Conv2D.from_config(cfg)


def _channel_axis(data_format: str) -> int:
    return -1 if data_format == "channels_last" else 1


def _height_width_axes(data_format: str) -> tuple[int, int]:
    return (1, 2) if data_format == "channels_last" else (2, 3)


class _BiasAdd(keras.layers.Layer):
    """Adds a fixed bias vector (broadcast across spatial dims)."""

    def __init__(self, bias, data_format="channels_last", name=None):
        super().__init__(name=name)
        self._bias_value = bias
        self.data_format = data_format

    def build(self, input_shape):
        # Store bias as a non-trainable weight so it serializes and moves with the model.
        b = keras.ops.convert_to_tensor(self._bias_value)
        self.bias = self.add_weight(
            name="bias",
            shape=tuple(b.shape),
            initializer="zeros",
            trainable=False,
        )
        self.bias.assign(b)

    def call(self, x):
        # x: (N, H, W, C) or (N, C, H, W)
        if self.data_format == "channels_last":
            return x + self.bias
        # channels_first
        return x + keras.ops.reshape(self.bias, (1, -1, 1, 1))


def conv2d_split(
    conv: keras.layers.Conv2D,
    split_type: SplitType,
    partitions: int | tuple[int, int],
    *,
    input_shape=None,
    name_prefix: str | None = None,
) -> tuple[list[keras.Model], keras.Model]:
    """
    Split a Conv2D layer into multiple submodels + a combiner that reconstitutes
    the original Conv2D output exactly.

    Args:
        conv: a built keras.layers.Conv2D (weights available).
        split_type: "filters", "in_channels", "spatial_h", "spatial_w", "spatial_hw"
        partitions:
            - for "filters"/"in_channels"/"spatial_h"/"spatial_w": int
            - for "spatial_hw": (parts_h, parts_w) or int (interpreted as square grid)
        input_shape: optional input shape excluding batch, used especially for spatial splits.
        name_prefix: optional naming prefix.

    Returns:
        (submodels, combiner_model)
    """
    if not isinstance(conv, keras.layers.Conv2D):
        raise TypeError(f"Expected keras.layers.Conv2D, got {type(conv)!r}")

    if conv.groups != 1:
        raise NotImplementedError("conv2d_split currently supports Conv2D with groups=1 only.")

    if not conv.built:
        # We need weights to slice/copy. You can conv.build(input_shape_with_batch) beforehand.
        raise ValueError("Conv2D layer must be built (have weights) before splitting.")

    prefix = name_prefix or conv.name
    data_format = conv.data_format or "channels_last"
    ch_axis = _channel_axis(data_format)
    h_axis, w_axis = _height_width_axes(data_format)

    # Infer input shape (may contain None; spatial split needs concrete H/W)
    in_shape = _infer_input_shape_for_layer(conv, input_shape=input_shape)

    # Standard Keras kernel layout is (kh, kw, in_c, out_c)
    kernel, bias = (conv.get_weights() + [None])[:2]  # bias may be missing
    kh, kw, in_c, out_c = kernel.shape

    sh, sw = conv.strides
    dh, dw = conv.dilation_rate
    k_eff_h = (kh - 1) * dh + 1
    k_eff_w = (kw - 1) * dw + 1

    # ---------- 1) Split by output filters ----------
    if split_type == "filters":
        if not isinstance(partitions, int):
            raise TypeError("partitions must be an int for split_type='filters'")

        ranges = _partition_1d(out_c, partitions)
        submodels: list[keras.Model] = []
        inputs = []

        # Build submodels
        for i, (s, e) in enumerate(ranges):
            x_in = keras.Input(shape=in_shape, name=f"{prefix}.in.{i}")
            part = _clone_conv2d(
                conv,
                filters=(e - s),
                name=f"{prefix}.filters.{i}",
            )
            y = part(x_in)
            # Set weights
            part_kernel = kernel[:, :, :, s:e]
            if conv.use_bias:
                part_bias = bias[s:e]
                part.set_weights([part_kernel, part_bias])
            else:
                part.set_weights([part_kernel])

            submodels.append(keras.Model(x_in, y, name=f"{prefix}.sub.{i}"))
            inputs.append(keras.Input(shape=tuple(y.shape)[1:], name=f"{prefix}.comb_in.{i}"))

        # Combiner: concat along channel axis
        concat = keras.layers.Concatenate(axis=ch_axis, name=f"{prefix}.combine.concat")
        y_out = concat(inputs)
        combiner = keras.Model(inputs=inputs, outputs=y_out, name=f"{prefix}.combiner")

        return submodels, combiner

    # ---------- 2) Split by input channels (kernel split) ----------
    if split_type == "in_channels":
        if not isinstance(partitions, int):
            raise TypeError("partitions must be an int for split_type='in_channels'")
        ranges = _partition_1d(in_c, partitions)

        submodels = []
        comb_inputs = []

        # Conv parts: use_bias=False; add bias once in combiner if original had bias
        for i, (s, e) in enumerate(ranges):
            x_in = keras.Input(shape=in_shape, name=f"{prefix}.in.{i}")

            # Slice input channels
            def _slice_channels(x, s=s, e=e):
                if data_format == "channels_last":
                    return x[:, :, :, s:e]
                return x[:, s:e, :, :]

            x_part = keras.layers.Lambda(
                _slice_channels, name=f"{prefix}.slice_inch.{i}"
            )(x_in)

            part = _clone_conv2d(
                conv,
                use_bias=False,
                name=f"{prefix}.inch.{i}",
            )
            y = part(x_part)

            # Set weights (slice kernel on in-channel axis)
            part_kernel = kernel[:, :, s:e, :]
            part.set_weights([part_kernel])

            submodels.append(keras.Model(x_in, y, name=f"{prefix}.sub.{i}"))
            comb_inputs.append(keras.Input(shape=tuple(y.shape)[1:], name=f"{prefix}.comb_in.{i}"))

        added = keras.layers.Add(name=f"{prefix}.combine.add")(comb_inputs)
        if conv.use_bias:
            added = _BiasAdd(bias, data_format=data_format, name=f"{prefix}.combine.bias")(added)

        combiner = keras.Model(inputs=comb_inputs, outputs=added, name=f"{prefix}.combiner")
        return submodels, combiner

    # ---------- 3) Spatial tiling splits (exact output tiling) ----------
    if split_type in {"spatial_h", "spatial_w", "spatial_hw"}:
        if isinstance(partitions, int):
            parts_h = parts_w = partitions if split_type == "spatial_hw" else None
            if split_type == "spatial_h":
                parts_h, parts_w = partitions, 1
            elif split_type == "spatial_w":
                parts_h, parts_w = 1, partitions
            elif split_type == "spatial_hw":
                parts_h, parts_w = partitions, partitions
        else:
            if split_type != "spatial_hw":
                raise TypeError("tuple partitions is only valid for split_type='spatial_hw'")
            parts_h, parts_w = partitions

        # Need concrete H/W to compute output tiling
        if data_format == "channels_last":
            h_in, w_in, _ = in_shape
        else:
            _, h_in, w_in = in_shape

        if h_in is None or w_in is None:
            raise ValueError(
                "Spatial splits require static input height/width. "
                "Pass input_shape with concrete H and W."
            )

        padding = conv.padding.lower()
        if padding not in {"same", "valid"}:
            raise NotImplementedError(f"Unsupported padding for spatial split: {padding!r}")

        # Compute output H/W
        h_out = _conv2d_output_dim(h_in, k_eff_h, sh, padding)
        w_out = _conv2d_output_dim(w_in, k_eff_w, sw, padding)

        # Convert implicit padding='same' to explicit ZeroPadding2D + padding='valid'
        if padding == "same":
            pad = _compute_same_padding_2d(h_in, w_in, k_eff_h, k_eff_w, sh, sw)
            pad_layer = keras.layers.ZeroPadding2D(
                padding=((pad.top, pad.bottom), (pad.left, pad.right)),
                data_format=data_format,
                name=f"{prefix}.pad",
            )
        else:
            pad = _SamePadding(0, 0, 0, 0)
            pad_layer = keras.layers.Identity(name=f"{prefix}.pad")

        # Conv clone with padding='valid' (since we explicitly padded if needed)
        conv_valid = _clone_conv2d(conv, padding="valid", name=f"{prefix}.conv_valid")

        # Build + set weights once (we'll clone per-submodel to avoid layer-sharing issues)
        # We will create a fresh conv_valid per tile so weights are copied, not shared.
        # (Sharing is possible but can be annoying with serialization / testing.)

        # Partition OUTPUT space
        h_ranges = _partition_1d(h_out, parts_h)
        w_ranges = _partition_1d(w_out, parts_w)

        submodels = []
        tile_inputs_for_combiner = []
        tile_output_shapes = []

        tile_index = 0
        for hi, (oh0, oh1) in enumerate(h_ranges):
            for wi, (ow0, ow1) in enumerate(w_ranges):
                # Required input region on PADDED input for this output tile:
                # ih0 = oh0*sh
                # ih1 = (oh1-1)*sh + k_eff_h
                ih0 = oh0 * sh
                ih1 = (oh1 - 1) * sh + k_eff_h
                iw0 = ow0 * sw
                iw1 = (ow1 - 1) * sw + k_eff_w

                x_in = keras.Input(shape=in_shape, name=f"{prefix}.in.{tile_index}")
                x_pad = pad_layer(x_in)

                def _slice_hw(x, ih0=ih0, ih1=ih1, iw0=iw0, iw1=iw1):
                    if data_format == "channels_last":
                        return x[:, ih0:ih1, iw0:iw1, :]
                    return x[:, :, ih0:ih1, iw0:iw1]

                x_tile = keras.layers.Lambda(_slice_hw, name=f"{prefix}.slice.{hi}.{wi}")(x_pad)

                # Fresh conv for this tile
                part_conv = _clone_conv2d(conv, padding="valid", name=f"{prefix}.tileconv.{hi}.{wi}")
                y_tile = part_conv(x_tile)

                # Copy weights from original conv
                if conv.use_bias:
                    part_conv.set_weights([kernel, bias])
                else:
                    part_conv.set_weights([kernel])

                submodels.append(keras.Model(x_in, y_tile, name=f"{prefix}.sub.{tile_index}"))

                out_shape = tuple(y_tile.shape)[1:]
                tile_output_shapes.append(out_shape)
                tile_inputs_for_combiner.append(
                    keras.Input(shape=out_shape, name=f"{prefix}.comb_in.{tile_index}")
                )

                tile_index += 1

        # Combiner: stitch tiles back.
        # Row-major order: for each hi, concatenate across width, then concatenate rows across height.
        # (Axes depend on data_format.)
        row_tensors = []
        idx = 0
        for hi in range(len(h_ranges)):
            row = tile_inputs_for_combiner[idx : idx + len(w_ranges)]
            idx += len(w_ranges)
            if len(row) == 1:
                row_tensors.append(row[0])
            else:
                row_tensors.append(
                    keras.layers.Concatenate(axis=w_axis, name=f"{prefix}.combine.row{hi}")(row)
                )

        if len(row_tensors) == 1:
            stitched = row_tensors[0]
        else:
            stitched = keras.layers.Concatenate(axis=h_axis, name=f"{prefix}.combine.col")(row_tensors)

        combiner = keras.Model(
            inputs=tile_inputs_for_combiner,
            outputs=stitched,
            name=f"{prefix}.combiner",
        )
        return submodels, combiner

    raise ValueError(f"Unknown split_type: {split_type!r}")
