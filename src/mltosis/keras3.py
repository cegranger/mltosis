"""Keras 3.x split methods. Includes HGQ2
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Union

import keras
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

