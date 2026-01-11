"""Keras 3.x split methods. Includes HGQ2
"""
from __future__ import annotations

import keras

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

import hgq
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
                for i in [14, 13, 12, 11, 1]:
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