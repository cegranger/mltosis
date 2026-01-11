import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # must be set before importing keras

import numpy as np
import keras

from mltosis.keras3 import dense_split


def _build_original_dense(in_dim: int, units: int, activation: str = "linear"):
    layer = keras.layers.Dense(units, activation=activation, use_bias=True, name="dense")
    x_in = keras.Input(shape=(in_dim,), name="x")
    x_out = layer(x_in)
    model = keras.Model(x_in, x_out)
    return model, layer


def _set_known_weights(layer: keras.layers.Dense, in_dim: int, units: int):
    # deterministic weights so failures are reproducible
    w = np.arange(in_dim * units, dtype="float32").reshape(in_dim, units) / 10.0
    b = (np.arange(units, dtype="float32") - units / 2.0) / 7.0
    layer.set_weights([w, b])
    return w, b


def _run_split_pipeline_axis0(x: np.ndarray, parts: list[keras.Model], combiner: keras.Model):
    # axis=0: each part consumes a slice of the input, combiner adds outputs
    part_outs = []
    start = 0
    for sm in parts:
        in_dim = int(sm.input_shape[-1])
        x_part = x[:, start : start + in_dim]
        start += in_dim
        part_outs.append(sm.predict(x_part, verbose=0))
    y = combiner.predict(part_outs, verbose=0)
    return y


def _run_split_pipeline_axis1(x: np.ndarray, parts: list[keras.Model], combiner: keras.Model):
    # axis=1: each part consumes full input, combiner concatenates outputs
    part_outs = [sm.predict(x, verbose=0) for sm in parts]
    y = combiner.predict(part_outs, verbose=0)
    return y


def test_dense_split_axis0_matches_original_with_remainder():
    in_dim, units = 10, 7
    n_parts = 3  # 10 // 3 = 3 remainder 1 goes to last split (3,3,4)

    model, layer = _build_original_dense(in_dim, units, activation="relu")
    _set_known_weights(layer, in_dim, units)

    x = np.linspace(-1, 1, num=5 * in_dim, dtype="float32").reshape(5, in_dim)
    y_ref = model.predict(x, verbose=0)

    sub_models = dense_split(layer, axis=0, number_of_partitions=n_parts)
    parts, combiner = sub_models[:-1], sub_models[-1]

    y_split = _run_split_pipeline_axis0(x, parts, combiner)

    np.testing.assert_allclose(y_split, y_ref, rtol=1e-5, atol=1e-6)


def test_dense_split_axis1_matches_original_with_remainder():
    in_dim, units = 8, 10
    n_parts = 4  # 10 // 4 = 2 remainder 2 => (2,2,2,4)

    model, layer = _build_original_dense(in_dim, units, activation="linear")
    _set_known_weights(layer, in_dim, units)

    x = np.random.RandomState(0).randn(6, in_dim).astype("float32")
    y_ref = model.predict(x, verbose=0)

    sub_models = dense_split(layer, axis=1, number_of_partitions=n_parts)
    parts, combiner = sub_models[:-1], sub_models[-1]

    y_split = _run_split_pipeline_axis1(x, parts, combiner)

    np.testing.assert_allclose(y_split, y_ref, rtol=1e-5, atol=1e-6)


def test_dense_split_rejects_invalid_axis():
    model, layer = _build_original_dense(4, 3)
    _set_known_weights(layer, 4, 3)

    try:
        dense_split(layer, axis=2, number_of_partitions=2)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for axis > 1")