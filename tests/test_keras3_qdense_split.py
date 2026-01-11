# tests/test_keras3_qdense_split.py
import os

# Must be set before importing keras (and anything that imports keras).
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pytest

import keras

hgq = pytest.importorskip("hgq", reason="hgq is not installed; skipping QDense split tests")

from mltosis.keras3 import qdense_split


def _forward(model: keras.Model, x: np.ndarray) -> np.ndarray:
    y = model(x, training=False)
    return keras.ops.convert_to_numpy(y)


def _set_deterministic_weights(layer) -> list[np.ndarray]:
    """
    Create deterministic weights matching layer.get_weights() shapes,
    then set them back onto the layer.
    """
    og = layer.get_weights()
    new = []
    for i, w in enumerate(og):
        if np.ndim(w) == 0:
            # scalar
            new.append(np.array(0.1 * (i + 1), dtype=w.dtype))
        else:
            arr = (np.arange(w.size, dtype="float32").reshape(w.shape) / 50.0).astype(w.dtype)
            # add a little offset per weight tensor so they're not too symmetric
            arr = arr + np.array((i + 1) / 100.0, dtype=w.dtype)
            new.append(arr)
    layer.set_weights(new)
    return new


def _build_qdense_model(in_dim: int, units: int, activation: str = "linear"):
    """
    Build a tiny model: Input -> QDense -> Output
    If hgq's API differs, skip with a clear message.
    """
    x_in = keras.Input(shape=(in_dim,), name="x")
    try:
        layer = hgq.layers.QDense(units, activation=activation, use_bias=True, name="qdense")
    except Exception as e:
        pytest.skip(f"Could not construct hgq.layers.QDense with minimal args: {e}")

    y_out = layer(x_in)
    model = keras.Model(x_in, y_out)
    # Ensure weights exist
    _ = model(np.zeros((1, in_dim), dtype="float32"), training=False)
    return model, layer


def _run_split_pipeline_axis0(x: np.ndarray, parts: list[keras.Model], combiner: keras.Model) -> np.ndarray:
    # axis=0: each part consumes a slice of the input; combiner adds
    outs = []
    start = 0
    for sm in parts:
        part_in_dim = int(sm.input_shape[-1])
        x_part = x[:, start : start + part_in_dim]
        start += part_in_dim
        outs.append(_forward(sm, x_part))
    return _forward(combiner, outs)


def _run_split_pipeline_axis1(x: np.ndarray, parts: list[keras.Model], combiner: keras.Model) -> np.ndarray:
    # axis=1: each part consumes full input; combiner concatenates
    outs = [_forward(sm, x) for sm in parts]
    return _forward(combiner, outs)


@pytest.mark.parametrize("activation", ["linear", "relu"])
def test_qdense_split_axis0_matches_original_with_remainder(activation: str):
    in_dim, units = 10, 7
    n_parts = 3  # remainder on input split: 10 -> (3,3,4)

    model, layer = _build_qdense_model(in_dim, units, activation=activation)
    _set_deterministic_weights(layer)

    x = np.linspace(-1, 1, num=6 * in_dim, dtype="float32").reshape(6, in_dim)
    y_ref = _forward(model, x)

    sub_models = qdense_split(layer, axis=0, number_of_partitions=n_parts)
    parts, combiner = sub_models[:-1], sub_models[-1]

    # basic sanity: input dims across parts should sum to original in_dim
    assert sum(int(sm.input_shape[-1]) for sm in parts) == in_dim

    y_split = _run_split_pipeline_axis0(x, parts, combiner)

    np.testing.assert_allclose(y_split, y_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("activation", ["linear", "relu"])
def test_qdense_split_axis1_matches_original_with_remainder(activation: str):
    in_dim, units = 8, 10
    n_parts = 4  # remainder on neuron split: 10 -> (2,2,2,4)

    model, layer = _build_qdense_model(in_dim, units, activation=activation)
    _set_deterministic_weights(layer)

    rng = np.random.RandomState(0)
    x = rng.randn(5, in_dim).astype("float32")
    y_ref = _forward(model, x)

    sub_models = qdense_split(layer, axis=1, number_of_partitions=n_parts)
    parts, combiner = sub_models[:-1], sub_models[-1]

    # sanity: output dims across parts should sum to original units
    assert sum(int(sm.output_shape[-1]) for sm in parts) == units

    y_split = _run_split_pipeline_axis1(x, parts, combiner)

    np.testing.assert_allclose(y_split, y_ref, rtol=1e-5, atol=1e-6)


def test_qdense_split_rejects_invalid_axis():
    model, layer = _build_qdense_model(4, 3, activation="linear")
    _set_deterministic_weights(layer)

    with pytest.raises(ValueError):
        qdense_split(layer, axis=2, number_of_partitions=2)