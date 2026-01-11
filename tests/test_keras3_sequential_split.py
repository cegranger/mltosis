# tests/test_keras3_sequential_split.py
import os

# Must be set before importing keras (and anything importing keras).
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pytest

import keras

from mltosis.keras3 import sequential_split


def _to_numpy(y):
    if isinstance(y, (list, tuple)):
        return [keras.ops.convert_to_numpy(t) for t in y]
    return keras.ops.convert_to_numpy(y)


def _assert_allclose(a, b, rtol=1e-6, atol=1e-7):
    a_np, b_np = _to_numpy(a), _to_numpy(b)
    if isinstance(a_np, list):
        assert isinstance(b_np, list)
        assert len(a_np) == len(b_np)
        for aa, bb in zip(a_np, b_np):
            np.testing.assert_allclose(aa, bb, rtol=rtol, atol=atol)
    else:
        np.testing.assert_allclose(a_np, b_np, rtol=rtol, atol=atol)


def _call(model: keras.Model, x):
    # Keras Model always has model.inputs as a list
    if len(model.inputs) == 1:
        return model(x, training=False)
    assert isinstance(x, (list, tuple)), "Model expects multiple inputs"
    return model(list(x), training=False)


def _chain(parts: list[keras.Model], x):
    y = x
    for m in parts:
        y = _call(m, y)
    return y


def _set_deterministic_weights(model: keras.Model):
    """
    Make Dense-like layers deterministic. Works for standard keras layers;
    HGQ layers (if present) will simply keep their own weights unless they
    implement get_weights/set_weights.
    """
    for layer in model.layers:
        try:
            w = layer.get_weights()
        except Exception:
            continue
        if not w:
            continue

        new_w = []
        for i, arr in enumerate(w):
            if np.ndim(arr) == 0:
                new_w.append(np.array(0.01 * (i + 1), dtype=arr.dtype))
            else:
                base = (np.arange(arr.size, dtype="float32").reshape(arr.shape) / 100.0).astype(
                    arr.dtype
                )
                base = base + np.array((i + 1) / 1000.0, dtype=arr.dtype)
                new_w.append(base)
        try:
            layer.set_weights(new_w)
        except Exception:
            # Some layers may reject setting (rare); ignore in tests.
            pass


def test_sequential_split_simple_chain_by_names():
    x_in = keras.Input(shape=(6,), name="x")
    x = keras.layers.Dense(8, activation="relu", name="d1")(x_in)
    x = keras.layers.Dense(5, activation="relu", name="d2")(x)
    y = keras.layers.Dense(3, activation="linear", name="d3")(x)
    model = keras.Model(x_in, y, name="m")

    _set_deterministic_weights(model)

    parts = sequential_split(
        model,
        layer_groups=[["d1", "d2"], ["d3"]],
        names=["part0", "part1"],
    )

    inp = np.linspace(-1, 1, 4 * 6, dtype="float32").reshape(4, 6)
    y_ref = _call(model, inp)
    y_split = _chain(parts, inp)

    _assert_allclose(y_ref, y_split)


def test_sequential_split_simple_chain_by_indices():
    x_in = keras.Input(shape=(4,), name="x")
    a = keras.layers.Dense(7, activation="relu", name="a")(x_in)
    b = keras.layers.Dense(2, activation="linear", name="b")(a)
    model = keras.Model(x_in, b, name="m2")

    _set_deterministic_weights(model)

    # Find indices by name (avoids InputLayer index assumptions)
    idx_a = [i for i, l in enumerate(model.layers) if l.name == "a"][0]
    idx_b = [i for i, l in enumerate(model.layers) if l.name == "b"][0]

    parts = sequential_split(
        model,
        layer_groups=[[idx_a], [idx_b]],
        names=["p0", "p1"],
    )

    x = np.random.RandomState(0).randn(5, 4).astype("float32")
    y_ref = _call(model, x)
    y_split = _chain(parts, x)

    _assert_allclose(y_ref, y_split)


def test_sequential_split_multi_input_concat_then_dense():
    x1 = keras.Input(shape=(3,), name="x1")
    x2 = keras.Input(shape=(2,), name="x2")
    c = keras.layers.Concatenate(name="cat")([x1, x2])
    y = keras.layers.Dense(4, activation="relu", name="d")(c)
    model = keras.Model([x1, x2], y, name="mi")

    _set_deterministic_weights(model)

    parts = sequential_split(
        model,
        layer_groups=[["cat"], ["d"]],
        names=["cat_part", "dense_part"],
    )

    r = np.random.RandomState(1)
    in1 = r.randn(6, 3).astype("float32")
    in2 = r.randn(6, 2).astype("float32")

    y_ref = _call(model, [in1, in2])
    y_split = _chain(parts, [in1, in2])

    _assert_allclose(y_ref, y_split)


def test_sequential_split_branched_then_add():
    x_in = keras.Input(shape=(5,), name="x")
    trunk = keras.layers.Dense(6, activation="relu", name="trunk")(x_in)
    b1 = keras.layers.Dense(6, activation="linear", name="b1")(trunk)
    b2 = keras.layers.Dense(6, activation="linear", name="b2")(trunk)
    y = keras.layers.Add(name="add")([b1, b2])
    model = keras.Model(x_in, y, name="branched")

    _set_deterministic_weights(model)

    parts = sequential_split(
        model,
        layer_groups=[["trunk"], ["b1", "b2"], ["add"]],
        names=["p_trunk", "p_branches", "p_add"],
    )

    x = np.random.RandomState(2).randn(3, 5).astype("float32")
    y_ref = _call(model, x)
    y_split = _chain(parts, x)

    _assert_allclose(y_ref, y_split)


def test_sequential_split_multi_output_in_last_part():
    x_in = keras.Input(shape=(4,), name="x")
    h = keras.layers.Dense(5, activation="relu", name="h")(x_in)
    y1 = keras.layers.Dense(2, activation="linear", name="y1")(h)
    y2 = keras.layers.Dense(3, activation="linear", name="y2")(h)
    model = keras.Model(x_in, [y1, y2], name="mo")

    _set_deterministic_weights(model)

    parts = sequential_split(
        model,
        layer_groups=[["h"], ["y1", "y2"]],
        names=["p0", "p1"],
    )

    x = np.random.RandomState(3).randn(7, 4).astype("float32")
    y_ref = _call(model, x)
    y_split = _chain(parts, x)

    _assert_allclose(y_ref, y_split)


def test_sequential_split_rejects_empty_group():
    x_in = keras.Input(shape=(2,), name="x")
    y = keras.layers.Dense(1, name="d")(x_in)
    model = keras.Model(x_in, y)

    with pytest.raises(ValueError, match="Empty list"):
        sequential_split(model, layer_groups=[[]], names=["p0"])


def test_sequential_split_rejects_duplicate_layer_in_groups():
    x_in = keras.Input(shape=(2,), name="x")
    y = keras.layers.Dense(1, name="d")(x_in)
    model = keras.Model(x_in, y)

    with pytest.raises(ValueError, match="more than one group"):
        sequential_split(model, layer_groups=[["d"], ["d"]], names=["p0", "p1"])


def test_sequential_split_rejects_invalid_index():
    x_in = keras.Input(shape=(2,), name="x")
    y = keras.layers.Dense(1, name="d")(x_in)
    model = keras.Model(x_in, y)

    with pytest.raises(IndexError):
        sequential_split(model, layer_groups=[[999]], names=["p0"])


def test_sequential_split_rejects_input_layer_in_group():
    x_in = keras.Input(shape=(2,), name="x")
    y = keras.layers.Dense(1, name="d")(x_in)
    model = keras.Model(x_in, y)

    # Functional models typically have an InputLayer at model.layers[0]
    with pytest.raises(ValueError, match="Do not include InputLayer"):
        sequential_split(model, layer_groups=[[0, "d"]], names=["p0"])


def test_sequential_split_rejects_not_topologically_ordered_groups():
    x_in = keras.Input(shape=(3,), name="x")
    a = keras.layers.Dense(4, activation="relu", name="a")(x_in)
    b = keras.layers.Dense(2, activation="linear", name="b")(a)
    model = keras.Model(x_in, b)

    # Wrong order: b depends on a, but b-group is first
    with pytest.raises(ValueError, match="not topologically ordered"):
        sequential_split(model, layer_groups=[["b"], ["a"]], names=["p0", "p1"])


def test_sequential_split_rejects_shared_reused_layer():
    x_in = keras.Input(shape=(3,), name="x")
    shared = keras.layers.Dense(4, activation="relu", name="shared")
    y1 = shared(x_in)
    y2 = shared(x_in)  # reuse same layer => shared/reused
    y = keras.layers.Add(name="add")([y1, y2])
    model = keras.Model(x_in, y)

    with pytest.raises(NotImplementedError, match="reused/shared"):
        sequential_split(model, layer_groups=[["shared", "add"]], names=["p0"])

def test_sequential_split_with_hgq_qdense_smoke():
    hgq = pytest.importorskip("hgq")

    x_in = keras.Input(shape=(4,), name="x")
    try:
        q = hgq.layers.QDense(6, activation="relu", name="q")(x_in)
    except Exception as e:
        pytest.skip(f"Could not build hgq.layers.QDense: {e}")

    y = keras.layers.Dense(2, activation="linear", name="out")(q)
    model = keras.Model(x_in, y)

    _set_deterministic_weights(model)

    parts = sequential_split(model, layer_groups=[["q"], ["out"]], names=["p0", "p1"])

    x = np.random.RandomState(0).randn(5, 4).astype("float32")
    _assert_allclose(_call(model, x), _chain(parts, x))