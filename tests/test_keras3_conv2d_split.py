# tests/test_keras3_conv2d_split.py
#
# Adjust the import path below to wherever you placed conv2d_split().
# These tests check that:
# - submodels + combiner reproduce the original Conv2D output (numerically)
# - errors are raised for unsupported / invalid cases

from __future__ import annotations

import numpy as np
import pytest
import keras

from mltosis.keras3 import conv2d_split


def _to_numpy(x):
    # backend-agnostic (Keras 3)
    return np.asarray(keras.ops.convert_to_numpy(x))


def _make_built_conv(
    *,
    input_shape=(11, 13, 6),
    filters=8,
    kernel_size=(3, 3),
    strides=(1, 1),
    dilation_rate=(1, 1),
    padding="same",
    use_bias=True,
    data_format="channels_last",
    groups=1,
    seed=123,
) -> keras.layers.Conv2D:
    keras.utils.set_random_seed(seed)

    conv = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        data_format=data_format,
        groups=groups,
        name="conv",
    )

    # Build by calling once
    x = keras.Input(shape=input_shape, name="x")
    _ = conv(x)

    # Set deterministic weights (don’t rely on initializer differences across backends)
    kernel_shape = conv.kernel.shape
    rng = np.random.default_rng(seed)
    kernel = (rng.standard_normal(kernel_shape) * 0.05).astype("float32")
    if use_bias:
        bias = (rng.standard_normal((filters,)) * 0.01).astype("float32")
        conv.set_weights([kernel, bias])
    else:
        conv.set_weights([kernel])

    return conv


def _clone_conv_with_same_weights(conv: keras.layers.Conv2D) -> keras.layers.Conv2D:
    clone = keras.layers.Conv2D.from_config(conv.get_config())
    # Build clone with the same input spec
    in_shape = tuple(conv.input.shape)[1:]
    clone(keras.Input(shape=in_shape))
    clone.set_weights(conv.get_weights())
    return clone


def _make_ref_and_reconstructed_models(
    conv: keras.layers.Conv2D, split_type: str, partitions, input_shape
):
    # Split
    submodels, combiner = conv2d_split(
        conv,
        split_type=split_type,
        partitions=partitions,
        input_shape=input_shape,  # important for spatial splits
        name_prefix=f"{conv.name}.{split_type}",
    )

    # Reference model uses a cloned conv (avoid graph reuse quirks)
    conv_ref = _clone_conv_with_same_weights(conv)

    x = keras.Input(shape=input_shape, name="x_in")
    y_ref = conv_ref(x)

    parts = [m(x, training=False) for m in submodels]
    y_rec = combiner(parts, training=False)

    ref_model = keras.Model(x, y_ref, name="ref")
    rec_model = keras.Model(x, y_rec, name="reconstructed")

    return ref_model, rec_model, submodels, combiner


@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("padding", ["valid", "same"])
def test_conv2d_split_filters_matches_original(use_bias, padding):
    input_shape = (12, 10, 6)
    conv = _make_built_conv(
        input_shape=input_shape,
        filters=10,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=padding,
        use_bias=use_bias,
        seed=1,
    )

    ref_model, rec_model, submodels, combiner = _make_ref_and_reconstructed_models(
        conv, "filters", partitions=4, input_shape=input_shape
    )

    x = (np.random.default_rng(0).standard_normal((2,) + input_shape)).astype("float32")
    y_ref = _to_numpy(ref_model(x, training=False))
    y_rec = _to_numpy(rec_model(x, training=False))

    assert len(submodels) == 4
    np.testing.assert_allclose(y_rec, y_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("use_bias", [True, False])
def test_conv2d_split_in_channels_matches_original(use_bias):
    input_shape = (9, 11, 7)  # in_channels=7 (not divisible by partitions on purpose)
    conv = _make_built_conv(
        input_shape=input_shape,
        filters=8,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        seed=2,
    )

    ref_model, rec_model, submodels, combiner = _make_ref_and_reconstructed_models(
        conv, "in_channels", partitions=3, input_shape=input_shape
    )

    x = (np.random.default_rng(3).standard_normal((3,) + input_shape)).astype("float32")
    y_ref = _to_numpy(ref_model(x, training=False))
    y_rec = _to_numpy(rec_model(x, training=False))

    assert len(submodels) == 3
    np.testing.assert_allclose(y_rec, y_ref, rtol=1e-5, atol=1e-5)


def test_conv2d_split_spatial_h_matches_original_stride2_same():
    # Exercises: padding='same' + stride>1 (explicit padding path)
    input_shape = (15, 14, 4)
    conv = _make_built_conv(
        input_shape=input_shape,
        filters=6,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=True,
        seed=4,
    )

    ref_model, rec_model, submodels, combiner = _make_ref_and_reconstructed_models(
        conv, "spatial_h", partitions=3, input_shape=input_shape
    )

    x = (np.random.default_rng(5).standard_normal((2,) + input_shape)).astype("float32")
    y_ref = _to_numpy(ref_model(x, training=False))
    y_rec = _to_numpy(rec_model(x, training=False))

    assert len(submodels) == 3
    np.testing.assert_allclose(y_rec, y_ref, rtol=1e-5, atol=1e-5)


def test_conv2d_split_spatial_w_matches_original_dilation_valid():
    # Exercises: dilation + padding='valid'
    input_shape = (16, 17, 3)
    conv = _make_built_conv(
        input_shape=input_shape,
        filters=5,
        kernel_size=(3, 3),
        strides=(1, 1),
        dilation_rate=(2, 2),
        padding="valid",
        use_bias=False,
        seed=6,
    )

    ref_model, rec_model, submodels, combiner = _make_ref_and_reconstructed_models(
        conv, "spatial_w", partitions=4, input_shape=input_shape
    )

    x = (np.random.default_rng(7).standard_normal((1,) + input_shape)).astype("float32")
    y_ref = _to_numpy(ref_model(x, training=False))
    y_rec = _to_numpy(rec_model(x, training=False))

    assert len(submodels) == 4
    np.testing.assert_allclose(y_rec, y_ref, rtol=1e-5, atol=1e-5)


def test_conv2d_split_spatial_hw_matches_original():
    input_shape = (13, 13, 4)
    conv = _make_built_conv(
        input_shape=input_shape,
        filters=7,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=True,
        seed=8,
    )

    ref_model, rec_model, submodels, combiner = _make_ref_and_reconstructed_models(
        conv, "spatial_hw", partitions=(2, 3), input_shape=input_shape
    )

    x = (np.random.default_rng(9).standard_normal((2,) + input_shape)).astype("float32")
    y_ref = _to_numpy(ref_model(x, training=False))
    y_rec = _to_numpy(rec_model(x, training=False))

    assert len(submodels) == 2 * 3
    np.testing.assert_allclose(y_rec, y_ref, rtol=1e-5, atol=1e-5)


def test_conv2d_split_raises_when_partitions_exceed_filters():
    conv = _make_built_conv(input_shape=(8, 8, 3), filters=4, seed=10)
    with pytest.raises(ValueError):
        conv2d_split(conv, split_type="filters", partitions=10, input_shape=(8, 8, 3))


def test_conv2d_split_spatial_requires_static_hw():
    # spatial splits need concrete H/W
    conv = _make_built_conv(input_shape=(None, None, 3), filters=4, seed=11)
    with pytest.raises(ValueError):
        conv2d_split(
            conv, split_type="spatial_h", partitions=2, input_shape=(None, None, 3)
        )


def test_conv2d_split_groups_not_supported():
    # groups != 1 should raise NotImplementedError in current implementation
    input_shape = (8, 8, 4)
    conv = _make_built_conv(
        input_shape=input_shape,
        filters=6,
        kernel_size=(3, 3),
        groups=2,  # in_channels=4 divisible by 2
        seed=12,
    )
    with pytest.raises(NotImplementedError):
        conv2d_split(conv, split_type="filters", partitions=2, input_shape=input_shape)