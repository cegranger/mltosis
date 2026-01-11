# tests/test_hgq2_qconv2d_split.py
#
# Pytest suite for qconv2d_split() with HGQ2 QConv2D layers.
#
# What it tests (when HGQ2 is installed and a QConv2D can be constructed):
# - "filters" split reproduces original output
# - "spatial_h", "spatial_w", "spatial_hw" splits reproduce original output
# - "in_channels" split reproduces original output when output/activation quantization
#   is NOT applied inside QConv2D (otherwise xfail with a clear reason)
#
# Notes:
# - You MUST adjust the import path for qconv2d_split.
# - HGQ2 package import paths can vary; this test tries a few common ones.
# - If your QConv2D requires mandatory quantizer kwargs, set them in the
#   `qconv_ctor_kwargs` fixture below.

from __future__ import annotations

import numpy as np
import pytest
import keras

from mltosis.keras3 import qconv2d_split


def _to_numpy(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


def _import_hgq2_qconv2d():
    """
    Try a few common HGQ2 import locations.
    Update/extend if your project uses a different path.
    """
    candidates = [
        ("hgq2.layers", "QConv2D"),
        ("hgq2.keras.layers", "QConv2D"),
        ("hgq.layers", "QConv2D"),
        ("hgq.keras.layers", "QConv2D"),
    ]
    for mod_name, sym in candidates:
        try:
            mod = __import__(mod_name, fromlist=[sym])
            return getattr(mod, sym)
        except Exception:
            continue
    return None


QConv2D = _import_hgq2_qconv2d()


pytestmark = pytest.mark.skipif(QConv2D is None, reason="HGQ2 QConv2D not importable in this environment.")


@pytest.fixture
def qconv_ctor_kwargs():
    """
    If HGQ2 QConv2D requires quantizer arguments, set them here.

    Examples (PLACEHOLDERS — adjust to your HGQ2 API):
      return {"kernel_quantizer": "hgq8", "bias_quantizer": None}
      return {"quantizer": "int8"}  # if HGQ2 uses a single quantizer kwarg

    By default, we try constructing QConv2D with no extra kwargs.
    """
    return {}


def _make_built_qconv(
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
    ctor_kwargs=None,
):
    keras.utils.set_random_seed(seed)
    ctor_kwargs = dict(ctor_kwargs or {})

    try:
        qconv = QConv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            use_bias=use_bias,
            data_format=data_format,
            groups=groups,
            name="qconv",
            **ctor_kwargs,
        )
    except TypeError as e:
        pytest.skip(f"Could not construct HGQ2 QConv2D with provided kwargs: {e}")

    # Build by calling once
    x = keras.Input(shape=input_shape, name="x")
    _ = qconv(x)

    # Deterministic kernel/bias assignment (only for the "main" weights).
    # Extra quantizer weights (scales, clip params, etc.) remain as initialized.
    if not hasattr(qconv, "kernel"):
        pytest.skip("This HGQ2 QConv2D does not expose .kernel; tests need adaptation.")

    rng = np.random.default_rng(seed)
    kernel = (rng.standard_normal(tuple(qconv.kernel.shape)) * 0.05).astype("float32")
    qconv.kernel.assign(kernel)

    if getattr(qconv, "use_bias", False):
        if not hasattr(qconv, "bias"):
            pytest.skip("QConv2D claims use_bias=True but does not expose .bias.")
        bias = (rng.standard_normal(tuple(qconv.bias.shape)) * 0.01).astype("float32")
        qconv.bias.assign(bias)

    return qconv


def _clone_layer_with_same_weights(layer):
    clone = layer.__class__.from_config(layer.get_config())
    in_shape = tuple(layer.input.shape)[1:]
    clone(keras.Input(shape=in_shape))
    clone.set_weights(layer.get_weights())
    return clone


def _has_internal_output_or_activation_quantization(qconv) -> bool:
    """
    Heuristic: if config suggests activation/output quantization occurs inside the layer,
    then splitting by input channels may not be exactly equivalent unless the combiner
    reproduces that quantization step.

    Update this logic if you know the exact HGQ2 config keys.
    """
    try:
        cfg = qconv.get_config()
    except Exception:
        return False

    suspicious_keys = [
        "activation_quantizer",
        "act_quantizer",
        "output_quantizer",
        "out_quantizer",
        "activation",
        "quantize_activation",
        "quantize_output",
    ]
    for k in suspicious_keys:
        if k in cfg and cfg[k] not in (None, "linear", "identity", False):
            return True
    return False


def _make_ref_and_reconstructed_models(qconv, split_type, partitions, input_shape):
    submodels, combiner = qconv2d_split(
        qconv,
        split_type=split_type,
        partitions=partitions,
        input_shape=input_shape,
        name_prefix=f"{qconv.name}.{split_type}",
    )

    ref_layer = _clone_layer_with_same_weights(qconv)

    x = keras.Input(shape=input_shape, name="x_in")
    y_ref = ref_layer(x)

    parts = [m(x, training=False) for m in submodels]
    y_rec = combiner(parts, training=False)

    ref_model = keras.Model(x, y_ref, name="ref")
    rec_model = keras.Model(x, y_rec, name="reconstructed")
    return ref_model, rec_model, submodels, combiner


@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("padding", ["valid", "same"])
def test_qconv2d_split_filters_matches_original(qconv_ctor_kwargs, use_bias, padding):
    input_shape = (12, 10, 6)
    qconv = _make_built_qconv(
        input_shape=input_shape,
        filters=10,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=padding,
        use_bias=use_bias,
        seed=1,
        ctor_kwargs=qconv_ctor_kwargs,
    )

    ref_model, rec_model, submodels, _ = _make_ref_and_reconstructed_models(
        qconv, "filters", partitions=4, input_shape=input_shape
    )

    x = (np.random.default_rng(0).standard_normal((2,) + input_shape)).astype("float32")
    y_ref = _to_numpy(ref_model(x, training=False))
    y_rec = _to_numpy(rec_model(x, training=False))

    assert len(submodels) == 4
    np.testing.assert_allclose(y_rec, y_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("use_bias", [True, False])
def test_qconv2d_split_in_channels_matches_original_when_safe(qconv_ctor_kwargs, use_bias):
    input_shape = (9, 11, 7)  # in_channels=7 (not divisible by partitions on purpose)
    qconv = _make_built_qconv(
        input_shape=input_shape,
        filters=8,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        seed=2,
        ctor_kwargs=qconv_ctor_kwargs,
    )

    if _has_internal_output_or_activation_quantization(qconv):
        pytest.xfail(
            "HGQ2 QConv2D appears to quantize activation/output inside the layer; "
            "in_channels splitting may require moving that quantization into the combiner."
        )

    ref_model, rec_model, submodels, _ = _make_ref_and_reconstructed_models(
        qconv, "in_channels", partitions=3, input_shape=input_shape
    )

    x = (np.random.default_rng(3).standard_normal((3,) + input_shape)).astype("float32")
    y_ref = _to_numpy(ref_model(x, training=False))
    y_rec = _to_numpy(rec_model(x, training=False))

    assert len(submodels) == 3
    np.testing.assert_allclose(y_rec, y_ref, rtol=1e-5, atol=1e-5)


def test_qconv2d_split_spatial_h_matches_original_stride2_same(qconv_ctor_kwargs):
    input_shape = (15, 14, 4)
    qconv = _make_built_qconv(
        input_shape=input_shape,
        filters=6,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=True,
        seed=4,
        ctor_kwargs=qconv_ctor_kwargs,
    )

    ref_model, rec_model, submodels, _ = _make_ref_and_reconstructed_models(
        qconv, "spatial_h", partitions=3, input_shape=input_shape
    )

    x = (np.random.default_rng(5).standard_normal((2,) + input_shape)).astype("float32")
    y_ref = _to_numpy(ref_model(x, training=False))
    y_rec = _to_numpy(rec_model(x, training=False))

    assert len(submodels) == 3
    np.testing.assert_allclose(y_rec, y_ref, rtol=1e-5, atol=1e-5)


def test_qconv2d_split_spatial_w_matches_original_dilation_valid(qconv_ctor_kwargs):
    input_shape = (16, 17, 3)
    qconv = _make_built_qconv(
        input_shape=input_shape,
        filters=5,
        kernel_size=(3, 3),
        strides=(1, 1),
        dilation_rate=(2, 2),
        padding="valid",
        use_bias=False,
        seed=6,
        ctor_kwargs=qconv_ctor_kwargs,
    )

    ref_model, rec_model, submodels, _ = _make_ref_and_reconstructed_models(
        qconv, "spatial_w", partitions=4, input_shape=input_shape
    )

    x = (np.random.default_rng(7).standard_normal((1,) + input_shape)).astype("float32")
    y_ref = _to_numpy(ref_model(x, training=False))
    y_rec = _to_numpy(rec_model(x, training=False))

    assert len(submodels) == 4
    np.testing.assert_allclose(y_rec, y_ref, rtol=1e-5, atol=1e-5)


def test_qconv2d_split_spatial_hw_matches_original(qconv_ctor_kwargs):
    input_shape = (13, 13, 4)
    qconv = _make_built_qconv(
        input_shape=input_shape,
        filters=7,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=True,
        seed=8,
        ctor_kwargs=qconv_ctor_kwargs,
    )

    ref_model, rec_model, submodels, _ = _make_ref_and_reconstructed_models(
        qconv, "spatial_hw", partitions=(2, 3), input_shape=input_shape
    )

    x = (np.random.default_rng(9).standard_normal((2,) + input_shape)).astype("float32")
    y_ref = _to_numpy(ref_model(x, training=False))
    y_rec = _to_numpy(rec_model(x, training=False))

    assert len(submodels) == 2 * 3
    np.testing.assert_allclose(y_rec, y_ref, rtol=1e-5, atol=1e-5)


def test_qconv2d_split_groups_not_supported(qconv_ctor_kwargs):
    input_shape = (8, 8, 4)
    qconv = _make_built_qconv(
        input_shape=input_shape,
        filters=6,
        kernel_size=(3, 3),
        groups=2,
        seed=12,
        ctor_kwargs=qconv_ctor_kwargs,
    )

    with pytest.raises(NotImplementedError):
        qconv2d_split(qconv, split_type="filters", partitions=2, input_shape=input_shape)