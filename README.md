# mltosis
Machine-learning Toolkit for Splitting Inferences Systematically

![tests](https://github.com/cegranger/mltosis/actions/workflows/tests.yml/badge.svg)

## Known limitations (v0.1.0)

- **`conv2d_split` / `qconv2d_split` with `split_type="in_channels"` require a linear activation.**
  Each submodel keeps the original layer config, activation included, and the combiner only sums the
  partial outputs and adds the bias. With a built-in non-linear activation (e.g. `Conv2D(..., activation="relu")`)
  the combined output is `relu(a) + relu(b)` instead of `relu(a + b)`. The same applies to HGQ2 `QConv2D`
  layers that quantize their output internally (each partial sum is rounded before the sum).
  **Workaround:** use `activation=None` in the convolution and a separate `keras.layers.Activation` layer.
  **Planned fix:** strip the activation (and output quantization) from the submodels and apply it in the
  combiner after the sum and bias, as `dense_split` already does. The current tests do not cover this case
  (no activation in `tests/test_keras3_conv2d_split.py`).
- **`dense_split` / `qdense_split` give the whole remainder to the last partition.** Inputs or neurons are split
  with integer division and the last partition receives the remainder (e.g. 128 neurons in 43 partitions:
  42 × 2 + 1 × 44). `conv2d_split` uses a balanced split (`_partition_1d`).
- `conv2d_split` supports `groups=1` only.
- `keras2.py` (Keras 2 / QKeras) is a stub.
