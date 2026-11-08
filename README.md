# mltosis
Machine-learning Toolkit for Splitting Inferences Systematically

![tests](https://github.com/cegranger/mltosis/actions/workflows/tests.yml/badge.svg)

## Changelog

- **v0.1.2** — `dense_split` / `qdense_split` now use the same balanced split as `conv2d_split`
  (`_partition_1d`): partition sizes differ by at most one, the first partitions taking the remainder
  (e.g. 128 neurons in 43 partitions: 42 × 3 + 1 × 2 instead of 42 × 2 + 1 × 44). Before v0.1.2 the last
  partition received the whole remainder of the integer division. Regression tests check the partition
  sizes and the exactness of the recombined output for uneven splits.

- **v0.1.1** — Fix `conv2d_split` / `qconv2d_split` with `split_type="in_channels"` for non-linear
  activations. In v0.1.0 each submodel kept the original activation (and, for HGQ2 `QConv2D`, the output
  quantizer), so the combined output was `relu(a) + relu(b)` instead of `relu(a + b)`. The submodels now
  compute linear partial convolutions (bias only in the first one, no output quantizer) and the combiner
  applies sum → original activation → original output quantizer, as `dense_split` already does.
  Regression tests cover ReLU/tanh, biases, 2/3/7 partitions, `same`/`valid` padding and HGQ2 `enable_oq`.

## Known limitations (v0.1.2)

- With `split_type="in_channels"`, the submodels output partial sums: when converted to fixed point
  (e.g. hls4ml), each partial sum is rounded to the submodel's output precision before the combiner adds
  them, which can differ slightly from the unsplit layer's accumulator. Choose the partial-sum precision
  accordingly.
- Partitions are balanced by size (number of inputs, neurons, filters, channels or rows), not by cost: with
  per-weight quantization (HGQ2) and pruning, partitions of equal size can need very different resources.
- `conv2d_split` supports `groups=1` only.
- `keras2.py` (Keras 2 / QKeras) is a stub.
