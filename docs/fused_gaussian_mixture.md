# Gaussian mixture quadrature

`SplitLogGrid` integrates fixed two-width Gaussian mixtures in one Triton pass
on CUDA float32. The existing `--scatter-shape gaussian` and `partiality` heads
enable it automatically. Learned-shape mixtures and other backends use the
weighted component fallback in `Quadrature.marginalize_mixture`.

`PredictedMixtureScatter(fixed_gaussian=True)` is an explicit declaration that
every beta is fixed at two. Do not use it for a learned beta that currently
equals two. Mixture weights, component offsets, both widths and the shared
mean remain differentiable.

The kernel shares grid geometry and the background-corrected Poisson factor
across components. Each component retains a separate running maximum for its
window integral, adds its analytic tail and enters the final mixture reduction.
Backward scales four saved derivatives per component; it does not rebuild the
quadrature. Outer posterior locations still require separate inner calls.

`TORCHSX_FIXEDLOGGRID_NO_TRITON=1` retains the eager reference path. CPU,
float64, more than eight components and unsupported detection configurations
also use the existing fallback. Background mean and variance corrections are
included in the fused path.

On an A100, seeded synthetic four-component batches of 353K, 411K and 3.10M
observations took 5.2, 5.5 and 40.3 ms for marginal likelihood forward/backward,
versus 22.4, 27.1 and 130.8 ms for the component loop. These are likelihood-layer
timings, excluding the encoder, optimizer and data preparation. The benchmark
and full validation record are in the workspace experiment
`experiments/mixture_kernel_20260912/REPORT.md`.

`tests/test_split_mixture.py` checks the numerical kernel against an independent
float64 Gaussian-CDF/autograd reference and exercises addressing, background
corrections, checkpointing and network gradients. `tests/test_scatter_mixture.py`
and `tests/test_bg_mean_offset.py` cover the surrounding model and likelihood.
