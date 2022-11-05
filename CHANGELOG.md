# 0.11.0 (2022-11-04)

  - Add `ignore_nan` option to some methods to use `numpy.nanmean`, 
    `numpy.nanstd` and `numpy.nanpercentile` instead of `numpy.mean`, 
    `numpy.std` and `numpy.percentile`.
  - Add more flexibility to plotting functions to display percentiles or 
    mean/standard deviation.
  - Improved type hints.

# 0.10.0 (2022-03-24)

  - Improved type hinting.
  - Fix linter errors and bugs in CI.
  - Add a development Dockerfile.

# 0.9.0 (2021-12-09)

  - Implement BoostrapEfficiency for calculating binned efficiencies.
  - Implement binary operation (`+`) on `BootstrapMoment`.
  - Improve tests and new documentation.

# 0.8.0 (2021-11-25)

  - Implement binary operations (`+`, `-`, `*`, `/`) on histograms.
  - Fix issue [#5](https://github.com/davehadley/bootstraphistogram/issues/5).
  - Add `BootstrapMoment`, a class that calculates the first three moments of the filled data.
  - Improve tests and documentation.
  - Speed improvements when filling with small arrays.

# 0.7.0 (2021-11-10)

- Change BootstrapHistogram API to accept `numpy.ArrayLike` as input. 
- Fix issues [#1](https://github.com/davehadley/bootstraphistogram/issues/3) and [#4](https://github.com/davehadley/bootstraphistogram/issues/4).
- Improve documentation.