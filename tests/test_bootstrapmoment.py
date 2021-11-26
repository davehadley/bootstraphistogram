import itertools

import numpy as np

from bootstraphistogram import BootstrapMoment


def array_almost_equal(
    actual: np.ndarray,
    expected: np.ndarray,
    delta: float,
) -> None:
    return np.all(np.abs(actual - expected) < delta)


def test_bootstrapmoment_mean():
    moment = BootstrapMoment(numsamples=1000, rng=1234)
    values = np.arange(100, dtype=float)
    moment.fill(values)
    assert abs(moment.mean().nominal - np.average(values)) < 0.001
    assert abs(np.average(moment.mean().samples) - np.average(values)) < 1.0


def test_bootstrapmoment_std_deviation():
    moment = BootstrapMoment(numsamples=10000, rng=1234)
    values = np.arange(100, dtype=float)
    moment.fill(values)
    assert abs(moment.variance().nominal - np.var(values)) < 0.01
    assert abs(np.average(moment.variance().samples) - np.var(values)) < 10.0
    assert abs(moment.std().nominal - np.std(values)) < 0.01
    assert abs(np.average(moment.std().samples) - np.std(values)) < 1.0


def _skewness(array: np.ndarray) -> np.ndarray:
    mu = np.average(array)
    sigma = np.std(array)
    return np.average(((array - mu) / sigma) ** 3)


def test_bootstrapmoment_skewness():
    moment = BootstrapMoment(numsamples=10000, rng=1234)
    values = np.arange(100, dtype=float)
    moment.fill(values)
    assert abs(moment.skewness().nominal - _skewness(values)) < 0.01
    assert abs(np.average(moment.skewness().samples) - _skewness(values)) < 10.0


def test_bootstrap_correlations():
    moment = BootstrapMoment(numsamples=10000, rng=1234)
    values = [1]
    moment.fill(values)
    w = moment._sum_w.samples.view().flatten()
    t1 = moment._sum_wt.samples.view().flatten()
    t2 = moment._sum_wt2.samples.view().flatten()
    t3 = moment._sum_wt3.samples.view().flatten()
    cor = np.corrcoef([w, t1, t2, t3])
    for row, column in itertools.combinations(range(4), 2):
        assert abs(cor[row, column] - 1.0) < 1e-6


def test_bootstrap_correlations_many_samples():
    moment = BootstrapMoment(numsamples=10000, rng=1234)
    values = np.arange(100, dtype=float)
    moment.fill(values)
    w = moment._sum_w.samples.view().flatten()
    t1 = moment._sum_wt.samples.view().flatten()
    t2 = moment._sum_wt2.samples.view().flatten()
    t3 = moment._sum_wt3.samples.view().flatten()
    cor = np.corrcoef([w, t1, t2, t3])
    for row, column in itertools.combinations(range(4), 2):
        assert cor[row, column] > 0.5
