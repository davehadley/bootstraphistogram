import numpy as np

from bootstraphistogram import BootstrapMoment


def array_almost_equal(
    actual: np.ndarray,
    expected: np.ndarray,
    delta: float,
) -> None:
    return np.all(np.abs(actual - expected) < delta)


def test_boostrapmoment_mean():
    moment = BootstrapMoment(numsamples=1000, rng=1234)
    values = np.arange(100, dtype=float)
    moment.fill(values)
    assert abs(moment.mean().nominal - np.average(values)) < 0.001
    assert abs(np.average(moment.mean().samples) - np.average(values)) < 1.0


def test_boostrapmoment_std_deviation():
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


def test_boostrapmoment_skewness():
    moment = BootstrapMoment(numsamples=10000, rng=1234)
    values = np.arange(100, dtype=float)
    moment.fill(values)
    assert abs(moment.skewness().nominal - _skewness(values)) < 0.01
    assert abs(np.average(moment.skewness().samples) - _skewness(values)) < 10.0
