import itertools
import pickle
from typing import Callable

import numpy as np
import pytest

from bootstraphistogram import BootstrapMoment


def test_bootstrapmoment_numsamples():
    numsamples = 1000
    moment = BootstrapMoment(numsamples=numsamples, rng=1234)
    values = np.arange(100, dtype=float)
    moment.fill(values)
    assert len(moment.mean().samples) == moment.numsamples == numsamples


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


def test_bootstrapmoment_array_shapes():
    moment = BootstrapMoment(numsamples=3, rng=1234)
    values = np.arange(100, dtype=float)
    moment.fill(values)
    assert (
        moment.mean().samples.shape
        == moment.std().samples.shape
        == moment.variance().samples.shape
        == moment.skewness().samples.shape
        == (3,)
    )


@pytest.mark.parametrize("rng", [None, 1234])
def test_bootstrapmoment_correlations(rng):
    moment = BootstrapMoment(numsamples=10000, rng=rng)
    values = [1]
    moment.fill(values)
    w = moment._sum_w.samples.view().flatten()
    t1 = moment._sum_wt.samples.view().flatten()
    t2 = moment._sum_wt2.samples.view().flatten()
    t3 = moment._sum_wt3.samples.view().flatten()
    cor = np.corrcoef([w, t1, t2, t3])
    for row, column in itertools.combinations(range(4), 2):
        assert abs(cor[row, column] - 1.0) < 1e-6


def test_bootstrapmoment_correlations_many_samples():
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


def test_pickle():
    moment1 = BootstrapMoment(numsamples=1000, rng=1234)
    values = np.arange(100, dtype=float)
    moment1.fill(values)
    moment2 = pickle.loads(pickle.dumps(moment1))
    assert moment1 == moment2


@pytest.mark.parametrize(
    "generator,mu,sigma,skewness",
    [
        (lambda: np.random.default_rng(5678).normal(size=100000), 0.0, 1.0, 0.0),
        (
            lambda: np.random.default_rng(5678).uniform(size=100000),
            0.5,
            np.sqrt(1.0 / 12.0),
            0.0,
        ),
        (lambda: np.random.default_rng(5678).exponential(size=200000), 1.0, 1.0, 2.0),
    ],
    ids=["gaussian", "uniform", "exponential"],
)
@pytest.mark.parametrize(
    "with_weights", [False, True], ids=["without_weights", "with_weights"]
)
def test_bootstrapmoment_standard_distributions(
    with_weights: bool, generator: Callable[[], np.ndarray], mu, sigma, skewness
):
    moment = BootstrapMoment(numsamples=100, rng=1234)
    values = generator()
    if with_weights:
        weight = np.random.default_rng(91011).uniform(size=values.shape)
        moment.fill(values, weight=weight)
    else:
        moment.fill(values)
    tolerance = 0.01
    assert moment.mean().nominal == pytest.approx(mu, rel=tolerance, abs=tolerance)
    assert np.average(moment.mean().samples) == pytest.approx(
        mu, rel=tolerance, abs=tolerance
    )
    assert moment.std().nominal == pytest.approx(sigma, rel=tolerance, abs=tolerance)
    assert np.average(moment.std().samples) == pytest.approx(
        sigma, rel=tolerance, abs=tolerance
    )
    assert moment.skewness().nominal == pytest.approx(
        skewness, rel=tolerance, abs=tolerance
    )
    assert np.average(moment.skewness().samples) == pytest.approx(
        skewness, rel=tolerance, abs=tolerance
    )
