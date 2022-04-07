from typing import Optional

import boost_histogram as bh
import numpy as np  # type: ignore

from bootstraphistogram import BootstrapHistogram


def _standard_error_mean(size: int, sigma: float = 1.0) -> float:
    return float(sigma / np.sqrt(size))


def _standard_error_std(size: int, sigma: float = 1.0) -> float:
    return float(np.sqrt(sigma**2 / (2.0 * size)))


def array_almost_equal(
    actual: np.ndarray,
    expected: np.ndarray,
    delta: float,
    msg: Optional[str] = None,
) -> bool:
    return bool(np.all(np.abs(actual - expected) < delta))


def test_constructor() -> None:
    # check constructor works without raising error
    BootstrapHistogram(
        bh.axis.Regular(100, -1.0, 1.0), bh.axis.Variable([0.0, 1.0, 3.0, 6.0])
    )
    return


def test_fill() -> None:
    hist = BootstrapHistogram(
        bh.axis.Regular(100, -5.0, 5.0),
        bh.axis.Regular(100, -5.0, 5.0),
        numsamples=10,
        rng=1234,
    )
    size = 100000
    x = np.random.normal(loc=0.0, scale=1.0, size=size)
    y = np.random.normal(loc=0.0, scale=1.0, size=size)
    hist.fill(x, y)
    y = hist.view()[:, :, np.random.randint(0, hist.numsamples)]
    binwidth = hist.axes[0].edges[1] - hist.axes[0].edges[0]
    mean = np.average(hist.axes[0].centers, weights=np.sum(y, axis=1))
    std = np.average((hist.axes[0].centers - mean) ** 2, weights=np.sum(y, axis=1))
    assert array_almost_equal(
        mean, 0.0, delta=5.0 * _standard_error_mean(size=size) + binwidth
    )
    assert array_almost_equal(
        std, 1.0, delta=5.0 * _standard_error_std(size=size) + binwidth
    )
    mean = np.average(hist.axes[1].centers, weights=np.sum(y, axis=0))
    std = np.average((hist.axes[1].centers - mean) ** 2, weights=np.sum(y, axis=0))
    assert array_almost_equal(
        mean, 0.0, delta=5.0 * _standard_error_mean(size=size) + binwidth
    )
    assert array_almost_equal(
        std, 1.0, delta=5.0 * _standard_error_std(size=size) + binwidth
    )
    return


def test_samples() -> None:
    numsamples = 100
    hist = BootstrapHistogram(
        bh.axis.Regular(10, 0.0, 1.0),
        bh.axis.Regular(10, 0.0, 1.0),
        numsamples=numsamples,
        rng=1234,
    )
    size = 100000
    xdata = np.random.uniform(size=size)
    ydata = np.random.uniform(size=size)
    hist.fill(xdata, ydata)
    XY = hist.view()
    mean = np.average(XY, axis=2)
    std = np.std(XY, axis=2)
    nbins = len(hist.axes[0]) * len(hist.axes[1])
    assert array_almost_equal(mean, size / nbins, delta=5.0 * np.sqrt(size / nbins))
    assert array_almost_equal(
        std,
        np.sqrt(size / nbins),
        delta=5.0 * _standard_error_std(size=numsamples, sigma=np.sqrt(size / nbins)),
    )
    return


def test_projection() -> None:
    numsamples = 100
    hist = BootstrapHistogram(
        bh.axis.Regular(10, 0.0, 1.0),
        bh.axis.Regular(10, 0.0, 1.0),
        numsamples=numsamples,
        rng=1234,
    )
    size = 100000
    xdata = np.random.uniform(size=size)
    ydata = np.random.uniform(size=size)
    hist.fill(xdata, ydata)
    hist = hist.project(0)
    X = hist.view()
    mean = np.average(X, axis=1)
    std = np.std(X, axis=1)
    nbins = len(hist.axes[0])
    assert array_almost_equal(mean, size / nbins, delta=5.0 * np.sqrt(size / nbins))
    assert array_almost_equal(
        std,
        np.sqrt(size / nbins),
        delta=5.0 * _standard_error_std(size=numsamples, sigma=np.sqrt(size / nbins)),
    )


def test_projection2() -> None:
    hist = BootstrapHistogram(
        bh.axis.Regular(100, -5.0, 5.0),
        bh.axis.Regular(100, -5.0, 5.0),
        numsamples=10,
        rng=1234,
    )
    size = 100000
    x = np.random.normal(loc=0.0, scale=1.0, size=size)
    y = np.random.normal(loc=0.0, scale=1.0, size=size)
    hist.fill(x, y)
    hist = hist.project(0)
    y = hist.view()[:, np.random.randint(0, hist.numsamples)]
    binwidth = hist.axes[0].edges[1] - hist.axes[0].edges[0]
    mean = np.average(hist.axes[0].centers, weights=y)
    std = np.average((hist.axes[0].centers - mean) ** 2, weights=y)
    assert array_almost_equal(
        mean, 0.0, delta=5.0 * _standard_error_mean(size=size) + binwidth
    )
    assert array_almost_equal(
        std, 1.0, delta=5.0 * _standard_error_std(size=size) + binwidth
    )
    return


def test_projection3() -> None:
    hist = BootstrapHistogram(
        bh.axis.Regular(3, 0.0, 3.0),
        bh.axis.Regular(2, 0.0, 2.0),
        numsamples=1000,
        rng=1234,
    )
    X = [0.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    Y = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    hist.fill(X, Y)
    hx = hist.project(0)
    hy = hist.project(1)
    assert np.array_equal(hx.nominal.view(), np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(hy.nominal.view(), np.array([4.0, 2.0]))
    delta = 0.1
    assert array_almost_equal(
        np.average(hx.samples.view(), axis=1),
        np.array([1.0, 2.0, 3.0]),
        delta=delta,
    )
    assert array_almost_equal(
        np.average(hy.samples.view(), axis=1), np.array([4.0, 2.0]), delta=delta
    )
