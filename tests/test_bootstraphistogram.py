import pickle

import boost_histogram as bh
import numpy as np

from bootstraphistogram import BootstrapHistogram


def _standard_error_mean(size, sigma=1.0):
    return sigma / np.sqrt(size)


def _standard_error_std(size, sigma=1.0):
    return np.sqrt(sigma ** 2 / (2.0 * size))


def array_almost_equal(
    actual: np.ndarray,
    expected: np.ndarray,
    delta: float,
) -> None:
    return np.all(np.abs(actual - expected) < delta)


def test_contructor():
    # check constructor works without raising error
    BootstrapHistogram(bh.axis.Regular(100, -1.0, 1.0), rng=1234)
    return


def test_fill():
    hist = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), numsamples=10, rng=1234)
    size = 100000
    data = np.random.normal(loc=0.0, scale=1.0, size=size)
    hist.fill(data)
    x = hist.axes[0].centers
    y = hist.view()[:, np.random.randint(0, hist.numsamples)]
    mean = np.average(x, weights=y)
    std = np.average((x - mean) ** 2, weights=y)
    binwidth = hist.axes[0].edges[1] - hist.axes[0].edges[0]
    assert array_almost_equal(
        mean, 0.0, delta=5.0 * _standard_error_mean(size=size) + binwidth
    )
    assert array_almost_equal(
        std, 1.0, delta=5.0 * _standard_error_std(size=size) + binwidth
    )
    return


def test_samples():
    numsamples = 100
    hist = BootstrapHistogram(
        bh.axis.Regular(100, 0.0, 1.0), numsamples=numsamples, rng=1234
    )
    size = 100000
    data = np.random.uniform(size=size)
    hist.fill(data)
    y = hist.view()
    mean = np.average(y, axis=1)
    std = np.std(y, axis=1)
    nbins = len(hist.axes[0])
    assert array_almost_equal(mean, size / nbins, delta=5.0 * np.sqrt(size / nbins))
    assert array_almost_equal(
        std,
        np.sqrt(size / nbins),
        delta=5.0 * _standard_error_std(size=numsamples, sigma=np.sqrt(size / nbins)),
    )
    return


def test_numsamples_property():
    numsamples = 100
    hist = BootstrapHistogram(
        bh.axis.Regular(100, -5.0, 5.0), numsamples=numsamples, rng=1234
    )
    assert hist.numsamples == numsamples


def test_axes_property():
    axes = (bh.axis.Regular(100, -5.0, 5.0),)
    hist = BootstrapHistogram(*axes, rng=1234)
    assert hist.axes[:-1] == axes


def test_view_property():
    numsamples = 10
    nbins = 5
    hist = BootstrapHistogram(
        bh.axis.Regular(nbins, -5.0, 5.0), numsamples=numsamples, rng=1234
    )
    view = hist.view()
    assert np.array_equal(view, np.zeros(shape=(nbins, numsamples)))


def test_equality():
    hist1 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), rng=123)
    hist2 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), rng=123)
    data = np.random.normal(size=1000)
    hist1.fill(data)
    hist2.fill(data)
    assert hist1 == hist2


def test_inequality():
    hist1 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0))
    hist2 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0))
    data = np.random.normal(size=1000)
    hist1.fill(data)
    hist2.fill(data)
    assert hist1 != hist2


def test_add():
    hist1 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), rng=1234)
    hist2 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), rng=1234)
    hist1.fill(np.random.normal(size=1000))
    hist2.fill(np.random.normal(size=1000))
    a1 = hist1.view()
    a2 = hist2.view()
    hist3 = hist1 + hist2
    assert np.array_equal(hist3.view(), a1 + a2)


def test_multiply_by_scalar_samples():
    hist1 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), rng=1234)
    hist1.fill(np.random.normal(size=1000))
    scale = 2.0
    a1 = hist1.view() * scale
    hist3 = hist1 * scale
    assert np.array_equal(hist3.view(), a1)


def test_divide_by_scalar_samples():
    hist1 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), rng=1234)
    hist1.fill(np.random.normal(size=1000))
    scale = 2.0
    a1 = hist1.view() / scale
    hist3 = hist1 / scale
    assert np.array_equal(hist3.view(), a1)


def test_multiply_by_scalar_nominal():
    hist = BootstrapHistogram(bh.axis.Regular(1, -1.0, 1.0), numsamples=10, rng=1234)
    hist.fill(0.0)
    scaled = hist * 2.0
    assert np.array_equal(list(hist.nominal.view()), [1.0])
    assert np.array_equal(list(scaled.nominal.view()), [2.0])


def test_divide_by_scalar_nominal():
    hist = BootstrapHistogram(bh.axis.Regular(1, -1.0, 1.0), numsamples=10, rng=1234)
    hist.fill(0.0)
    hist.fill(0.0)
    scaled = hist / 2.0
    assert np.array_equal(list(hist.nominal.view()), [2.0])
    assert np.array_equal(list(scaled.nominal.view()), [1.0])


def test_multiply_by_histogram():
    hist1 = BootstrapHistogram(bh.axis.Regular(2, 0.0, 2.0), rng=1234)
    hist2 = BootstrapHistogram(bh.axis.Regular(2, 0.0, 2.0), rng=1234)
    hist1.fill([0.0, 1.0, 0.0, 1.0])
    hist2.fill([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    hist3 = hist1 * hist2
    assert np.array_equal(hist3.nominal.view(), [6.0, 6.0])
    assert np.array_equal(hist3.samples, hist1.samples * hist2.samples)


def test_add_by_histogram():
    hist1 = BootstrapHistogram(bh.axis.Regular(2, 0.0, 2.0), rng=1234)
    hist2 = BootstrapHistogram(bh.axis.Regular(2, 0.0, 2.0), rng=1234)
    hist1.fill([0.0, 1.0, 0.0, 1.0])
    hist2.fill([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    hist3 = hist1 + hist2
    assert np.array_equal(hist3.nominal.view(), [5.0, 5.0])
    assert np.array_equal(hist3.samples, hist1.samples + hist2.samples)


def test_add_scalar():
    hist1 = BootstrapHistogram(bh.axis.Regular(2, 0.0, 2.0), rng=1234)
    hist1.fill([0.0, 0.0, 1.0])
    hist3 = hist1 + 2
    assert np.array_equal(hist3.nominal.view(), [4.0, 3.0])
    assert np.array_equal(hist3.samples, hist1.samples + 2)


def test_sub_by_histogram():
    hist1 = BootstrapHistogram(bh.axis.Regular(2, 0.0, 2.0), rng=1234)
    hist2 = BootstrapHistogram(bh.axis.Regular(2, 0.0, 2.0), rng=1234)
    hist1.fill([0.0, 1.0, 0.0, 1.0])
    hist2.fill([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    hist3 = hist1 - hist2
    assert np.array_equal(hist3.nominal.view(), [-1.0, -1.0])
    assert np.array_equal(hist3.samples, hist1.samples - hist2.samples)


def test_sub_scalar():
    hist1 = BootstrapHistogram(bh.axis.Regular(2, 0.0, 2.0), rng=1234)
    hist1.fill([0.0, 0.0, 1.0])
    hist3 = hist1 - 2
    assert np.array_equal(hist3.nominal.view(), [0.0, -1.0])
    assert np.array_equal(hist3.samples, hist1.samples - 2)


def test_pickle():
    hist1 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), rng=1234)
    hist1.fill(np.random.normal(size=1000))
    hist2 = pickle.loads(pickle.dumps(hist1))
    assert hist1 == hist2


def test_nominal():
    hist = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), rng=1234)
    data = np.random.normal(size=1000)
    hist.fill(data)
    arr, _ = np.histogram(data, bins=hist.axes[0].edges)
    assert np.array_equal(hist.nominal.view(), arr)


def test_mean():
    size = 100000
    hist = BootstrapHistogram(bh.axis.Regular(100, 0.0, 1.0), numsamples=100, rng=1234)
    data = np.random.uniform(size=size)
    hist.fill(data)
    nbins = len(hist.axes[0])
    assert array_almost_equal(
        hist.mean(), size / nbins, delta=5.0 * np.sqrt(size / nbins)
    )
    return


def test_std():
    numsamples = 100
    hist = BootstrapHistogram(
        bh.axis.Regular(100, 0.0, 1.0), numsamples=numsamples, rng=1234
    )
    size = 100000
    data = np.random.uniform(size=size)
    hist.fill(data)
    nbins = len(hist.axes[0])
    assert array_almost_equal(
        hist.std(),
        np.sqrt(size / nbins),
        delta=5.0 * _standard_error_std(size=numsamples, sigma=np.sqrt(size / nbins)),
    )


def test_fill_with_float_weights():
    hist = BootstrapHistogram(bh.axis.Regular(5, 0.0, 5.0), rng=1234)
    values = [0.0, 1.0, 2.0, 3.0, 3.0, 4.0]
    weights = [0.0, 1.0, 2.0, 3.0, 4.0, -1.0]
    hist.fill(values, weight=weights)
    assert np.array_equal(hist.nominal.view(), [0.0, 1.0, 2.0, 7.0, -1.0])


def test_fill_with_integer_weights():
    hist = BootstrapHistogram(bh.axis.Regular(5, 0.0, 5.0), rng=1234)
    values = [0.0, 1.0, 2.0, 3.0, 3.0, 4.0]
    weights = [0, 1, 2, 3, 4, -1]
    hist.fill(values, weight=weights)
    assert np.array_equal(hist.nominal.view(), [0.0, 1.0, 2.0, 7.0, -1.0])


def test_fill_with_record_id_seed():
    hist1 = BootstrapHistogram(bh.axis.Regular(100, 0.0, 1.0), numsamples=100, rng=1234)
    hist2 = BootstrapHistogram(bh.axis.Regular(100, 0.0, 1.0), numsamples=100, rng=5678)
    data = np.random.uniform(size=10000)
    seed = np.arange(len(data))
    hist1.fill(data, seed=seed)
    hist2.fill(data, seed=seed)
    assert hist1 == hist2
    return
