import pickle
import unittest
from typing import Optional

import boost_histogram as bh
import numpy as np

from bootstraphistogram import BootstrapHistogram


def _standard_error_mean(size, sigma=1.0):
    return sigma / np.sqrt(size)


def _standard_error_std(size, sigma=1.0):
    return np.sqrt(sigma ** 2 / (2.0 * size))


class TestBootstrapHistogram1D(unittest.TestCase):

    def assertArrayEqual(self, actual: np.ndarray, expected: np.ndarray, msg: Optional[str] = None) -> None:
        return self.assertTrue(np.array_equal(actual, expected), msg=msg)

    def assertArrayAlmostEqual(self, actual: np.ndarray, expected: np.ndarray, delta: float,
                               msg: Optional[str] = None) -> None:
        return self.assertTrue(np.all(np.abs(actual - expected) < delta), msg=msg)

    def test_contructor(self):
        # check constructor works without raising error
        BootstrapHistogram(bh.axis.Regular(100, -1.0, 1.0))
        return

    def test_fill(self):
        hist = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), numsamples=10)
        size = 100000
        data = np.random.normal(loc=0.0, scale=1.0, size=size)
        hist.fill(data)
        x = hist.axes[0].centers
        y = hist.view()[:, np.random.randint(0, hist.numsamples)]
        mean = np.average(x, weights=y)
        std = np.average((x - mean) ** 2, weights=y)
        binwidth = hist.axes[0].edges[1] - hist.axes[0].edges[0]
        self.assertAlmostEqual(mean, 0.0, delta=5.0 * _standard_error_mean(size=size) + binwidth)
        self.assertAlmostEqual(std, 1.0, delta=5.0 * _standard_error_std(size=size) + binwidth)
        return

    def test_samples(self):
        numsamples = 100
        hist = BootstrapHistogram(bh.axis.Regular(100, 0.0, 1.0), numsamples=numsamples)
        size = 100000
        data = np.random.uniform(size=size)
        hist.fill(data)
        y = hist.view()
        mean = np.average(y, axis=1)
        std = np.std(y, axis=1)
        nbins = len(hist.axes[0])
        self.assertArrayAlmostEqual(mean, size / nbins, delta=5.0 * np.sqrt(size / nbins))
        self.assertArrayAlmostEqual(std, np.sqrt(size / nbins),
                                    delta=5.0 * _standard_error_std(size=numsamples,
                                                                    sigma=np.sqrt(size / nbins)))
        return

    def test_numsamples_property(self):
        numsamples = 100
        hist = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), numsamples=numsamples)
        self.assertEqual(hist.numsamples, numsamples)

    def test_axes_property(self):
        axes = (bh.axis.Regular(100, -5.0, 5.0),)
        hist = BootstrapHistogram(*axes)
        self.assertEqual(hist.axes[:-1], axes)

    def test_view_property(self):
        numsamples = 10
        nbins = 5
        hist = BootstrapHistogram(bh.axis.Regular(nbins, -5.0, 5.0), numsamples=numsamples)
        view = hist.view()
        self.assertArrayEqual(view, np.zeros(shape=(nbins, numsamples)))

    def test_equality(self):
        hist1 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), rng=123)
        hist2 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), rng=123)
        data = np.random.normal(size=1000)
        hist1.fill(data)
        hist2.fill(data)
        self.assertEqual(hist1, hist2)

    def test_inequality(self):
        hist1 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0))
        hist2 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0))
        data = np.random.normal(size=1000)
        hist1.fill(data)
        hist2.fill(data)
        self.assertNotEqual(hist1, hist2)

    def test_add(self):
        hist1 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0))
        hist2 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0))
        hist1.fill(np.random.normal(size=1000))
        hist2.fill(np.random.normal(size=1000))
        a1 = hist1.view()
        a2 = hist2.view()
        hist3 = hist1 + hist2
        self.assertArrayEqual(hist3.view(), a1 + a2)

    def test_multiply_by_scalar(self):
        hist1 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0))
        hist1.fill(np.random.normal(size=1000))
        scale = 2.0
        a1 = hist1.view() * scale
        hist3 = hist1 * scale
        self.assertArrayEqual(hist3.view(), a1)

    def test_divide_by_scalar(self):
        hist1 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0))
        hist1.fill(np.random.normal(size=1000))
        scale = 2.0
        a1 = hist1.view() / scale
        hist3 = hist1 / scale
        self.assertArrayEqual(hist3.view(), a1)

    def test_pickle(self):
        hist1 = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0))
        hist1.fill(np.random.normal(size=1000))
        hist2 = pickle.loads(pickle.dumps(hist1))
        self.assertEqual(hist1, hist2)

    def test_nominal(self):
        hist = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0))
        data = np.random.normal(size=1000)
        hist.fill(data)
        arr, _ = np.histogram(data, bins=hist.axes[0].edges)
        self.assertArrayEqual(hist.nominal.view(), arr)

    def test_mean(self):
        size = 100000
        hist = BootstrapHistogram(bh.axis.Regular(100, 0.0, 1.0), numsamples=100)
        data = np.random.uniform(size=size)
        hist.fill(data)
        nbins = len(hist.axes[0])
        self.assertArrayAlmostEqual(hist.mean(), size / nbins, delta=5.0 * np.sqrt(size / nbins))
        return


    def test_std(self):
        numsamples = 100
        hist = BootstrapHistogram(bh.axis.Regular(100, 0.0, 1.0), numsamples=numsamples)
        size = 100000
        data = np.random.uniform(size=size)
        hist.fill(data)
        nbins = len(hist.axes[0])
        self.assertArrayAlmostEqual(hist.std(), np.sqrt(size / nbins),
                                    delta=5.0 * _standard_error_std(size=numsamples,
                                                                    sigma=np.sqrt(size / nbins)))
        return