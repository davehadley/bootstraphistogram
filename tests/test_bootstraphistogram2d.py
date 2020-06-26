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


class TestBootstrapHistogram2D(unittest.TestCase):
    def assertArrayEqual(self, actual: np.ndarray, expected: np.ndarray, msg: Optional[str] = None) -> None:
        return self.assertTrue(np.array_equal(actual, expected), msg=msg)

    def assertArrayAlmostEqual(self, actual: np.ndarray, expected: np.ndarray, delta: float,
                               msg: Optional[str] = None) -> None:
        return self.assertTrue(np.all(np.abs(actual - expected) < delta), msg=msg)

    def test_contructor(self):
        # check constructor works without raising error
        BootstrapHistogram(bh.axis.Regular(100, -1.0, 1.0), bh.axis.Variable([0.0, 1.0, 3.0, 6.0]))
        return

    def test_fill(self):
        hist = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), bh.axis.Regular(100, -5.0, 5.0), numsamples=10)
        size = 100000
        x = np.random.normal(loc=0.0, scale=1.0, size=size)
        y = np.random.normal(loc=0.0, scale=1.0, size=size)
        hist.fill(x, y)
        y = hist.view()[:, :, np.random.randint(0, hist.numsamples)]
        binwidth = hist.axes[0].edges[1] - hist.axes[0].edges[0]
        mean = np.average(hist.axes[0].centers, weights=np.sum(y, axis=1))
        std = np.average((hist.axes[0].centers - mean) ** 2, weights=np.sum(y, axis=1))
        self.assertAlmostEqual(mean, 0.0, delta=5.0 * _standard_error_mean(size=size) + binwidth)
        self.assertAlmostEqual(std, 1.0, delta=5.0 * _standard_error_std(size=size) + binwidth)
        mean = np.average(hist.axes[1].centers, weights=np.sum(y, axis=0))
        std = np.average((hist.axes[1].centers - mean) ** 2, weights=np.sum(y, axis=0))
        self.assertAlmostEqual(mean, 0.0, delta=5.0 * _standard_error_mean(size=size) + binwidth)
        self.assertAlmostEqual(std, 1.0, delta=5.0 * _standard_error_std(size=size) + binwidth)
        return

    def test_samples(self):
        numsamples = 100
        hist = BootstrapHistogram(bh.axis.Regular(10, 0.0, 1.0), bh.axis.Regular(10, 0.0, 1.0), numsamples=numsamples)
        size = 100000
        xdata = np.random.uniform(size=size)
        ydata = np.random.uniform(size=size)
        hist.fill(xdata, ydata)
        XY = hist.view()
        mean = np.average(XY, axis=2)
        std = np.std(XY, axis=2)
        nbins = len(hist.axes[0]) * len(hist.axes[1])
        self.assertArrayAlmostEqual(mean, size / nbins, delta=5.0 * np.sqrt(size / nbins))
        self.assertArrayAlmostEqual(std, np.sqrt(size / nbins),
                                    delta=5.0 * _standard_error_std(size=numsamples,
                                                                    sigma=np.sqrt(size / nbins)))
        return
