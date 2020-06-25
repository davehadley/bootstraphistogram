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

    def assertArrayAlmostEqual(self, actual: np.ndarray, expected: np.ndarray, delta: float, msg: Optional[str] = None):
        return self.assertTrue(np.all(np.abs(actual - expected) < delta), msg=msg)

    def test_contructor(self):
        # check constructor works without raising error
        BootstrapHistogram(bh.axis.Regular(100, -1.0, 1.0))
        return

    def test_fill(self):
        hist = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), numbootstrapsamples=10)
        size = 100000
        data = np.random.normal(loc=0.0, scale=1.0, size=size)
        hist.fill(data)
        x = hist.axes[0].centers
        y = hist.view()[:, np.random.randint(0, hist.numbootstrapsamples)]
        mean = np.average(x, weights=y)
        std = np.average((x - mean) ** 2, weights=y)
        binwidth = hist.axes[0].edges[1] - hist.axes[0].edges[0]
        self.assertAlmostEqual(mean, 0.0, delta=5.0 * _standard_error_mean(size=size) + binwidth)
        self.assertAlmostEqual(std, 1.0, delta=5.0 * _standard_error_std(size=size) + binwidth)
        return

    def test_samples(self):
        numbootstrapsamples = 100
        hist = BootstrapHistogram(bh.axis.Regular(100, 0.0, 1.0), numbootstrapsamples=numbootstrapsamples)
        size = 100000
        data = np.random.uniform(size=size)
        hist.fill(data)
        y = hist.view()
        mean = np.average(y, axis=1)
        std = np.std(y, axis=1)
        nbins = len(hist.axes[0])
        self.assertArrayAlmostEqual(mean, size / nbins, delta=5.0 * np.sqrt(size/nbins))
        self.assertArrayAlmostEqual(std, np.sqrt(size / nbins), delta=5.0 * _standard_error_std(size=numbootstrapsamples, sigma=np.sqrt(size/nbins)))
        return

    def test_numbootstrapsamples_property(self):
        numbootstrapsamples = 100
        hist = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), numbootstrapsamples=numbootstrapsamples)
        self.assertEqual(hist.numbootstrapsamples, numbootstrapsamples)

    def test_axes_property(self):
        axes = (bh.axis.Regular(100, -5.0, 5.0),)
        hist = BootstrapHistogram(*axes)
        self.assertEqual(hist.axes[:-1], axes)

    def test_view_property(self):
        numbootstrapsamples = 100
        hist = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0), numbootstrapsamples=numbootstrapsamples)
        view = hist.view()
        self.assertEqual(view)
