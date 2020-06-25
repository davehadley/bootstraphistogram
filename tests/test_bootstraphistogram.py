import unittest
import boost_histogram as bh
import numpy as np

from bootstraphistogram import BootstrapHistogram


def _standard_error_mean(size, sigma=1.0):
    return sigma / np.sqrt(size)

def _standard_error_std(size, sigma=1.0):
    return np.sqrt(sigma**2/(2.0*size))

class TestBootstrapHistogram1D(unittest.TestCase):
    def test_contructor(self):
        # check constructor works without raising error
        BootstrapHistogram(bh.axis.Regular(100, -1.0, 1.0))

    def test_fill(self):
        hist = BootstrapHistogram(bh.axis.Regular(100, -5.0, 5.0))
        size = 100000
        data = np.random.normal(loc=0.0, scale=1.0,  size=size)
        hist.fill(data)
        x = hist.axes[0].centers
        y = hist.view()[:,0]
        mean = np.average(x, weights=y)
        std = np.average((x - mean)**2, weights=y)
        self.assertAlmostEqual(mean, 0.0, delta=5.0*_standard_error_mean(size=size))
        self.assertAlmostEqual(std, 1.0, delta=5.0*_standard_error_std(size=size))
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
