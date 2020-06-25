import unittest

import numpy as np
import matplotlib.pyplot as plt

from bootstraphistogram import BootstrapHistogram, axis, plot


class TestPlotBootstrapHistogram1D(unittest.TestCase):

    def normalhist(self, size=100000, nbins=100, numsamples=10, mu=0.0, sigma=1.0, low=-5.0, high=5.0):
        hist = BootstrapHistogram(axis.Regular(nbins, low, high), numsamples=numsamples)
        hist.fill(np.random.normal(loc=mu, scale=sigma, size=size))
        return hist

    def test_plot_errorbar(self):
        hist = self.normalhist(nbins=10, size=100)
        plot.errorbar(hist)
        plt.show()
        return