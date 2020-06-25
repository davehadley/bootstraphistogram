import unittest

import numpy as np
import matplotlib.pyplot as plt

from bootstraphistogram import BootstrapHistogram, axis, plot


class TestPlotBootstrapHistogram1D(unittest.TestCase):

    def normalhist(self, ax=axis.Regular(100, -5.0, 5.0), size=100000, numsamples=10, mu=0.0, sigma=1.0):
        hist = BootstrapHistogram(ax, numsamples=numsamples)
        hist.fill(np.random.normal(loc=mu, scale=sigma, size=size))
        return hist

    def uniformhist(self, ax=axis.Regular(100, 0.0, 1.0), size=100000, numsamples=10, low=0.0, high=1.0):
        hist = BootstrapHistogram(ax, numsamples=numsamples)
        hist.fill(np.random.uniform(low=0.0, high=1.0, size=size))
        return hist

    def test_plot_errorbar(self):
        hist = self.normalhist(ax=axis.Variable([-5.0, -2.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), size=100)
        plot.errorbar(hist)
        plt.show()
        return

    def test_plot_step(self):
        hist = self.uniformhist(ax=axis.Variable([0.0, 0.1, 0.3, 0.6, 1.0]), size=100)
        plot.step(hist)
        plt.legend()
        plt.show()