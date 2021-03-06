import unittest

import matplotlib.pyplot as plt
import numpy as np

from bootstraphistogram import BootstrapHistogram, axis, plot


class TestPlotBootstrapHistogram1D(unittest.TestCase):
    def normalhist(
        self,
        ax=axis.Regular(100, -5.0, 5.0),
        size=100000,
        numsamples=10,
        mu=0.0,
        sigma=1.0,
    ):
        hist = BootstrapHistogram(ax, numsamples=numsamples)
        hist.fill(np.random.normal(loc=mu, scale=sigma, size=size))
        return hist

    def uniformhist(
        self,
        ax=axis.Variable([0.0, 0.1, 0.3, 0.6, 1.0]),
        size=100,
        numsamples=10,
        low=0.0,
        high=1.0,
    ):
        hist = BootstrapHistogram(ax, numsamples=numsamples)
        hist.fill(np.random.uniform(low=0.0, high=1.0, size=size))
        return hist

    def test_plot_errorbar(self):
        hist = self.uniformhist()
        plot.errorbar(hist)
        return

    def test_plot_step(self):
        hist = self.uniformhist()
        plot.step(hist)
        plt.legend()
        return

    def test_plot_fillbetween(self):
        hist = self.uniformhist()
        plot.fill_between(hist)
        return

    def test_plot_scatter(self):
        hist = self.uniformhist()
        plot.scatter(hist)
        return

    def test_plot_multi(self):
        hist = self.uniformhist()
        plot.fill_between(hist, color="blue", alpha=0.25)
        plot.step(hist, percentile=50.0)
        plot.errorbar(hist, color="black")
        plot.scatter(hist, color="red")
        return
