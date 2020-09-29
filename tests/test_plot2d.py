import unittest

import numpy as np

from bootstraphistogram import BootstrapHistogram, axis, plot
from bootstraphistogram.plot import HistogramRankError


class TestPlotBootstrapHistogram2D(unittest.TestCase):
    def normalhist(
        self,
        ax=(axis.Regular(100, -5.0, 5.0), axis.Regular(100, -5.0, 5.0)),
        size=100000,
        numsamples=10,
        mu=0.0,
        sigma=1.0,
    ):
        hist = BootstrapHistogram(*ax, numsamples=numsamples)
        hist.fill(
            np.random.normal(loc=mu, scale=sigma, size=size),
            np.random.normal(loc=mu, scale=sigma, size=size),
        )
        return hist

    def uniformhist(
        self,
        ax=(axis.Regular(10, 0.0, 1.0), axis.Regular(10, 0.0, 1.0)),
        size=100,
        numsamples=10,
        low=0.0,
        high=1.0,
    ):
        hist = BootstrapHistogram(*ax, numsamples=numsamples)
        hist.fill(
            np.random.uniform(low=0.0, high=1.0, size=size),
            np.random.uniform(low=0.0, high=1.0, size=size),
        )
        return hist

    def test_plot_1d_errorbar(self):
        hist = self.uniformhist()
        with self.assertRaises(HistogramRankError):
            plot.errorbar(hist)
        return

    def test_plot_1d_errorbar_projection(self):
        hist = self.uniformhist()
        plot.errorbar(hist.project(np.random.choice([0, 1])))
        return

    def test_plot_1d_scatter(self):
        hist = self.uniformhist()
        with self.assertRaises(HistogramRankError):
            plot.scatter(hist)
        return

    def test_plot_1d_scatter_projection(self):
        hist = self.uniformhist()
        plot.scatter(hist.project(np.random.choice([0, 1])))
        return

    def test_plot_1d_step(self):
        hist = self.uniformhist()
        with self.assertRaises(HistogramRankError):
            plot.step(hist)
        return

    def test_plot_1d_step_projection(self):
        hist = self.uniformhist()
        plot.step(hist.project(np.random.choice([0, 1])))
        return

    def test_plot_1d_fillbetween(self):
        hist = self.uniformhist()
        with self.assertRaises(HistogramRankError):
            plot.fill_between(hist)
        return

    def test_plot_1d_fillbetween_projection(self):
        hist = self.uniformhist()
        plot.fill_between(hist.project(np.random.choice([0, 1])))
        return
