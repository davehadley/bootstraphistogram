from typing import Tuple

import numpy as np  # type: ignore
import pytest

from bootstraphistogram import BootstrapHistogram, axis, plot
from bootstraphistogram.plot import HistogramRankError


def uniformhist(
    ax: Tuple[axis.Axis, axis.Axis] = (
        axis.Regular(10, 0.0, 1.0),
        axis.Regular(10, 0.0, 1.0),
    ),
    size: int = 100,
    numsamples: int = 10,
    low: float = 0.0,
    high: float = 1.0,
) -> BootstrapHistogram:
    hist = BootstrapHistogram(*ax, numsamples=numsamples)
    hist.fill(
        np.random.uniform(low=0.0, high=1.0, size=size),
        np.random.uniform(low=0.0, high=1.0, size=size),
    )
    return hist


def test_plot_1d_errorbar() -> None:
    hist = uniformhist()
    with pytest.raises(HistogramRankError):
        plot.errorbar(hist)
    return


def test_plot_1d_errorbar_projection() -> None:
    hist = uniformhist()
    plot.errorbar(hist.project(np.random.choice([0, 1])))
    return


def test_plot_1d_scatter() -> None:
    hist = uniformhist()
    with pytest.raises(HistogramRankError):
        plot.scatter(hist)
    return


def test_plot_1d_scatter_projection() -> None:
    hist = uniformhist()
    plot.scatter(hist.project(np.random.choice([0, 1])))
    return


def test_plot_1d_step() -> None:
    hist = uniformhist()
    with pytest.raises(HistogramRankError):
        plot.step(hist)
    return


def test_plot_1d_step_projection() -> None:
    hist = uniformhist()
    plot.step(hist.project(np.random.choice([0, 1])))
    return


def test_plot_1d_fillbetween() -> None:
    hist = uniformhist()
    with pytest.raises(HistogramRankError):
        plot.fill_between(hist)
    return


def test_plot_1d_fillbetween_projection() -> None:
    hist = uniformhist()
    plot.fill_between(hist.project(np.random.choice([0, 1])))
    return
