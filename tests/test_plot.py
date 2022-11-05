import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from bootstraphistogram import BootstrapHistogram, axis, plot


def uniformhist(
    ax: axis.Axis = axis.Variable([0.0, 0.1, 0.3, 0.6, 1.0]),
    size: int = 100,
    numsamples: int = 10,
    low: float = 0.0,
    high: float = 1.0,
) -> BootstrapHistogram:
    hist = BootstrapHistogram(ax, numsamples=numsamples)
    hist.fill(np.random.uniform(low=0.0, high=1.0, size=size))
    return hist


def test_plot_errorbar() -> None:
    hist = uniformhist()
    plot.errorbar(hist)
    return


def test_plot_errorbar_with_percentiles() -> None:
    hist = uniformhist()
    plot.errorbar(hist, (10.0, 50.0, 90.0))
    plt.show()
    return


def test_plot_step() -> None:
    hist = uniformhist()
    plot.step(hist)
    plt.legend()
    return


def test_plot_fillbetween() -> None:
    hist = uniformhist()
    plot.fill_between(hist)
    return


def test_plot_fillbetween_with_std_and_mean() -> None:
    hist = uniformhist()
    plot.fill_between(hist, percentiles=None)
    return


def test_plot_scatter() -> None:
    hist = uniformhist()
    plot.scatter(hist)
    return


def test_plot_multi() -> None:
    hist = uniformhist()
    plot.fill_between(hist, color="blue", alpha=0.25)
    plot.step(hist, percentile=50.0)
    plot.errorbar(hist, color="black")
    plot.scatter(hist, color="red")
    return
