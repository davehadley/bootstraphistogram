"""Functions to plot `BootstrapHistogram` objects with `matplotlib`."""
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib.axes import Axes as MplAxes  # type: ignore

from bootstraphistogram.bootstraphistogram import BootstrapHistogram

_PERCENTILES_1SIGMA = (50.0 - 68.27 / 2.0, 50.0 + 68.27 / 2.0)
_PERCENTILES_2SIGMA = (50.0 - 95.45 / 2.0, 50.0 - 95.45 / 2.0)

_PERCENTILES_MEDIAN_AND_1SIGMA = (_PERCENTILES_1SIGMA[0], 50.0, _PERCENTILES_1SIGMA[1])
_PERCENTILES_MEDIAN_AND_2SIGMA = (_PERCENTILES_2SIGMA[0], 50.0, _PERCENTILES_2SIGMA[1])


class HistogramRankError(ValueError):
    """Error raised when trying to plot a histogram with the wrong number of dimensions."""


def _enforce1d(hist: BootstrapHistogram) -> None:
    if len(hist.nominal.shape) != 1:
        raise HistogramRankError(
            "this function only supports plotting 1D histograms. Try "
            "BoostrapHistogram.project to reduce inputs to 1D."
        )


def _getaxes(ax: Optional[MplAxes]) -> MplAxes:
    if ax is None:
        ax = plt.gca()
    return ax


def errorbar(
    hist: BootstrapHistogram,
    percentiles: Optional[Tuple[float, float, float]] = None,
    ax: Optional[MplAxes] = None,
    **kwargs: Any,
) -> Any:
    """
    Plot the bootstrap sample mean and standard deviation.

    Parameters
    ----------
    hist: bootstraphistogram.BootstrapHistogram
        the :py:class:`bootstraphistogram.BootstrapHistogram` to plot.
    percentiles: Optional[Tuple[float, float]]
        lower, central, and upper percentiles to use for error bar.
        If None, the mean +-1 standard deviation is plotted.
    ax: Optional[matplotlib.axes.Axes]
        :py:class:`matplotlib.axes.Axes` to plot on.
    **kwargs : Any
        passed on to :py:meth:`matplotlib.axes.Axes.errorbar`

    Returns
    -------
    mplerrorbarresult : Any
        returns the result of call to :py:meth:`matplotlib.axes.Axes.errorbar`.
    """
    _enforce1d(hist)
    ax = _getaxes(ax)
    edges = hist.axes[0].edges
    x = hist.axes[0].centers
    xerr = [x - edges[:-1], edges[1:] - x]
    if percentiles is None:
        y = hist.mean()
        yerr: Any = hist.std()
    else:
        (errlow, centralpoint, errhigh) = percentiles
        y = hist.percentile(centralpoint)
        yerrlow = hist.percentile(errlow)
        yerrhigh = hist.percentile(errhigh)
        yerr = (y - yerrlow, yerrhigh - y)
    return ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, **kwargs)


def step(
    hist: BootstrapHistogram,
    percentile: Optional[float] = None,
    ax: Optional[MplAxes] = None,
    **kwargs: Any,
) -> Any:
    """
    Plot a curve corresponding to the histogram bootstrap sample mean (or the given percentile).

    Parameters
    ----------
    hist: bootstraphistogram.BootstrapHistogram
        the :py:class:`bootstraphistogram.BootstrapHistogram` to plot.
    percentile: Optional[float]
        the sample percentile to plot. A number between 0 and 100. 50 corresponds to
        the median. If ``None``, the mean is plotted.
    ax: Optional[matplotlib.axes.Axes]
        :py:class:`matplotlib.axes.Axes` to plot on.
    **kwargs : Any
        passed on to :py:meth:`matplotlib.axes.Axes.step`

    Returns
    -------
    mplstepresult : Any
        returns the result of call to :py:meth:`matplotlib.axes.Axes.step`.
    """
    _enforce1d(hist)
    ax = _getaxes(ax)
    edges = hist.axes[0].edges
    if percentile is not None:
        Y = hist.percentile(percentile)
    else:
        Y = hist.mean()
    return ax.step(
        edges, np.concatenate((Y, [Y[-1]])), where="post", **kwargs  # type: ignore
    )


def fill_between(
    hist: BootstrapHistogram,
    percentiles: Optional[Tuple[float, float]] = _PERCENTILES_1SIGMA,
    ax: Optional[MplAxes] = None,
    **kwargs: Any,
) -> Any:
    """
    Fill the area between two percentiles.

    Parameters
    ----------
    hist: bootstraphistogram.BootstrapHistogram
        the :py:class:`bootstraphistogram.BootstrapHistogram` to plot.
    percentiles: Optional[Tuple[float, float]]
        upper and lower percentile bounds to fill. A pair of numbers between 0 and 100.
        Defaults to fill an equal-tailed 68.27% interval.
        If None, the mean +-1 standard deviation is plotted.
    ax: Optional[matplotlib.axes.Axes]
        :py:class:`matplotlib.axes.Axes` to plot on.
    **kwargs : Any
        passed on to :py:meth:`matplotlib.axes.Axes.fill_between`

    Returns
    -------
    mplfillbetweenresult : Any
        returns the result of call to :py:meth:`matplotlib.axes.Axes.fill_between`.
    """
    _enforce1d(hist)
    ax = _getaxes(ax)
    X = hist.axes[0].edges
    if percentiles is None:
        mean = hist.mean()
        std = hist.std()
        Y1 = mean - std
        Y2 = mean - std
    else:
        low, high = min(percentiles), max(percentiles)
        Y1 = hist.percentile(low)
        Y2 = hist.percentile(high)
    Y1 = np.concatenate((Y1, [Y1[-1]]))
    Y2 = np.concatenate((Y2, [Y2[-1]]))
    return ax.fill_between(x=X, y1=Y1, y2=Y2, step="post", **kwargs)


def scatter(
    hist: BootstrapHistogram, ax: Optional[MplAxes] = None, **kwargs: Any
) -> Any:  #
    """
    Scatter plot of the bootstrap samples.

    The scatter-point x-coordinate within a bin drawn is from a uniform random distribution.

    Parameters
    ----------
    hist: bootstraphistogram.BootstrapHistogram
        the :py:class:`bootstraphistogram.BootstrapHistogram` to plot.
    ax: Optional[matplotlib.axes.Axes]
        :py:class:`matplotlib.axes.Axes` to plot on.
    **kwargs : Any
        passed on to :py:meth:`matplotlib.axes.Axes.fill_between`

    Returns
    -------
    mplfillbetweenresult : Any
        returns the result of call to :py:meth:`matplotlib.axes.Axes.scatter`.
    """
    _enforce1d(hist)
    ax = _getaxes(ax)
    X = []
    Y = []
    for binnum, binlow, binhigh in zip(
        range(len(hist.axes[0])), hist.axes[0].edges[:-1], hist.axes[0].edges[1:]
    ):
        y = hist.view()[binnum]
        x = np.random.uniform(low=binlow, high=binhigh, size=len(y))
        X.append(x)
        Y.append(y)
    return ax.scatter(X, Y, **kwargs)
