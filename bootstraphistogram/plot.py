from typing import Optional, Any, List, Tuple

import numpy as np
from matplotlib.axes import Axes as MplAxes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bootstraphistogram import BootstrapHistogram

_PERCENTILES_1SIGMA = (50.0 - 68.27 / 2.0, 50.0 + 68.27 / 2.0)
_PERCENTILES_2SIGMA = (50.0 - 95.45 / 2.0, 50.0 - 95.45 / 2.0)

_PERCENTILES_MEDIAN_AND_1SIGMA = (_PERCENTILES_1SIGMA[0], 50.0, _PERCENTILES_1SIGMA[1])
_PERCENTILES_MEDIAN_AND_2SIGMA = (_PERCENTILES_2SIGMA[0], 50.0, _PERCENTILES_2SIGMA[1])

class HistogramRankError(ValueError):
    pass

def _enforce1d(hist: BootstrapHistogram) -> None:
    if hist.nominal.rank != 1:
        raise HistogramRankError("this function only supports plotting 1D histograms. Try BoostrapHistogram.project to reduce inputs to 1D.")


def _getaxes(ax: Optional[MplAxes]):
    if ax is None:
        ax = plt.gca()
    return ax


def errorbar(hist: BootstrapHistogram, ax: Optional[MplAxes] = None, **kwargs: Any) -> Any:
    """
    Plot the bootstrap sample mean and standard deviation.

    Parameters
    ----------
    hist: bootstraphistogram.BootstrapHistogram
        the :py:class:`bootstraphistogram.BootstrapHistogram` to plot.
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
    y = hist.mean()
    yerr = hist.std()
    return ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, **kwargs)


def step(hist: BootstrapHistogram, percentile: Optional[float] = None, ax: Optional[MplAxes] = None,
         **kwargs: Any) -> Any:
    """
    Plot a curve corresponding to the histogram bootstrap sample mean (or the given percentile).

    Parameters
    ----------
    hist: bootstraphistogram.BootstrapHistogram
        the :py:class:`bootstraphistogram.BootstrapHistogram` to plot.
    percentile: Optional[float]
        the sample percentile to plot. A number between 0 and 100. 50 corresponds to the median. If ``None``, the mean is plotted.
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
    return ax.step(edges, np.concatenate((Y, [Y[-1]])), where="post", **kwargs)


def fill_between(hist: BootstrapHistogram, percentiles: Tuple[float, float] = _PERCENTILES_1SIGMA,
                 ax: Optional[MplAxes] = None, **kwargs: Any) -> Any:
    """
    Fill the area between two percentiles.

    Parameters
    ----------
    hist: bootstraphistogram.BootstrapHistogram
        the :py:class:`bootstraphistogram.BootstrapHistogram` to plot.
    percentiles: Tuple[float, float]
        upper and lower percentile bounds to fill. A pair of numbers between 0 and 100. Defaults to fill an equal-tailed 68.27% interval.
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
    low, high = min(percentiles), max(percentiles)
    X = hist.axes[0].edges
    Y1 = hist.percentile(low)
    Y2 = hist.percentile(high)
    Y1 = np.concatenate((Y1, [Y1[-1]]))
    Y2 = np.concatenate((Y2, [Y2[-1]]))
    return ax.fill_between(x=X, y1=Y1, y2=Y2, step="post", **kwargs)


def scatter(hist: BootstrapHistogram, ax: Optional[MplAxes] = None, **kwargs: Any) -> Any:  #
    """
    Scatter plot of the bootstrap samples.

    The scatter-point x-coordinate within a bin drawn from a uniform random distribution.

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
    for binnum, binlow, binhigh in zip(range(len(hist.axes[0])), hist.axes[0].edges[:-1], hist.axes[0].edges[1:]):
        y = hist.view()[binnum]
        x = np.random.uniform(low=binlow, high=binhigh, size=len(y))
        X.append(x)
        Y.append(y)
    return ax.scatter(X, Y, **kwargs)
