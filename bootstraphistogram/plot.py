from collections import defaultdict
from itertools import zip_longest
from typing import Optional, Any, Iterable, List, Dict, Tuple

import numpy as np
from matplotlib.axes import Axes as MplAxes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bootstraphistogram import BootstrapHistogram

_PERCENTILES_1SIGMA = (68.27 / 2.0, 100.0 - 68.27 / 2.0)
_PERCENTILES_2SIGMA = (95.45 / 2.0, 100.0 - 95.45 / 2.0)
_PERCENTILES_3SIGMA = (99.73 / 2.0, 100.0 - 99.73 / 2.0)

_PERCENTILES_MEDIAN_AND_1SIGMA = (68.27 / 2.0, 50.0, 100.0 - 68.27 / 2.0)
_PERCENTILES_MEDIAN_AND_2SIGMA = (95.45 / 2.0, 50.0, 100.0 - 95.45 / 2.0)
_PERCENTILES_MEDIAN_AND_3SIGMA = (99.73 / 2.0, 50.0, 100.0 - 99.73 / 2.0)


def _getaxes(ax: Optional[MplAxes]):
    if ax is None:
        ax = plt.gca()
    return ax


def errorbar(hist: BootstrapHistogram, ax: Optional[MplAxes] = None, **kwargs: Any):
    ax = _getaxes(ax)
    edges = hist.axes[0].edges
    x = hist.axes[0].centers
    xerr = [x - edges[:-1], edges[1:] - x]
    y = hist.mean()
    yerr = hist.std()
    return ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, **kwargs)


def step(hist: BootstrapHistogram, percentiles: Iterable[float] = _PERCENTILES_MEDIAN_AND_1SIGMA,
         ax: Optional[MplAxes] = None, pckwargs: Optional[Dict[float, Dict[Any, Any]]] = None,
         autolabel: bool = False) -> List[Line2D]:
    ax = _getaxes(ax)
    edges = hist.axes[0].edges
    percentiles = list(percentiles)
    if pckwargs is None:
        pckwargs = defaultdict(dict)
    if autolabel:
        for q in percentiles:
            pckwargs[q]["label"] = f"{q:.1f}%"
    result = []
    for q in percentiles:
        Y = hist.percentile(q)
        try:
            kwargs = pckwargs[q]
        except KeyError:
            kwargs = {}
        result.append(ax.step(edges, np.concatenate((Y, [Y[-1]])), where="post", **kwargs))
    return result


def fill_between(hist: BootstrapHistogram, percentiles: Tuple[float, float] = _PERCENTILES_1SIGMA,
                 ax: Optional[MplAxes] = None, **kwargs: Any):
    ax = _getaxes(ax)
    low, high = min(percentiles), max(percentiles)
    X = hist.axes[0].edges
    Y1 = hist.percentile(low)
    Y2 = hist.percentile(high)
    Y1 = np.concatenate((Y1, [Y1[-1]]))
    Y2 = np.concatenate((Y2, [Y2[-1]]))
    return ax.fill_between(x=X, y1=Y1, y2=Y2, step="post", **kwargs)


def scatter(hist: BootstrapHistogram, ax: Optional[MplAxes] = None, **kwargs: Any):
    ax = _getaxes(ax)
    X = []
    Y = []
    for binnum, binlow, binhigh in zip(range(len(hist.axes[0])), hist.axes[0].edges[:-1], hist.axes[0].edges[1:]):
        y = hist.view()[binnum]
        x = np.random.uniform(low=binlow, high=binhigh, size=len(y))
        X.append(x)
        Y.append(y)
    return ax.scatter(X, Y, **kwargs)