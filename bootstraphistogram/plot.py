from itertools import zip_longest
from typing import Optional, Any, Iterable, List, Dict

import numpy as np
from matplotlib.axes import Axes as MplAxes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bootstraphistogram import BootstrapHistogram

_PERCENTILES_MEDIAN_AND_1SIGMA = [68.27 / 2.0, 50.0, 100.0 - 68.27 / 2.0, ]
_PERCENTILES_MEDIAN_AND_2SIGMA = [95.45 / 2.0, 50.0, 100.0 - 95.45 / 2.0]
_PERCENTILES_MEDIAN_AND_3SIGMA = [99.73 / 2.0, 50.0, 100.0 - 99.73 / 2.0]


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
         ax: Optional[MplAxes] = None, pckwargs: Optional[Dict[float, Dict[Any, Any]]] = None) -> List[Line2D]:
    ax = _getaxes(ax)
    edges = hist.axes[0].edges
    percentiles = list(percentiles)
    if pckwargs is None:
        pckwargs = {q: {"label": f"{q:.1f}%"} for q in percentiles}
    result = []
    for q in percentiles:
        Y = hist.percentile(q)
        try:
            kwargs = pckwargs[q]
        except KeyError:
            kwargs = {}
        result.append(ax.step(edges, np.concatenate((Y, [Y[-1]])), where="post", **kwargs))
    return result


def fill_between(hist: BootstrapHistogram, ax: Optional[MplAxes] = None, **kwargs: Any):
    ax = _getaxes(ax)
    return ax.fill_between(x=x, y1=y1, y2=y2, **kwargs)
