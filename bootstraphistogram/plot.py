from typing import Optional

import matplotlib
import matplotlib.pyplot as plt

from bootstraphistogram import BootstrapHistogram


def errorbar(hist: BootstrapHistogram, ax: Optional[matplotlib.axes.Axes]=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    edges = hist.axes[0].edges
    x = hist.axes[0].centers
    xerr = [x-edges[:-1], edges[1:]-x]
    y = hist.mean()
    yerr = hist.std()
    return ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, **kwargs)
