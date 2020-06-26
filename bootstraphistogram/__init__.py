"""bootstraphistogram

A multi-dimensional histogram. The distribution of the histograms bin values is computed with the Possion bootstrap
re-sampling method. The main class is implemented in :py:class:`bootstraphistogram.BootstrapHistogram`.
Some basic plotting functions are provided in :py:mod:`bootstraphistogram.plot`

"""

from bootstraphistogram import _version

__version__ = _version.__version__
__license__ = "MIT"
__author__ = "David Hadley"
url = "https://github.com/davehadley/bootstraphistogram"

from bootstraphistogram.bootstraphistogram import BootstrapHistogram
import boost_histogram.axis as axis
import bootstraphistogram.plot as plot

__all__ = ["BootstrapHistogram", "axis", "plot"]
