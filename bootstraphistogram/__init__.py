"""bootstraphistogram

TODO
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

