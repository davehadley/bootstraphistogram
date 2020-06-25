"""bootstraphistogram

TODO
"""

__version__ = "0.1"
__license__ = "MIT"
__author__ = "David Hadley"
url = "https://github.com/davehadley/bootstraphistogram"

from bootstraphistogram.bootstraphistogram import BootstrapHistogram
import boost_histogram.axis as axis
import bootstraphistogram.plot as plot

__all__ = ["BootstrapHistogram", "axis", "plot"]

