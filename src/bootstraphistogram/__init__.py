"""bootstraphistogram

A multi-dimensional histogram. The distribution of the histograms bin values is
computed with the Possion bootstrap re-sampling method.

* :py:class:`bootstraphistogram.BootstrapHistogram` is the main histogram class.
* Some basic plotting functions are provided in :py:mod:`bootstraphistogram.plot`.
* :py:class:`bootstraphistogram.BootstrapMoment` calculates the first three moments of
  a dataset.

"""

from boost_histogram import axis

from bootstraphistogram import plot
from bootstraphistogram.bootstraphistogram import BootstrapHistogram
from bootstraphistogram.bootstrapmoment import BootstrapMoment
from bootstraphistogram.valuewithsamples import ValueWithSamples

__version__ = "0.8.0"
__license__ = "MIT"
__author__ = "David Hadley"
__url__ = "https://github.com/davehadley/bootstraphistogram"

__all__ = ["BootstrapHistogram", "BootstrapMoment", "ValueWithSamples", "axis", "plot"]
