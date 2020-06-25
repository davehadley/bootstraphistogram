import unittest
import boost_histogram as bh
from bootstraphistogram import BootstrapHistogram


class TestBootstrapHistogram1D(unittest.TestCase):
    def test_contructor(self):
        # check constructor works without raising error
        BootstrapHistogram(bh.axis.Regular(100, -1.0, 1.0))
