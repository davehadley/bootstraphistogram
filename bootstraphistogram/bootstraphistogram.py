from typing import Any

import boost_histogram as bh

class BootstrapHistogram:
    def __init__(self, *axes: bh.axis.Axis, numbootstrapsamples: int=1000, **kwargs: Any):
        self._hist = bh.Histogram(*axes, **kwargs)