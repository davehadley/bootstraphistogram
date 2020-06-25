from ctypes import Array
from typing import Any, Union, List, Tuple, Optional

import boost_histogram as bh
import numpy as np


class BootstrapHistogram:
    def __init__(self, *axes: bh.axis.Axis, numbootstrapsamples: int = 1000, **kwargs: Any):
        axes = list(axes)
        axes.append(bh.axis.Integer(0, numbootstrapsamples))
        self._hist = bh.Histogram(*axes, **kwargs)

    @property
    def numbootstrapsamples(self) -> int:
        return len(self.axes[-1])

    @property
    def axes(self) -> Tuple[bh.axis.Axis, ...]:
        return self._hist.axes

    def fill(self, *args: np.ndarray,
             weight: Optional[np.ndarray] = None,
             **kwargs: Any) -> "BootstrapHistogram":
        hist = self._hist
        shape = args[0].shape
        for index in range(self.numbootstrapsamples):
            w = np.random.poisson(1.0, size=shape)
            if weight is not None:
                w *= weight
            hist.fill(*args, index, weight=w, **kwargs)
        # turns out this is slower...
        # shape = args[0].shape + (self.numbootstrapsamples,)
        # ax = [np.tile(a, shape[-1]) for a in args]
        # ax.append(np.repeat(self.axes[-1].centers, shape[:-1]))
        # w = np.random.poisson(1.0, size=shape)
        # w = w if weight is None else w*weight
        # self._hist.fill(*ax, weight=w, **kwargs)
        return self

    def view(self, flow=False) -> Any:
        return self._hist.view(flow=flow)
