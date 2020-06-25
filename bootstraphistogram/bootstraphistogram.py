from copy import deepcopy
from ctypes import Array
from typing import Any, Union, List, Tuple, Optional, Generator

import boost_histogram as bh
import numpy as np


class BootstrapHistogram:
    def __init__(self, *axes: bh.axis.Axis, numbootstrapsamples: int = 1000,
                 rng: Union[int, np.random.Generator, None] = None, **kwargs: Any):
        axes = list(axes)
        axes.append(bh.axis.Integer(0, numbootstrapsamples))
        self._random = np.random.default_rng(rng)
        self._hist = bh.Histogram(*axes, **kwargs)

    @property
    def hist(self):
        return self._hist

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
            w = self._random.poisson(1.0, size=shape)
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

    def __eq__(self, other: "BootstrapHistogram") -> bool:
        return self._hist == other._hist

    def __add__(self, other: "BootstrapHistogram") -> "BootstrapHistogram":
        result = deepcopy(self)
        result._hist += other._hist
        return result

    def __mul__(self, other: float):
        result = deepcopy(self)
        result._hist *= other
        return result

    def __truediv__(self, other: float):
        result = deepcopy(self)
        result._hist /= other
        return result
