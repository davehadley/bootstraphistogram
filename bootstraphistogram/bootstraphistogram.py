from copy import deepcopy
from typing import Any, Union, Tuple, Optional

import boost_histogram as bh
import numpy as np


class BootstrapHistogram:
    """This is the form of a docstring.

    It can be spread over several lines.

    """
    def __init__(self, *axes: bh.axis.Axis, numsamples: int = 1000,
                 rng: Union[int, np.random.Generator, None] = None, **kwargs: Any):
        axes = list(axes)
        self._nominal = bh.Histogram(*axes, **kwargs)
        axes.append(bh.axis.Integer(0, numsamples))
        self._random = np.random.default_rng(rng)
        self._hist = bh.Histogram(*axes, **kwargs)

    @property
    def nominal(self) -> bh.Histogram:
        return self._nominal

    @property
    def samples(self) -> bh.Histogram:
        return self._hist

    def mean(self, flow=False) -> np.ndarray:
        return np.mean(self.view(flow=flow), axis=-1)

    def std(self, flow=False) -> np.ndarray:
        return np.std(self.view(flow=flow), axis=-1)

    def percentile(self, q: float, flow=False, interpolation: str = "linear") -> np.ndarray:
        return np.percentile(self.view(flow=flow), q, axis=-1, interpolation=interpolation)

    @property
    def numsamples(self) -> int:
        return len(self.axes[-1])

    @property
    def axes(self) -> Tuple[bh.axis.Axis, ...]:
        return self._hist.axes

    def fill(self, *args: np.ndarray,
             weight: Optional[np.ndarray] = None,
             **kwargs: Any) -> "BootstrapHistogram":
        self._nominal.fill(*args, weight=weight, **kwargs)
        hist = self._hist
        shape = args[0].shape
        for index in range(self.numsamples):
            w = self._random.poisson(1.0, size=shape)
            if weight is not None:
                w *= weight
            hist.fill(*args, index, weight=w, **kwargs)
        return self

    def view(self, flow=False) -> Any:
        return self._hist.view(flow=flow)

    def __eq__(self, other: "BootstrapHistogram") -> bool:
        return isinstance(other, BootstrapHistogram) and self._hist == other._hist and self._nominal == other._nominal

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
