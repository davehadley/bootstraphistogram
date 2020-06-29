from copy import deepcopy, copy
from typing import Any, Union, Tuple, Optional

import boost_histogram as bh
import numpy as np


class BootstrapHistogram:
    """
    A histogram with automatic Poission bootstrap resampling

    The implementation is backed by boost :std:doc:`usage/histogram` (<https://github.com/scikit-hep/boost-histogram>) and
    thus :py:class:`BoostrapHistogram` mimics the :py:class:`boost_histogram.Histogram` interface.

    Parameters
    ----------
    *axes : boost_histogram.axis.Axis
        Any number of :py:class:`boost_histogram.axis.Axis` objects that define the histogram binning. See <https://boost-histogram.readthedocs.io/en/latest/usage/axes.html>.
    numsamples : int
        The number of bootstrap samples. Increasing this number improves the accuracy of estimators derived from the
        bootstrap samples, at the cost of increase memory and CPU usage.
    rng : Union[int, np.random.Generator, None]
        A numpy generator. If not provided, the numpy default from :py:func:`numpy.random.default_rng` will be used.
    **kwargs : Any
        Passed on to the :py:class:`boost_histogram.Histogram` constructor. :py:class:`numpy.ndarray`

    Attributes
    ----------
    nominal : boost_histogram.Histogram
        A histogram of the filled values, with no bootstrap samples.
    samples : boost_histogram.Histogram
        A histogram of the bootstrap samples. The last dimension corresponds to the bootstrap sample index and is
        of size :py:attr:`BootstrapHistogram.numsamples`.
    numsamples : int
        Number of bootstrap re-samplings.
    axes : Tuple[bh.axis.Axis, ...]
        :py:class:`boost_histogram.axis.Axis` representing the histogram binning. The last dimension corresponds to the
        bootstrap sample index.
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
        """
        Binned sample mean.

        Returns
        -------
        numpy.ndarray
            an array containing the mean value of all bootstrap samples for each bin in the histogram, .
        """
        return np.mean(self.view(flow=flow), axis=-1)

    def std(self, flow=False) -> np.ndarray:
        """
        Binned sample standard deviation.

        Returns
        -------
        numpy.ndarray
            an array containing the standard deviation of all bootstrap sample values for each bin in the histogram, .
        """
        return np.std(self.view(flow=flow), axis=-1)

    def percentile(self, q: float, flow=False, interpolation: str = "linear") -> np.ndarray:
        """
        Binned q-th percentile.

        Parameters
        ----------
        q : float
            The percentile, a number between 0 and 100 (inclusive).
        interpolation : str
            As :py:func:`numpy.percentile`.

        Returns
        -------
        numpy.ndarray
            an array containing the q-th percentile of all bootstrap samples for each bin in the histogram, .
        """
        return np.percentile(self.view(flow=flow), q, axis=-1, interpolation=interpolation)

    @property
    def numsamples(self) -> int:
        return len(self.axes[-1])

    @property
    def axes(self) -> Tuple[bh.axis.Axis, ...]:
        return self._hist.axes

    def fill(self, *args: np.ndarray,
             weight: Optional[np.ndarray] = None,
             seed: Optional[np.ndarray]=None,
             **kwargs: Any) -> "BootstrapHistogram":
        """
        Fill the histogram with some values.


        Parameters
        ----------
        *args : np.ndarray
            An 1D array containing coordinates for each dimension of the histogram.
        weight : Optional[np.ndarray]
            Entry weights used to fill the histogram.
        seed: Optional[np.ndarray]
            Per-element seed. Overrides the Generator given in the constructor and
            uses a pseudo-random number generator seeded by the given value.
            This may be useful when filling multiple histograms with data that is not statistically independent
            (where if may be desirable to seed the generator with a record ID).
        **kwargs : Any
            Passed on to :py:class:`boost_histogram.Histogram.fill`.

        Returns
        -------
        self : BootstrapHistogram
            Reference to this object. This is done to maintain consistency with `boost_histogram.Histogram`.
        """
        self._nominal.fill(*args, weight=weight, **kwargs)
        hist = self._hist
        shape = np.shape(args[0])
        if seed is not None:
            generators = np.array([np.random.Generator(np.random.PCG64(s)) for s in seed])
        for index in range(self.numsamples):
            if seed is None:
                w = self._random.poisson(1.0, size=shape)
            else:
                w = np.fromiter((r.poisson(1.0) for r in generators), dtype=np.float, count=len(generators))
            if weight is not None:
                w *= weight
            hist.fill(*args, index, weight=w, **kwargs)
        return self

    def view(self, flow=False) -> Any:
        """
        Return a view of the underlying histogram bootstrap sample values.
        """
        return self._hist.view(flow=flow)

    def __eq__(self, other: "BootstrapHistogram") -> bool:
        """
        BootstrapHistogram's are considered to be equal if the data in their underlying histograms are equal.
        """
        return isinstance(other, BootstrapHistogram) and self._hist == other._hist and self._nominal == other._nominal

    def __add__(self, other: "BootstrapHistogram") -> "BootstrapHistogram":
        """
        Add together the values of two bootstrap histograms together.

        This is useful to merge results when parallellizing filling in a map-reduce pattern.
        This histograms must have the same binning and the same number of bootstrap samples.
        The merge histogram will have the same binning and the same number of bootstrap samples as the input
        histograms.

        Returns
        -------
        hist : BootstrapHistogram
            A new instance of the summed histogram.
        """
        result = deepcopy(self)
        result._hist += other._hist
        result._nominal += other._nominal
        return result

    def __mul__(self, other: float):
        result = deepcopy(self)
        result._hist *= other
        return result

    def __truediv__(self, other: float):
        result = deepcopy(self)
        result._hist /= other
        return result

    def project(self, *args: int) -> "BootstrapHistogram":
        """
        Reduce histogram dimensionality but summing over some dimensions.
        The bootstrap sample axis is always kept by the operation.

        Parameters
        ----------
        *args: int
            The dimensions to be kept.

        Returns
        -------
        hist: BootstrapHistogram
            a copy of the histogram with only axes in *args and the bootstrap sample axes.
        """
        result = copy(self)
        args = list(args)
        result._nominal = result._nominal.project(*args)
        sampleaxis = len(self.axes) - 1
        if not sampleaxis in args:
            args.append(sampleaxis)
        result._hist = result._hist.project(*args)
        return result