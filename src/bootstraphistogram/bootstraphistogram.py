"""Implements the main class of this package: :py:class:`BoostrapHistogram`."""

from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import boost_histogram as bh
import numpy as np

if TYPE_CHECKING:
    try:
        from numpy.typing import ArrayLike, NDArray
    except (ImportError, ModuleNotFoundError):
        pass


class BootstrapHistogram:
    """
    A histogram with automatic Poission bootstrap resampling

    The implementation is backed by boost :doc:`boost_histogram:user-guide/histogram`
    (<https://github.com/scikit-hep/boost-histogram>) and thus
    :py:class:`BoostrapHistogram` mimics the :py:class:`boost_histogram.Histogram`
    interface.

    Parameters
    ----------
    *axes : boost_histogram.axis.Axis
        Any number of :py:class:`boost_histogram.axis.Axis` objects that define the
        histogram binning. See
        <https://boost-histogram.readthedocs.io/en/latest/usage/axes.html>.
    numsamples : int
        The number of bootstrap samples. Increasing this number improves the accuracy
        of estimators derived from the bootstrap samples, at the cost of increased
        memory and CPU usage.
    rng : Union[int, np.random.Generator, None]
        A numpy generator. If not provided, the numpy default from
        :py:func:`numpy.random.default_rng` will be used.
    **kwargs : Any
        Passed on to the :py:class:`boost_histogram.Histogram` constructor.
        :py:class:`numpy.ndarray`
    """

    def __init__(
        self,
        *axes: bh.axis.Axis,
        numsamples: int = 100,
        rng: Union[int, np.random.Generator, None] = None,
        **kwargs: Any,
    ):
        # we defer the initialization of these variables until _intialize.
        self._nominal: bh.Histogram = None  # type: ignore
        self._hist: bh.Histogram = None  # type: ignore
        self._random: np.random.Generator = None  # type: ignore
        axeslist = list(axes)
        nominal = bh.Histogram(*axeslist, **kwargs)
        axeslist.append(bh.axis.Integer(0, numsamples, underflow=False, overflow=False))
        samples = bh.Histogram(*axeslist, **kwargs)
        self._initialize(nominal, samples, rng)

    def _initialize(
        self,
        nominal: bh.Histogram,
        samples: bh.Histogram,
        rng: Union[int, np.random.Generator, None] = None,
    ) -> None:
        self._nominal = nominal
        self._hist = samples
        self._random = np.random.default_rng(rng)
        # when filling with very large arrays, the fast filling method may use too
        # much memory, fall back to the slower method when the array size gets above
        # this threshold
        self._threshold_for_fast_method = 1000000

    @classmethod
    def _from_bh_histogram(
        cls,
        nominal: bh.Histogram,
        samples: bh.Histogram,
        rng: Union[int, np.random.Generator, None],
    ) -> "BootstrapHistogram":
        result = cls.__new__(cls)
        result._nominal = nominal
        result._initialize(nominal, samples, rng)
        return result

    @property
    def nominal(self) -> bh.Histogram:
        """A histogram of the filled values, with no
        bootstrap samples.
        """
        return self._nominal

    @property
    def samples(self) -> bh.Histogram:
        """A histogram of the bootstrap samples. The last
        dimension corresponds to the bootstrap sample index and is of size
        :py:attr:`BootstrapHistogram.numsamples`.
        """
        return self._hist

    def mean(self, flow: bool = False, ignore_nan: bool = True) -> "NDArray[Any]":
        """
        Binned sample mean.

        Parameters
        ----------
        flow: bool
            If True under and overflow bins are included.
        ignore_nan : bool
            If True :py:func:`numpy.nanmean` is used.

        Returns
        -------
        numpy.ndarray
            an array containing the mean value of all bootstrap samples for each bin
            in the histogram.
        """
        mean = np.nanmean if ignore_nan else np.mean
        return np.asarray(mean(self.view(flow=flow), axis=-1))

    def std(self, flow: bool = False, ignore_nan: bool = True) -> "NDArray[Any]":
        """
        Binned sample standard deviation.

        Parameters
        ----------
        flow: bool
            If True under and overflow bins are included.
        ignore_nan : bool
            If True :py:func:`numpy.nanstd` is used.

        Returns
        -------
        numpy.ndarray
            an array containing the standard deviation of all bootstrap sample values
            for each bin in the histogram, .
        """
        std = np.nanstd if ignore_nan else np.std
        return np.asarray(std(self.view(flow=flow), axis=-1))

    def percentile(
        self,
        q: float,
        flow: bool = False,
        interpolation: str = "linear",
        ignore_nan: bool = True,
    ) -> "NDArray[Any]":
        """
        Binned q-th percentile.


        Parameters
        ----------
        q : float
            The percentile, a number between 0 and 100 (inclusive).
        flow: bool
            If True under and overflow bins are included.
        interpolation : str
            As :py:func:`numpy.percentile`.
        ignore_nan : bool
            If True :py:func:`numpy.nanpercentile` is used.

        Returns
        -------
        numpy.ndarray
            an array containing the q-th percentile of all bootstrap samples for each
            bin in the histogram, .
        """
        percentile = np.nanpercentile if ignore_nan else np.percentile
        return np.asarray(
            percentile(  # type: ignore
                self.view(flow=flow),
                q,
                axis=-1,
                interpolation=interpolation,
            )
        )

    @property
    def numsamples(self) -> int:
        """Number of bootstrap re-samplings."""
        return len(self.axes[-1])

    @property
    def axes(self) -> Tuple[bh.axis.Axis, ...]:
        """:py:class:`boost_histogram.axis.Axis` representing
        the histogram binning. The last dimension corresponds to the bootstrap sample
        index."""
        return self._hist.axes

    def fill(
        self,
        *args: "ArrayLike",
        weight: Optional["ArrayLike"] = None,
        seed: Optional["ArrayLike"] = None,
        **kwargs: Any,
    ) -> "BootstrapHistogram":
        """
        Fill the histogram with some values.


        Parameters
        ----------
        *args : "ArrayLike"
            An 1D array containing coordinates for each dimension of the histogram.
        weight : Optional["ArrayLike"]
            Entry weights used to fill the histogram.
        seed: Optional["ArrayLike"]
            Per-element seed. Overrides the Generator given in the constructor and
            uses a pseudo-random number generator seeded by the given value.
            This may be useful when filling multiple histograms with data that is not
            statistically independent (where it may be desirable to seed the generator
            with a record ID).
        **kwargs : Any
            Passed on to :py:class:`boost_histogram.Histogram.fill`.

        Returns
        -------
        self : BootstrapHistogram
            Reference to this object. This is done to maintain consistency with
            `boost_histogram.Histogram`.
        """
        arrays = tuple(np.asarray(a) for a in args)
        weight = np.asarray(weight) if weight is not None else None
        seed = np.asarray(seed) if seed is not None else None
        self._validate_fill_inputs(arrays, weight, seed)
        if (self.numsamples * arrays[0].size) < self._threshold_for_fast_method:
            return self._fill_fast(*arrays, weight=weight, seed=seed, **kwargs)
        else:
            return self._fill_slow(*arrays, weight=weight, seed=seed, **kwargs)

    @staticmethod
    def _validate_fill_inputs(
        args: Tuple["NDArray[Any]", ...],
        weight: Optional["NDArray[Any]"] = None,
        seed: Optional["NDArray[Any]"] = None,
    ) -> None:
        if len(args) <= 0:
            raise ValueError("fill must be provided at least 1 array as input.")
        sizes = [a.size for a in args]
        if not all(s == sizes[0] for s in sizes):
            raise ValueError("all arrays must have the same length.")
        if weight is not None and weight.size != sizes[0]:
            raise ValueError(
                "weight array size does not match the other input array sizes."
            )
        if seed is not None and seed.size != sizes[0]:
            raise ValueError(
                "seed array size does not match the other input array sizes."
            )

    def _fill_fast(
        self,
        *args: "NDArray[Any]",
        weight: Optional["NDArray[Any]"] = None,
        seed: Optional["NDArray[Any]"] = None,
        **kwargs: Any,
    ) -> "BootstrapHistogram":
        self._nominal.fill(*args, weight=weight, **kwargs)
        hist = self._hist
        shape = (self.numsamples,) + np.shape(args[0])
        if seed is not None:
            generators = np.asarray(
                [np.random.Generator(np.random.PCG64(s)) for s in seed]
            )
        flatiter = tuple(np.broadcast_to(values, shape).T.flat for values in args)
        index = np.broadcast_to(np.arange(self.numsamples), reversed(shape)).flat
        if seed is None:
            sampleweights = np.asarray(self._random.poisson(1.0, size=shape))
        else:
            sampleweights = np.asarray(
                [r.poisson(1.0, size=(self.numsamples,)) for r in generators],
                dtype=np.float64,
            ).T
        sampleweights = sampleweights.T
        if weight is not None:
            shapedweight = np.broadcast_to(weight, shape).T
            assert sampleweights.shape == shapedweight.shape
            sampleweights = sampleweights * shapedweight
        hist.fill(*flatiter, index, weight=sampleweights.flat, **kwargs)
        return self

    def _fill_slow(
        self,
        *args: "NDArray[Any]",
        weight: Optional["NDArray[Any]"] = None,
        seed: Optional["NDArray[Any]"] = None,
        **kwargs: Any,
    ) -> "BootstrapHistogram":
        self._nominal.fill(*args, weight=weight, **kwargs)
        hist = self._hist
        shape = np.shape(args[0])
        if seed is not None:
            generators = np.asarray(
                [np.random.Generator(np.random.PCG64(s)) for s in seed]
            )
        for index in range(self.numsamples):
            if seed is None:
                sampleweights = self._random.poisson(1.0, size=shape)
            else:
                sampleweights = np.fromiter(
                    (r.poisson(1.0) for r in generators),
                    dtype=np.float64,
                    count=len(generators),
                )
            if weight is not None:
                sampleweights = sampleweights * weight
            hist.fill(*args, index, weight=sampleweights, **kwargs)
        return self

    def view(self, flow: bool = False) -> Any:
        """
        Return a view of the underlying histogram bootstrap sample values.
        """
        return self._hist.view(flow=flow)

    def __eq__(self, other: object) -> bool:
        """
        BootstrapHistogram's are considered to be equal if the data in their underlying
        histograms are equal.
        """
        return (
            isinstance(other, BootstrapHistogram)
            and self._hist == other._hist
            and self._nominal == other._nominal
        )

    def __add__(
        self, other: Union["BootstrapHistogram", "ArrayLike", float]
    ) -> "BootstrapHistogram":
        """
        Add together the values of two bootstrap histograms together.

        This is useful to merge results when parallellizing filling in a map-reduce
        pattern. This histograms must have the same binning and the same number of
        bootstrap samples. The merged histogram will have the same binning and the same
        number of bootstrap samples as the input histograms.

        Returns
        -------
        hist : BootstrapHistogram
            A new instance of the summed histogram.
        """
        result = deepcopy(self)
        if isinstance(other, BootstrapHistogram):
            result._hist += other._hist
            result._nominal += other._nominal
        else:
            result._hist += other  # type: ignore
            result._nominal += other  # type: ignore
        return result

    def __radd__(
        self, other: Union["BootstrapHistogram", "ArrayLike", float]
    ) -> "BootstrapHistogram":
        return self + other

    def __sub__(
        self, other: Union["BootstrapHistogram", "ArrayLike", float]
    ) -> "BootstrapHistogram":
        result = deepcopy(self)
        if isinstance(other, BootstrapHistogram):
            result._hist -= other._hist
            result._nominal -= other._nominal
        else:
            result._hist -= other  # type: ignore
            result._nominal -= other  # type: ignore
        return result

    def __mul__(
        self, other: Union["BootstrapHistogram", "ArrayLike", float]
    ) -> "BootstrapHistogram":
        result = deepcopy(self)
        if isinstance(other, BootstrapHistogram):
            result._hist *= other._hist
            result._nominal *= other._nominal
        else:
            result._hist *= other  # type: ignore
            result._nominal *= other  # type: ignore
        return result

    def __rmul__(
        self, other: Union["BootstrapHistogram", "ArrayLike", float]
    ) -> "BootstrapHistogram":
        return self * other

    def __truediv__(
        self, other: Union["BootstrapHistogram", "ArrayLike", float]
    ) -> "BootstrapHistogram":
        result = deepcopy(self)
        if isinstance(other, BootstrapHistogram):
            result._hist /= other._hist
            result._nominal /= other._nominal
        else:
            result._hist /= other  # type: ignore
            result._nominal /= other  # type: ignore
        return result

    def project(self, *args: int) -> "BootstrapHistogram":
        """
        Reduce histogram dimensionality by summing over some dimensions.
        The bootstrap sample axis is always kept by this operation.

        Parameters
        ----------
        *args: int
            The dimensions to be kept.

        Returns
        -------
        hist: BootstrapHistogram
            a copy of the histogram with only axes in args and the bootstrap sample
            axes.
        """
        #  pylint: disable=protected-access
        result = copy(self)
        arglist = list(args)
        sampleaxis = len(self.axes) - 1
        result._nominal = cast(
            bh.Histogram,
            self._nominal.project(*[arg for arg in arglist if arg != sampleaxis]),
        )
        if sampleaxis not in arglist:
            arglist.append(sampleaxis)
        result._hist = cast(bh.Histogram, self._hist.project(*arglist))
        return result
