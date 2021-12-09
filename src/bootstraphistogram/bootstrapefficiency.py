"""Implements :py:class:`BoostrapEfficiency`, a tool for calculating binned efficiencies."""

from copy import copy, deepcopy
from typing import Any, NamedTuple, Optional, Tuple, Union

import boost_histogram as bh
import numpy as np

from bootstraphistogram.bootstraphistogram import BootstrapHistogram

try:
    from numpy.typing import ArrayLike
except (ImportError, ModuleNotFoundError):
    ArrayLike = np.ndarray


class BootstrapEfficiency:
    """Calculates binned efficiencies with uncertainties calculated with Poission
    bootstrap resampling.

    Parameters
    ----------
    *axes : boost_histogram.axis.Axis
        Any number of :py:class:`boost_histogram.axis.Axis` objects that define the
        efficiency binning. See
        <https://boost-histogram.readthedocs.io/en/latest/usage/axes.html>.
    numsamples : int
        The number of bootstrap samples. Increasing this number improves the accuracy
        of estimators derived from the bootstrap samples, at the cost of increased
        memory and CPU usage.
    rng : Union[int, np.random.Generator, None]
        A numpy generator. If not provided, the numpy default from
        :py:func:`numpy.random.default_rng` will be used.
    nanto: Optional[float]
        When calculating efficiencies empty bins will result in NaN.
        If not None these values will be set to `nanto`.
    **kwargs : Any
        Passed on to the :py:class:`bootstraphistogram.BootstrapHistogram` constructor.
        :py:class:`numpy.ndarray`
    """

    class Histogram(NamedTuple):
        """A result type to store histograms returned by
        :py:class:`bootstraphistogram.BootstrapEfficiency`"""

        numerator: bh.Histogram
        denominator: bh.Histogram
        efficiency: bh.Histogram

    class Array(NamedTuple):
        """A result type to store arrays returned by
        :py:class:`bootstraphistogram.BootstrapEfficiency`"""

        numerator: np.ndarray
        denominator: np.ndarray
        efficiency: np.ndarray

    def __init__(
        self,
        *axes: bh.axis.Axis,
        numsamples: int = 100,
        rng: Union[int, np.random.Generator, None] = None,
        nanto: Optional[float] = None,
        **kwargs: Any,
    ):
        self._hist = BootstrapHistogram(
            bh.axis.Integer(0, 2, overflow=False, underflow=False),
            *axes,
            numsamples=numsamples,
            rng=rng,
            **kwargs,
        )
        self._nanto = nanto

    @staticmethod
    def _hist_to_result(
        hist: bh.Histogram, nanto: Optional[float] = None
    ) -> "BootstrapEfficiency.Histogram":
        numerator = hist[bh.loc(1), ...]
        notselected = hist[bh.loc(0), ...]
        denominator = numerator + notselected
        ratio = numerator / denominator
        if nanto is not None:
            np.nan_to_num(
                ratio.view(), copy=False, nan=nanto, posinf=nanto, neginf=nanto
            )
        return BootstrapEfficiency.Histogram(numerator, denominator, ratio)

    @property
    def efficiency(self) -> BootstrapHistogram:
        """The efficiency as a `BootstrapHistogram`."""
        #  pylint: disable=protected-access
        return BootstrapHistogram._from_bh_histogram(
            self.nominal.efficiency, self.samples.efficiency, self._hist._random
        )

    @property
    def numerator(self) -> BootstrapHistogram:
        """The numerator as a `BootstrapHistogram`."""
        #  pylint: disable=protected-access
        return BootstrapHistogram._from_bh_histogram(
            self.nominal.numerator, self.samples.numerator, self._hist._random
        )

    @property
    def denominator(self) -> BootstrapHistogram:
        """The denominator as a `BootstrapHistogram`."""
        #  pylint: disable=protected-access
        return BootstrapHistogram._from_bh_histogram(
            self.nominal.denominator, self.samples.denominator, self._hist._random
        )

    @property
    def nominal(self) -> "BootstrapEfficiency.Histogram":
        """A histogram of the filled values, with no
        bootstrap re-sampling applied.
        """
        return self._hist_to_result(self._hist.nominal, nanto=self._nanto)

    @property
    def samples(self) -> "BootstrapEfficiency.Histogram":
        """A histogram of the bootstrap samples. The last
        dimension corresponds to the bootstrap sample index and is of size
        :py:attr:`BootstrapEfficiency.numsamples`.
        """
        return self._hist_to_result(self._hist.samples, nanto=self._nanto)

    def mean(self, flow=False) -> "BootstrapEfficiency.Array":
        """Binned mean of the bootstrap samples."""
        samples = self.samples
        return BootstrapEfficiency.Array(
            *[np.mean(hst.view(flow=flow), axis=-1) for hst in samples]
        )

    def std(self, flow=False) -> np.ndarray:
        """Binned standard deviation of the boostrap samples."""
        samples = self.samples
        return BootstrapEfficiency.Array(
            *[np.std(hst.view(flow=flow), axis=-1) for hst in samples]
        )

    def percentile(
        self, q: float, flow=False, interpolation: str = "linear"
    ) -> np.ndarray:
        """
        Binned q-th percentile of the bootstrap samples.

        Parameters
        ----------
        q : float
            The percentile, a number between 0 and 100 (inclusive).
        interpolation : str
            As :py:func:`numpy.percentile`.

        Returns
        -------
        numpy.ndarray
            an array containing the q-th percentile of all bootstrap samples for each
            bin in the histogram, .
        """
        samples = self.samples
        return BootstrapEfficiency.Array(
            *[
                np.nanpercentile(
                    hst.view(flow=flow), q, axis=-1, interpolation=interpolation
                )
                for hst in samples
            ]
        )

    @property
    def numsamples(self) -> int:
        """Number of bootstrap re-samplings."""
        return self._hist.numsamples

    @property
    def axes(self) -> Tuple[bh.axis.Axis, ...]:
        """:py:class:`boost_histogram.axis.Axis` representing
        the histogram binning.
        The first dimension corresponds to whether an entry is included in the numerator
        or not (index 0 = not in numerator, index 1 = included in numerator).
        The last dimension corresponds to the bootstrap sample index."""
        return self._hist.axes

    def fill(
        self,
        selected: ArrayLike,
        *args: ArrayLike,
        weight: Optional[ArrayLike] = None,
        seed: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> "BootstrapEfficiency":
        """
        Fill the histogram with some values.


        Parameters
        ----------
        selected: ArrayLike
            A 1D boolean array determining whether an event enters the numerator or
            denominator.
        *args : ArrayLike
            An 1D array containing coordinates for each dimension of the histogram.
        weight : Optional[ArrayLike]
            Entry weights used to fill the histogram.
        seed: Optional[ArrayLike]
            Per-element seed. Overrides the Generator given in the constructor and
            uses a pseudo-random number generator seeded by the given value.
            This may be useful when filling multiple histograms with data that is not
            statistically independent (where it may be desirable to seed the generator
            with a record ID).
        **kwargs : Any
            Passed on to :py:class:`boostraphistogram.BootstrapHistogram.fill`.

        Returns
        -------
        self : BootstrapEfficiency
            Reference to this object. This is done to maintain consistency with
            `boost_histogram.Histogram`.
        """
        args = tuple(np.asarray(a) for a in args)
        selected = np.asarray(selected).astype(bool)
        self._validate_fill_inputs(args, selected)
        self._hist.fill(selected, *args, weight=weight, seed=seed, **kwargs)
        return self

    @staticmethod
    def _validate_fill_inputs(
        args: ArrayLike,
        selected: ArrayLike,
    ) -> None:
        if len(args) <= 0:
            raise ValueError("fill must be provided at least 1 array as input.")
        sizes = [a.size for a in args]
        if selected.size != sizes[0]:
            raise ValueError(
                "selected array size does not match the other input array sizes."
            )

    def view(self, flow=False) -> Any:
        """
        Return a view of the underlying histogram bootstrap sample values.
        """
        return self._hist.view(flow=flow)

    def __eq__(self, other: object) -> bool:
        """
        BootstrapEfficiency's are considered to be equal if the data in their underlying
        histograms are equal.
        """
        return isinstance(other, BootstrapEfficiency) and self._hist == other._hist

    def __add__(self, other: "BootstrapEfficiency") -> "BootstrapEfficiency":
        """
        Add together the values of two bootstrap histograms together.

        This is useful to merge results when parallellizing filling in a map-reduce
        pattern. The histograms must have the same binning and the same number of
        bootstrap samples. The merged histogram will have the same binning and the same
        number of bootstrap samples as the input histograms.

        Returns
        -------
        hist : BootstrapEfficiency
            A new instance of the summed histogram.
        """
        result = deepcopy(self)
        result._hist += other._hist
        return result

    def project(self, *args: int) -> "BootstrapEfficiency":
        """
        Reduce histogram dimensionality by summing over some dimensions.
        The efficiency "selected" axis (first axis) and the bootstrap sample axis
        (final axis) are always kept by this operation.

        Parameters
        ----------
        *args: int
            The dimensions to be kept.

        Returns
        -------
        hist: BootstrapEfficiency
            a copy of the histogram with only axes in args and the bootstrap sample
            axes.
        """
        #  pylint: disable=protected-access
        result = copy(self)
        arglist = list(args)
        selectedaxis = 0
        if selectedaxis not in arglist:
            arglist = [selectedaxis] + arglist
        result._hist = result._hist.project(*arglist)
        return result
