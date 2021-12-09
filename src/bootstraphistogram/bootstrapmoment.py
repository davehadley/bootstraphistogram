"""Tools for calculating the moments of a data set."""
from copy import deepcopy
from typing import Any, Optional, Union

import boost_histogram as bh
import numpy as np

from bootstraphistogram.bootstraphistogram import BootstrapHistogram
from bootstraphistogram.valuewithsamples import ValueWithSamples

try:
    from numpy.typing import ArrayLike  # pylint: disable=E0611,E0401
except (ImportError, ModuleNotFoundError):
    ArrayLike = np.ndarray


class BootstrapMoment:
    """
    Computes the mean, variance and skewness of a (optionally weighted) dataset with
    bootstrap resampling.

    Parameters
    ----------
    numsamples : int
        The number of bootstrap samples. Increasing this number improves the accuracy
        of estimators derived from the bootstrap samples, at the cost of increased
        memory and CPU usage.
    rng : Union[int, np.random.Generator, None]
        A numpy generator. If not provided, the numpy default from
        :py:func:`numpy.random.default_rng` will be used.
    **kwargs: Any
        Passed on to the underlying :py:class:`bootstraphistogram.BootstrapHistogram`.

    Examples
    --------

    To create and fill an instance:

    >>> from bootstraphistogram import BootstrapMoment
    >>> import numpy as np
    >>> moments = BootstrapMoment(3, rng=1234)
    >>> moments.fill(np.arange(101))

    Compute the mean:

    >>> moments.mean().nominal
    50.0
    >>> moments.mean().samples
    array([51.4, 50.3, 46.1 ])

    Compute the standard deviation:

    >>> moments.std().nominal
    29.15
    >>> moments.std().samples
    array([29.5, 29.5, 28.6])

    Compute the skewness:

    >>> moments.skewness().nominal
    0.0
    >>> moments.skewness().samples
    array([-0.00,  0.10,  0.06])
    """

    def __init__(
        self,
        numsamples: int = 100,
        rng: Union[int, np.random.Generator, None] = None,
        **kwargs: Any,
    ):
        ax = bh.axis.Regular(1, -1.0, 1.0)
        # provide identical generator to each histogram to ensure that sample weights
        # are identical
        if rng is None:
            rng = int(np.random.default_rng(rng).integers(np.iinfo(int).max))
        # we must deepcopy the rng in case it is a Generator with some internal state
        self._sum_w = BootstrapHistogram(
            ax, numsamples=numsamples, rng=deepcopy(rng), **kwargs
        )
        self._sum_wt = BootstrapHistogram(
            ax, numsamples=numsamples, rng=deepcopy(rng), **kwargs
        )
        self._sum_wt2 = BootstrapHistogram(
            ax, numsamples=numsamples, rng=deepcopy(rng), **kwargs
        )
        self._sum_wt3 = BootstrapHistogram(
            ax, numsamples=numsamples, rng=deepcopy(rng), **kwargs
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BootstrapMoment)
            and self._sum_w == other._sum_w
            and self._sum_wt == other._sum_wt
            and self._sum_wt2 == other._sum_wt2
            and self._sum_wt3 == other._sum_wt3
        )

    def fill(
        self,
        values: ArrayLike,
        weight: Optional[ArrayLike] = None,
        seed: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> None:
        """
        Fill the object with some values.


        Parameters
        ----------
        values : ArrayLike
            A 1D array containing the values from which moments will be calculated.
        weight : Optional[ArrayLike]
            weights associated with the values.
        seed: Optional[ArrayLike]
            Per-element seed. Overrides the Generator given in the constructor and
            uses a pseudo-random number generator seeded by the given value.
            In some cases it is desirable to seed the generator with a record ID to
            allow bootstrap samples to be statistically correlated between objects.
        **kwargs : Any
            Passed on to the underlying :py:class:`boost_histogram.Histogram.fill`.
        """
        values = np.array(values)
        if weight is None:
            weight = np.ones(values.shape)
        wt = values * np.array(weight)
        wt2 = values * values * np.array(weight)
        wt3 = values * values * values * np.array(weight)
        zeros = np.zeros(values.shape)
        self._sum_w.fill(zeros, weight=weight, seed=seed, **kwargs)
        self._sum_wt.fill(zeros, weight=wt, seed=seed, **kwargs)
        self._sum_wt2.fill(zeros, weight=wt2, seed=seed, **kwargs)
        self._sum_wt3.fill(zeros, weight=wt3, seed=seed, **kwargs)

    def mean(self) -> ValueWithSamples:
        """
        Compute the mean.

        Returns
        -------
        ValueWithSamples
            the (weighted) mean of the (weighted) fill values and bootstrap resamples.
        """
        nominal = float(
            _mean(sumwt=self._sum_wt.nominal.view(), sumw=self._sum_w.nominal.view())
        )
        samples = _mean(
            sumwt=self._sum_wt.samples.view(), sumw=self._sum_w.samples.view()
        )
        return ValueWithSamples(nominal, samples.flatten())

    def variance(self) -> ValueWithSamples:
        """
        Compute the variance.

        Returns
        -------
        ValueWithSamples
            the (weighted) variance of the (weighted) fill values and bootstrap
            resamples.
        """
        nominal = float(
            _variance(
                sumw=self._sum_w.nominal.view(),
                sumwt=self._sum_wt.nominal.view(),
                sumwt2=self._sum_wt2.nominal.view(),
            )
        )
        samples = _variance(
            sumw=self._sum_w.samples.view(),
            sumwt=self._sum_wt.samples.view(),
            sumwt2=self._sum_wt2.samples.view(),
        )
        return ValueWithSamples(nominal, samples.flatten())

    def std(self) -> ValueWithSamples:
        """
        Compute the standard deviation.

        Returns
        -------
        ValueWithSamples
            the (weighted) standard deviation of the (weighted) fill values and
            bootstrap resamples.
        """
        variance = self.variance()
        return ValueWithSamples(np.sqrt(variance.nominal), np.sqrt(variance.samples))

    def skewness(self) -> ValueWithSamples:
        """
        Compute the skewness.

        Returns
        -------
        ValueWithSamples
            the (weighted) skewness of the (weighted) fill values and bootstrap
            resamples.
        """
        nominal = float(
            _skewness(
                sumw=self._sum_w.nominal.view(),
                sumwt=self._sum_wt.nominal.view(),
                sumwt2=self._sum_wt2.nominal.view(),
                sumwt3=self._sum_wt3.nominal.view(),
            )
        )
        samples = _skewness(
            sumw=self._sum_w.samples.view(),
            sumwt=self._sum_wt.samples.view(),
            sumwt2=self._sum_wt2.samples.view(),
            sumwt3=self._sum_wt3.samples.view(),
        )
        return ValueWithSamples(nominal, samples.flatten())

    @property
    def numsamples(self) -> int:
        """Number of bootstrap re-samplings."""
        return self._sum_w.numsamples

    def __add__(self: "BootstrapMoment", rhs: "BootstrapMoment") -> "BootstrapMoment":
        """Merge two BootstrapMoment objects together by summing their underlying histograms."""
        result = deepcopy(self)
        result._sum_w += rhs._sum_w
        result._sum_wt += rhs._sum_wt
        result._sum_wt2 += rhs._sum_wt2
        result._sum_wt3 += rhs._sum_wt3
        return result


def _mean(sumw: ArrayLike, sumwt: ArrayLike) -> ArrayLike:
    return np.divide(sumwt, sumw)


def _variance(sumw: ArrayLike, sumwt: ArrayLike, sumwt2: ArrayLike) -> ArrayLike:
    mu = _mean(sumwt=sumwt, sumw=sumw)
    mu2 = np.multiply(mu, mu)
    return mu2 + np.divide((sumwt2 - 2.0 * np.multiply(sumwt, mu)), sumw)


def _skewness(
    sumw: ArrayLike, sumwt: ArrayLike, sumwt2: ArrayLike, sumwt3: ArrayLike
) -> ArrayLike:
    mu = _mean(sumwt=sumwt, sumw=sumw)
    sigma = np.sqrt(_variance(sumw=sumw, sumwt=sumwt, sumwt2=sumwt2))
    mut3 = np.divide(sumwt3, sumw)
    return np.divide(
        (mut3 - 3 * np.multiply(mu, np.power(sigma, 2)) - np.power(mu, 3)),
        np.power(sigma, 3),
    )
