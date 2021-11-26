from copy import deepcopy
from typing import Any, NamedTuple, Optional, Union

import boost_histogram as bh
import numpy as np

from bootstraphistogram.bootstraphistogram import BootstrapHistogram

try:
    from numpy.typing import ArrayLike
except Exception:
    ArrayLike = np.ndarray


class Moment(NamedTuple):
    nominal: float
    samples: np.ndarray


class BootstrapMoment:
    def __init__(
        self, numsamples: int = 1000, rng: Union[int, np.random.Generator, None] = None
    ):
        ax = bh.axis.Regular(1, -1.0, 1.0)
        # provide identical generator to each histogram to ensure that sample weights
        # are identical
        if rng is None:
            rng = int(np.random.default_rng(rng).integers(np.iinfo(int).max))
        # we must deepcopy the rng in case it is a Generator with some internal state
        self._sum_w = BootstrapHistogram(ax, numsamples=numsamples, rng=deepcopy(rng))
        self._sum_wt = BootstrapHistogram(ax, numsamples=numsamples, rng=deepcopy(rng))
        self._sum_wt2 = BootstrapHistogram(ax, numsamples=numsamples, rng=deepcopy(rng))
        self._sum_wt3 = BootstrapHistogram(ax, numsamples=numsamples, rng=deepcopy(rng))

    def fill(
        self,
        values: ArrayLike,
        weight: Optional[ArrayLike] = None,
        seed: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> None:
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

    def mean(self) -> Moment:
        nominal = float(
            _mean(sumwt=self._sum_wt.nominal.view(), sumw=self._sum_w.nominal.view())
        )
        samples = _mean(
            sumwt=self._sum_wt.samples.view(), sumw=self._sum_w.samples.view()
        )
        return Moment(nominal, samples.flatten())

    def variance(self) -> Moment:
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
        return Moment(nominal, samples.flatten())

    def std(self) -> Moment:
        variance = self.variance()
        return Moment(np.sqrt(variance.nominal), np.sqrt(variance.samples))

    def skewness(self) -> Moment:
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
        return Moment(nominal, samples)

    @property
    def numsamples(self) -> int:
        return self._sum_w.numsamples


def _mean(sumw: ArrayLike, sumwt: ArrayLike) -> ArrayLike:
    return sumwt / sumw


def _variance(sumw: ArrayLike, sumwt: ArrayLike, sumwt2: ArrayLike) -> ArrayLike:
    mu = _mean(sumwt=sumwt, sumw=sumw)
    mu2 = mu * mu
    return mu2 + ((sumwt2 - 2.0 * sumwt * mu) / sumw)


def _skewness(
    sumw: ArrayLike, sumwt: ArrayLike, sumwt2: ArrayLike, sumwt3: ArrayLike
) -> ArrayLike:
    mu = _mean(sumwt=sumwt, sumw=sumw)
    sigma = np.sqrt(_variance(sumw=sumw, sumwt=sumwt, sumwt2=sumwt2))
    mut3 = sumwt3 / sumw
    return (mut3 - 3 * mu * np.power(sigma, 2) - np.power(mu, 3)) / np.power(sigma, 3)
