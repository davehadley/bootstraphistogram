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
        # are indentical
        if rng is None:
            rng = int(np.random.default_rng(rng).integers(np.iinfo(int).max))
        # we must deepcopy the rng in case it is a Generator with some internal state
        self._sum_w = BootstrapHistogram(ax, numsamples=numsamples, rng=deepcopy(rng))
        self._sum_w2 = BootstrapHistogram(ax, numsamples=numsamples, rng=deepcopy(rng))
        self._sum_wt = BootstrapHistogram(ax, numsamples=numsamples, rng=deepcopy(rng))
        self._sum_wt2 = BootstrapHistogram(ax, numsamples=numsamples, rng=deepcopy(rng))
        # self._sum_wt3 = BootstrapHistogram(ax, numsamples=numsamples, rng=deepcopy(rng))

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
        w2 = np.array(weight) * np.array(weight)
        wt2 = values * values * np.array(weight)
        # wt3 = values * values * values * np.array(weight)
        self._sum_w.fill(np.zeros(values.shape), weight=weight, seed=seed, **kwargs)
        self._sum_w2.fill(np.zeros(values.shape), weight=w2, seed=seed, **kwargs)
        self._sum_wt.fill(np.zeros(values.shape), weight=wt, seed=seed, **kwargs)
        self._sum_wt2.fill(np.zeros(values.shape), weight=wt2, seed=seed, **kwargs)
        # self._sum_wt3.fill(np.zeros(values.shape), weight=wt3, seed=seed, **kwargs)

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
                sumwt=self._sum_wt.nominal.view(),
                sumw=self._sum_w.nominal.view(),
                sumwt2=self._sum_wt2.nominal.view(),
            )
        )
        samples = _variance(
            sumwt=self._sum_wt.samples.view(),
            sumw=self._sum_w.samples.view(),
            sumwt2=self._sum_wt2.samples.view(),
        )
        return Moment(nominal, samples.flatten())

    def std(self) -> Moment:
        variance = self.variance()
        return Moment(np.sqrt(variance.nominal), np.sqrt(variance.samples))

    def skewness(self) -> Moment:
        # return (x3 - 3*mu*sigma**2 - mu**3) / sigma**3
        pass

    def kurtosis(self) -> Moment:
        pass

    @property
    def numsamples(self) -> int:
        return self._sum_w.numsamples


def _mean(sumw: ArrayLike, sumwt: ArrayLike) -> ArrayLike:
    return sumwt / sumw


def _variance(sumw: ArrayLike, sumwt: ArrayLike, sumwt2: ArrayLike) -> ArrayLike:
    mu = _mean(sumwt=sumwt, sumw=sumw)
    mu2 = mu * mu
    return mu2 + ((sumwt2 - 2.0 * sumwt * mu) / sumw)
