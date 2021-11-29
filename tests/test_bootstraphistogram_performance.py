from timeit import timeit

import boost_histogram as bh
import numpy as np
import pytest

from bootstraphistogram import BootstrapHistogram


@pytest.mark.parametrize(
    "withweight", [True, False], ids=["withweight", "withoutweight"]
)
@pytest.mark.parametrize("withseed", [True, False], ids=["withseed", "withoutseed"])
def test_boostraphistogram_fast_and_slow_filling_give_identical_results(
    withweight, withseed
):
    size = 10000
    hist_slow = BootstrapHistogram(
        bh.axis.Regular(100, 0.0, 1.0), numsamples=100, rng=1234
    )
    hist_fast = BootstrapHistogram(
        bh.axis.Regular(100, 0.0, 1.0), numsamples=100, rng=1234
    )
    data = np.random.uniform(size=size)
    weight = np.random.uniform(size=size) if withweight else None
    seed = np.arange(size) if withseed else None
    hist_slow._fill_slow(data, weight=weight, seed=seed)
    hist_fast._fill_fast(data, weight=weight, seed=seed)
    assert hist_slow == hist_fast


@pytest.mark.parametrize(
    "withweight", [True, False], ids=["withweight", "withoutweight"]
)
@pytest.mark.parametrize("withseed", [True, False], ids=["withseed", "withoutseed"])
def test_boostraphistogram_fast_is_faster_than_slow(withweight, withseed):
    numsamples = 10000
    arraysize = 100
    hist_slow = BootstrapHistogram(
        bh.axis.Regular(100, 0.0, 1.0), numsamples=numsamples, rng=1234
    )
    hist_fast = BootstrapHistogram(
        bh.axis.Regular(100, 0.0, 1.0), numsamples=numsamples, rng=1234
    )
    data = np.random.uniform(size=arraysize)
    weight = np.random.uniform(size=arraysize) if withweight else None
    seed = np.arange(arraysize) if withseed else None
    slowtime = timeit(
        lambda: hist_slow._fill_slow(data, weight=weight, seed=seed), number=5
    )
    fasttime = timeit(
        lambda: hist_fast._fill_fast(data, weight=weight, seed=seed), number=5
    )
    assert slowtime > fasttime
