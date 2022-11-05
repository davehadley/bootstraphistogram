import numpy as np
import pytest
from boost_histogram.axis import Integer, Regular

from bootstraphistogram.bootstrapefficiency import BootstrapEfficiency


def test_bootstrap_efficiency_fill_1d_nominal() -> None:
    efficiency = BootstrapEfficiency(Regular(3, 0.0, 3.0), rng=1234)
    efficiency.fill(
        [1, 1, 0, 0, 0, 1],
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
    )
    nominal = efficiency.nominal
    assert np.all(nominal.numerator.view() == np.array([2.0, 0.0, 1.0]))
    assert np.all(nominal.denominator.view() == np.array([2.0, 2.0, 2.0]))
    assert np.all(nominal.efficiency.view() == np.array([1.0, 0.0, 0.5]))


def test_bootstrap_efficiency_samples() -> None:
    efficiency = BootstrapEfficiency(Regular(3, 0.0, 3.0), numsamples=1000, rng=1234)
    efficiency.fill(
        [1, 1, 0, 0, 0, 1] * 1000,
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0] * 1000,
    )
    mean = efficiency.mean()
    std = efficiency.std()
    assert mean.numerator == pytest.approx(np.array([2000.0, 0.0, 1000.0]), rel=0.05)
    assert mean.denominator == pytest.approx(
        np.array([2000.0, 2000.0, 2000.0]), rel=0.05
    )
    assert mean.efficiency == pytest.approx(np.array([1.0, 0.0, 0.5]), rel=0.05)
    assert std.numerator == pytest.approx(np.sqrt([2000.0, 0.0, 1000.0]), rel=0.05)
    assert std.denominator == pytest.approx(np.sqrt([2000.0, 2000.0, 2000.0]), rel=0.05)
    assert std.efficiency == pytest.approx(
        np.array([0.0, 0.0, 0.5 / np.sqrt(2000.0)]), rel=0.05
    )


def test_bootstrap_efficiency_samples_percentile() -> None:
    efficiency = BootstrapEfficiency(Regular(3, 0.0, 3.0), numsamples=1000, rng=1234)
    rng = np.random.default_rng(5678)
    efficiency.fill(
        rng.integers(2, size=10000),
        rng.uniform(0.0, 3.0, size=10000),
    )
    mean = efficiency.mean()
    std = efficiency.std()
    minus1sig = efficiency.percentile(50.0 - 34.1)
    plus1sig = efficiency.percentile(50.0 + 34.1)
    assert minus1sig.numerator == pytest.approx(
        mean.numerator - std.numerator, rel=0.01
    )
    assert minus1sig.denominator == pytest.approx(
        mean.denominator - std.denominator, rel=0.01
    )
    assert minus1sig.efficiency == pytest.approx(
        mean.efficiency - std.efficiency, rel=0.01
    )
    assert plus1sig.numerator == pytest.approx(mean.numerator + std.numerator, rel=0.01)
    assert plus1sig.denominator == pytest.approx(
        mean.denominator + std.denominator, rel=0.01
    )
    assert plus1sig.efficiency == pytest.approx(
        mean.efficiency + std.efficiency, rel=0.01
    )


def test_bootstrap_efficiency_numsamples() -> None:
    efficiency = BootstrapEfficiency(Regular(3, 0.0, 3.0), numsamples=123, rng=1234)
    assert efficiency.numsamples == 123


def test_bootstrap_efficiency_axes() -> None:
    efficiency = BootstrapEfficiency(Regular(3, 0.0, 3.0), numsamples=123, rng=1234)
    assert efficiency.axes == (
        Integer(0, 2, underflow=False, overflow=False),
        Regular(3, 0.0, 3.0),
        Integer(0, 123, underflow=False, overflow=False),
    )


def test_bootstrap_efficiency_view() -> None:
    efficiency = BootstrapEfficiency(Regular(3, 0.0, 3.0), numsamples=10, rng=1234)
    expectedwithflow = np.zeros(shape=(2, 5, 10))
    expectedwithoutflow = np.zeros(shape=(2, 3, 10))
    assert np.all(efficiency.view(flow=True) == expectedwithflow)
    assert np.all(efficiency.view(flow=False) == expectedwithoutflow)


def test_bootstrap_efficiency_equality() -> None:
    efficiency1 = BootstrapEfficiency(Regular(3, 0.0, 3.0), rng=1234)
    efficiency2 = BootstrapEfficiency(Regular(3, 0.0, 3.0), rng=1234)
    for efficiency in [efficiency1, efficiency2]:
        efficiency.fill(
            [1, 1, 0, 0, 0, 1],
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        )
    assert efficiency1 == efficiency2
    efficiency2.fill([1], [0.0])
    assert efficiency1 != efficiency2


def test_bootstrap_efficiency_add() -> None:
    efficiency1 = BootstrapEfficiency(Regular(3, 0.0, 3.0))
    efficiency2 = BootstrapEfficiency(Regular(3, 0.0, 3.0))
    efficiency3 = BootstrapEfficiency(Regular(3, 0.0, 3.0))
    for efficiency in [efficiency1, efficiency2, efficiency3, efficiency3]:
        efficiency.fill(
            [1, 1, 0, 0, 0, 1],
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            seed=[1, 2, 3, 4, 5, 6],
        )
    efficiency_added = efficiency1 + efficiency2
    assert efficiency_added == efficiency3


def test_bootstrap_efficiency_project() -> None:
    xax = Regular(2, 0.0, 2.0)
    yax = Regular(3, 0.0, 3.0)
    eff = BootstrapEfficiency(xax, yax, numsamples=4, rng=1234)
    assert eff.view(flow=True).shape == (2, 2 + 2, 3 + 2, 4)
    assert eff.project(0).view(flow=True).shape == (2, 4)
    assert eff.project(1).view(flow=True).shape == (2, 2 + 2, 4)
    assert eff.project(2).view(flow=True).shape == (2, 3 + 2, 4)
    assert eff.project(3).view(flow=True).shape == (2, 4)
    assert eff.project(1, 2).view(flow=True).shape == (2, 2 + 2, 3 + 2, 4)


def test_bootstrap_empty_fill_args_raises() -> None:
    eff = BootstrapEfficiency(Regular(3, 0.0, 3.0))
    with pytest.raises(ValueError):
        eff.fill([1, 0])


def test_fill_with_empty_array() -> None:
    eff = BootstrapEfficiency(Regular(3, 0.0, 3.0), numsamples=9)
    eff.fill([], [])
    assert np.all(eff.view(flow=True) == np.zeros((2, 3 + 2, 9)))


def test_fill_with_mismatched_size_raises() -> None:
    eff = BootstrapEfficiency(Regular(3, 0.0, 3.0), numsamples=9)
    with pytest.raises(ValueError):
        eff.fill([1], [])


def test_nanto() -> None:
    eff = BootstrapEfficiency(Regular(3, 0.0, 3.0), numsamples=9, nanto=0.0)
    assert np.all(eff.nominal.efficiency.view() == [0.0, 0.0, 0.0])


def test_efficiency_property() -> None:
    eff = BootstrapEfficiency(Regular(3, 0.0, 3.0), rng=1234)
    eff.fill(
        [1, 1, 0, 0, 0, 1] * 1000,
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0] * 1000,
    )
    efficiency = eff.efficiency
    numerator = eff.numerator
    denominator = eff.denominator
    assert np.all(numerator.nominal.view() == np.array([2000.0, 0.0, 1000.0]))
    assert np.all(denominator.nominal.view() == np.array([2000.0, 2000.0, 2000.0]))
    assert np.all(efficiency.nominal.view() == np.array([1.0, 0.0, 0.5]))
    assert numerator.mean() == pytest.approx(np.array([2000.0, 0.0, 1000.0]), rel=0.05)
    assert denominator.mean() == pytest.approx(
        np.array([2000.0, 2000.0, 2000.0]), rel=0.05
    )
    assert efficiency.mean() == pytest.approx(np.array([1.0, 0.0, 0.5]), rel=0.05)
