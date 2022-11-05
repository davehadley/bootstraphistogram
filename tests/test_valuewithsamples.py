import numpy as np

from bootstraphistogram.valuewithsamples import ValueWithSamples


def test_valuewithsamples_addition() -> None:
    lhs = ValueWithSamples(1, np.array([1, 2, 3]))
    rhs = ValueWithSamples(2, np.array([4, 5, 6]))
    total = lhs + rhs
    assert total.nominal == 3
    assert np.array_equal(total.samples, [5, 7, 9])


def test_valuewithsamples_subtraction() -> None:
    lhs = ValueWithSamples(1, np.array([1, 2, 3]))
    rhs = ValueWithSamples(2, np.array([4, 5, 6]))
    total = lhs - rhs
    assert total.nominal == -1
    assert np.array_equal(total.samples, [-3, -3, -3])


def test_valuewithsamples_multiplication() -> None:
    lhs = ValueWithSamples(2, np.array([1, 2, 3]))
    rhs = ValueWithSamples(3, np.array([4, 5, 6]))
    total = lhs * rhs
    assert total.nominal == 6
    assert np.array_equal(total.samples, [4, 10, 18])


def test_valuewithsamples_division() -> None:
    lhs = ValueWithSamples(4, np.array([4, 9, 16]))
    rhs = ValueWithSamples(2, np.array([2, 3, 4]))
    total = lhs / rhs
    assert total.nominal == 2
    assert np.array_equal(total.samples, [2, 3, 4])


def test_valuewithsamples_equality() -> None:
    value = ValueWithSamples(1, np.array([1, 2, 3]))
    isequal = ValueWithSamples(1, np.array([1, 2, 3]))
    isnotequal_1 = ValueWithSamples(2, np.array([1, 2, 3]))
    isnotequal_2 = ValueWithSamples(1, np.array([2, 2, 3]))
    isnotequal_3 = ValueWithSamples(2, np.array([2, 2, 3]))
    assert value == isequal
    assert value != isnotequal_1
    assert value != isnotequal_2
    assert value != isnotequal_3
