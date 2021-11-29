"""Implements `ValueWithSamples`."""
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


class ValueWithSamples(Generic[T]):
    """Container class storing a calculated value along with its bootstrap resamples.

    Parameters
    ----------
    nominal: T
        the value without any resampling.
    samples: np.ndarray
        the same value with Poisson bootstrap resampling applied.
    """

    def __init__(self, nominal: T, samples: np.ndarray):
        self._nominal = nominal
        self._samples = samples

    @property
    def nominal(self) -> T:
        """the value without any resampling."""
        return self._nominal

    @property
    def samples(self) -> np.ndarray:
        """the value with Poisson bootstrap resampling applied."""
        return self._samples

    def __eq__(self, other: object) -> bool:
        """
        ValueWithSamples are considered to be equal when both the values and samples
        are equal.
        """
        return (
            isinstance(other, ValueWithSamples)
            and self.nominal == other.nominal
            and np.array_equal(self.samples, other.samples)
        )

    def __add__(self, other: "ValueWithSamples") -> "ValueWithSamples":
        """
        Sum both values and their associated samples.

        Returns
        -------
        sum_ : ValueWithSamples
            A new instance of the summed values.
        """
        return ValueWithSamples(
            self.nominal + other.nominal, self.samples + other.samples
        )

    def __sub__(self, other: "ValueWithSamples") -> "ValueWithSamples":
        return ValueWithSamples(
            self.nominal - other.nominal, self.samples - other.samples
        )

    def __mul__(self, other: "ValueWithSamples"):
        return ValueWithSamples(
            self.nominal * other.nominal, np.multiply(self.samples, other.samples)
        )

    def __truediv__(self, other: "ValueWithSamples") -> "ValueWithSamples":
        return ValueWithSamples(
            self.nominal / other.nominal, np.divide(self.samples, other.samples)
        )
