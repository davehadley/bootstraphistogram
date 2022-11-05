"""Implements `ValueWithSamples`."""
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    try:
        from typing import Any

        from numpy.typing import NDArray
    except (ImportError, ModuleNotFoundError):
        pass

T = TypeVar("T")


class ValueWithSamples(Generic[T]):
    """Container class storing a calculated value along with its bootstrap resamples.

    Parameters
    ----------
    nominal: T
        the value without any resampling.
    samples: "NDArray[Any]"
        the same value with Poisson bootstrap resampling applied.
    """

    def __init__(self, nominal: T, samples: "NDArray[Any]"):
        self._nominal = nominal
        self._samples = samples

    @property
    def nominal(self) -> T:
        """the value without any resampling."""
        return self._nominal

    @property
    def samples(self) -> "NDArray[Any]":
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

    def __add__(self, other: "ValueWithSamples[T]") -> "ValueWithSamples[T]":
        """
        Sum both values and their associated samples.

        Returns
        -------
        sum_ : ValueWithSamples
            A new instance of the summed values.
        """
        return ValueWithSamples(
            self.nominal + other.nominal,  # type: ignore
            self.samples + other.samples,
        )

    def __sub__(self, other: "ValueWithSamples[T]") -> "ValueWithSamples[T]":
        return ValueWithSamples(
            self.nominal - other.nominal, self.samples - other.samples  # type: ignore
        )

    def __mul__(self, other: "ValueWithSamples[T]") -> "ValueWithSamples[T]":
        return ValueWithSamples(
            self.nominal * other.nominal,  # type: ignore
            np.multiply(self.samples, other.samples),
        )

    def __truediv__(self, other: "ValueWithSamples[T]") -> "ValueWithSamples[T]":
        return ValueWithSamples(
            self.nominal / other.nominal,  # type: ignore
            np.divide(self.samples, other.samples),
        )
