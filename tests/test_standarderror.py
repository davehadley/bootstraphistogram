import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore


def _standard_error_mean(size: int, sigma: float = 1.0) -> float:
    return float(sigma / np.sqrt(size))


def _mc_error_mean(size: int, sigma: float = 1.0, nmc: int = 1000) -> float:
    return float(
        np.std(
            [
                np.average(np.random.normal(size=size, scale=sigma)) - 0.0
                for _ in range(nmc)
            ]
        )
    )


def _standard_error_std(size: int, sigma: float = 1.0) -> float:
    return float(np.sqrt(sigma**2 / (2.0 * size)))


def _mc_error_std(size: int, sigma: float = 1.0, nmc: int = 1000) -> float:
    return float(
        np.std(
            [
                np.std(np.random.normal(size=size, scale=sigma)) - sigma
                for _ in range(nmc)
            ]
        )
    )


def test_standarderror() -> None:
    fig = plt.figure()
    ax = fig.add_subplot(211)
    X = np.arange(10, 100, 10)
    sigma = 2.0
    # check mean
    Y = [_mc_error_mean(size=x, sigma=sigma) for x in X]
    ax.plot(X, Y, label="MC")
    Y = [_standard_error_mean(size=x, sigma=sigma) for x in X]
    ax.plot(X, Y, label="Analytic")
    ax.legend()
    # check std dev
    ax = fig.add_subplot(212)
    Y = [_mc_error_std(size=x, sigma=sigma) for x in X]
    ax.plot(X, Y, label="MC")
    Y = [_standard_error_std(size=x, sigma=sigma) for x in X]
    ax.plot(X, Y, label="Analytic")
    ax.legend()
    plt.show()
    return
