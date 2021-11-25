import matplotlib.pyplot as plt
import numpy as np


def _standard_error_mean(size, sigma=1.0):
    return sigma / np.sqrt(size)


def _mc_error_mean(size, sigma=1.0, nmc=1000):
    return np.std(
        [np.average(np.random.normal(size=size, scale=sigma)) - 0.0 for _ in range(nmc)]
    )


def _standard_error_std(size, sigma=1.0):
    return np.sqrt(sigma ** 2 / (2.0 * size))


def _mc_error_std(size, sigma=1.0, nmc=1000):
    return np.std(
        [np.std(np.random.normal(size=size, scale=sigma)) - sigma for _ in range(nmc)]
    )


def test_standarderror():
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


if __name__ == "__main__":
    test_standarderror()
