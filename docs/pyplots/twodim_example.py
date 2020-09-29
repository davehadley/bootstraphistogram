import numpy as np
import matplotlib.pyplot as plt

from bootstraphistogram import BootstrapHistogram, axis
import bootstraphistogram.plot


def example2d():
    # create a 2D histogram
    hist2d = BootstrapHistogram(
        axis.Regular(20, -5.0, 5.0), axis.Regular(20, -5.0, 5.0), numsamples=100
    )

    # fill with some random, correlated data
    mu = [0.0, 1.0]
    cov = [[1.0, 0.5], [0.5, 2.0]]
    rawX, rawY = np.random.multivariate_normal(mu, cov, size=1000).T
    hist2d.fill(rawX, rawY)

    # plot the 2D distribution
    figure = plt.figure()
    ax = figure.add_subplot(221)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.imshow(hist2d.nominal.view(), origin="lower", extent=[-5.0, 5.0, -5.0, 5.0])

    # plot the 1D projection
    histX = hist2d.project(0)
    histY = hist2d.project(1)
    ax = figure.add_subplot(222)
    ax.set_xlabel("X")
    bootstraphistogram.plot.errorbar(histX, ax=ax, ls="")
    ax = figure.add_subplot(223)
    ax.set_xlabel("Y")
    bootstraphistogram.plot.errorbar(histY, ax=ax, ls="")

    # plot the sum
    histRatio = histX / histY
    ax = figure.add_subplot(224)
    ax.set_xlabel("X/Y")
    bootstraphistogram.plot.errorbar(histRatio, ax=ax, ls="")

    plt.show()

    return


if __name__ == "__main__":
    example2d()
