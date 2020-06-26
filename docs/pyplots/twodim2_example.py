import numpy as np
import matplotlib.pyplot as plt

from bootstraphistogram import BootstrapHistogram, axis
import bootstraphistogram.plot


def example2d():
    # create a 2D histogram
    hist2d = BootstrapHistogram(axis.Regular(2, 0.0, 1.0), axis.Regular(2, 0.0, 1.0), numsamples=100)

    # fill with some random data
    N = 1000
    hist2d.fill(np.random.uniform(size=1000), np.random.uniform(size=N))

    # plot the 2D distribution
    figure = plt.figure()
    ax = figure.add_subplot(221)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.imshow(hist2d.nominal.view(), origin="lower", extent=[0.0, 1.0, 0.0, 1.0])

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
    histSum = histX + histY
    ax = figure.add_subplot(224)
    ax.set_xlabel("X+Y")
    bootstraphistogram.plot.errorbar(histSum, ax=ax, ls="")

    print("bootstrap error =", histSum.std())
    sigma = np.sqrt(N/2)
    correlation = 1.0/len(histSum.axes[0])
    cov = correlation * sigma**2
    print("expected error =", np.sqrt(2.0*sigma**2 + 2.0*cov))
    #print("expected error no correlation =", np.sqrt(2.0 * sigma ** 2))

    plt.show()

    return


if __name__ == '__main__':
    example2d()
