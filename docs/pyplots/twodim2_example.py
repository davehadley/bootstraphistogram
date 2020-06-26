import numpy as np
import matplotlib.pyplot as plt

from bootstraphistogram import BootstrapHistogram, axis

@np.vectorize
def stddev_bootstrap(nevents=100, numbins=2, numsamples=100):
    # create a 2D histogram
    hist2d = BootstrapHistogram(axis.Regular(numbins, 0.0, 1.0), axis.Regular(numbins, 0.0, 1.0), numsamples=numsamples)
    # fill with some random data
    hist2d.fill(np.random.uniform(size=nevents), np.random.uniform(size=nevents))
    histX = hist2d.project(0)
    histY = hist2d.project(1)
    histSum = histX + histY
    return histSum.std()[0]

@np.vectorize
def stddev_analytic(nevents=100, numbins=2):
    sigma = np.sqrt(nevents / numbins)
    correlation = 1.0 / numbins
    cov = correlation * sigma ** 2
    return np.sqrt(2.0 * sigma ** 2 + 2.0 * cov)

def example2d():
    N = list(range(10, 1000, 10))
    plt.plot(N, stddev_analytic(N), label=r"$\sigma$ analytic")
    plt.plot(N, stddev_bootstrap(N), label=r"$\sigma$ bootstrap")
    plt.xlabel("N")
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    example2d()
