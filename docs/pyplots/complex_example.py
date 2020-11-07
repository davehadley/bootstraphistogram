import matplotlib.pyplot as plt
import numpy as np

import bootstraphistogram

# create histogram
hist = bootstraphistogram.BootstrapHistogram(
    bootstraphistogram.axis.Regular(10, -3.0, 3.0), numsamples=25
)

# fill with some random normal data
data = np.random.normal(size=1000)
hist.fill(data)

# plot the samples
bootstraphistogram.plot.fill_between(
    hist, percentiles=(5.0, 95.0), color="blue", alpha=0.25
)
bootstraphistogram.plot.step(hist, percentile=50.0, ls="--")
bootstraphistogram.plot.scatter(hist, color="red", alpha=0.25)
bootstraphistogram.plot.errorbar(hist, color="black", ls="")
plt.show()
