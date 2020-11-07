import matplotlib.pyplot as plt
import numpy as np

import bootstraphistogram

# create histogram
hist = bootstraphistogram.BootstrapHistogram(
    bootstraphistogram.axis.Regular(10, -3.0, 3.0), numsamples=10
)

# fill with some random normal data
data = np.random.normal(size=1000)
hist.fill(data)

# plot the samples
bootstraphistogram.plot.scatter(hist)
plt.show()
