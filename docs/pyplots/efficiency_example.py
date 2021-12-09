import matplotlib.pyplot as plt
import numpy as np

import bootstraphistogram

# create histogram
hist = bootstraphistogram.BootstrapEfficiency(
    bootstraphistogram.axis.Regular(10, -3.0, 3.0), numsamples=100
)

# fill with some random normal data
xvalues = np.random.uniform(-3.0, 3.0, size=1000)
selected = np.random.normal(size=1000) < xvalues
hist.fill(selected, xvalues)

# plot the efficiency curve
bootstraphistogram.plot.fill_between(
    hist.efficiency, percentiles=(5.0, 95.0), color="blue", alpha=0.25
)
bootstraphistogram.plot.step(hist.efficiency, percentile=50.0, ls="--")
plt.show()
