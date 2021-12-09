Introduction
============

:py:mod:`bootstraphistogram` provides a multi-dimensional histogram that implements
`Poisson bootstrap resampling <https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Poisson_bootstrap>`_ of the bin values.
This provides a computationally efficient numerical method for estimating the distribution of the statistic
represented by the bin values.

The main class is implemented in :py:class:`bootstraphistogram.BootstrapHistogram`.
Some basic plotting functions are provided in :py:mod:`bootstraphistogram.plot`.
:py:class:`bootstraphistogram.BootstrapEfficiency` handles the common case of 
computing the fraction of data points passing some selection criteria.
See :ref:`examples`.