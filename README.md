# bootstraphistogram 

[![Github Build Status](https://img.shields.io/github/workflow/status/davehadley/bootstraphistogram/ci?label=Github%20Build)](https://github.com/davehadley/bootstraphistogram/actions?query=workflow%3Aci)
[![Documentation Status](https://readthedocs.org/projects/bootstraphistogram/badge/?version=latest)](https://bootstraphistogram.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/bootstraphistogram)](https://pypi.org/project/bootstraphistogram/)
[![License: MIT](https://img.shields.io/pypi/l/bootstraphistogram)](https://github.com/davehadley/bootstraphistogram/blob/master/LICENSE.txt)
[![Last Commit](https://img.shields.io/github/last-commit/davehadley/bootstraphistogram/dev)](https://github.com/davehadley/bootstraphistogram)

A python package that provides a multi-dimensional histogram with automatic Poisson bootstrap re-sampling.

# Installation

Install with pip from PyPI:
```bash
python -m pip install bootstraphistogram
```

# Usage Instructions

For usage instructions and examples see the documentation at: <https://bootstraphistogram.readthedocs.io>.

# Development Instructions

For Linux systems, the provided setup script will setup a suitable python virtual environment 
and install pre-commit-hooks.
```
source setup.sh
```

This package uses [Python poetry](https://python-poetry.org/) for dependency management.
```
poetry install
```

To run the unit tests run:
```
poetry run pytest tests
```

