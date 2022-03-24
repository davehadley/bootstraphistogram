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
```bash
source setup.sh
```

Alternatively, a `Dockerfile` is provided for a consistent development environment.
```bash
docker build -tbootstraphistogram:latest . && \
docker start bootstraphistogram && \
docker run --name bootstraphistogram -it -d bootstraphistogram:latest /bin/bash
```

This package uses [Python poetry](https://python-poetry.org/) for dependency management.
```bash
poetry install
```

To run the unit tests run:
```bash
poetry run pytest
```

To build documentation run:
```bash
poetry run pip install -r docs/requirements.txt && \
poetry run sphinx-build -W docs docs-build
```

To auto-build the documentation while editing:
```
poetry run pip install sphinx-autobuild && sphinx-autobuild docs docs/_build/html 
```
and find your documentation on <http://localhost:8000>.

To generate a test coverage report run:
```bash
poetry run coverage run -m pytest tests && poetry run coverage report -m
```