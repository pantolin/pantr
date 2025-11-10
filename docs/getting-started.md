# Getting Started

## Installation

Requires Python 3.10–3.12.

```bash
pip install pantr
```

For development:

```bash
pip install -e ".[dev,docs]"
pre-commit install
```

## Quick Example

```python
from __future__ import annotations

import numpy as np

import pantr

print(pantr.__version__)
```

## Building the Documentation

```bash
pip install -e ".[docs]"
cd docs
make html
```
