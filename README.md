# Optimi-Lab

An extensible optimization toolbox for prototyping evolutionary and surrogate-based
optimization workflows.

## Overview

Optimi-Lab provides implementations and utilities for evolutionary algorithms, surrogate models,
and sensitivity analysis. It is intended for researchers and practitioners who need a
lightweight framework for experiments and benchmarking.

## Key Features

- Multi-objective and single-objective optimization algorithms
- Surrogate modeling utilities and model selection helpers
- Built-in logging and experiment output management
- Examples and tests for reproducibility


## Installation

### Recommended: install from pypi
```shell
uv pip install optimi-lab # use uv
```
or
```
pip install optimi-lab
```

### Install from source (development mode).

1. Clone the repository

```bash
git clone https://github.com/DawnEver/optimi-lab.git
cd optimi-lab
```

2. Create and activate a virtual environment, uv is recommanded:
```bash
uv venv
# or use venv
pip -m venv .venv
```

3. Install the package
```bash
uv pip install -e . # for normal user
uv pip install -e ".[dev]" # for developer
```
## Usage

Typical steps:

1. Define an objective function.
2. Configure an optimizer (algorithm, population size, termination criteria).
3. Run the optimizer and inspect outputs in the `output/` directory.

See `examples/` for concrete usage examples.


## Contributing

Contributions are welcome — please open an issue to discuss large changes before submitting
a pull request. Add tests for new features and follow the existing code style.

## License

This project is licensed under the Apache License 2.0 — see the `LICENSE` file for details.

## Contact

For questions or collaboration, open an issue or contact the maintainers through the repository.
