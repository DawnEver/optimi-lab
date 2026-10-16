# Optimi-Lab

Multi-objective optimization and surrogate models, as pure numerics.

## Overview

Optimi-Lab is a library: it optimizes over declared variables, objectives and constraints, and
returns arrays. It reads no file, writes no file, configures no logger and depends on numpy,
scipy and scikit-learn only, so it can sit inside another program without reconfiguring it.

## Key Features

- Multi-objective algorithms behind one ask/tell loop: NSGA-II, NSGA-III, MODE, MOEA/D, MOPSO
- Constraints as declarations (`Constraint(name, '<=' | '>=', bound)`), ranked by Deb's rule in
  every algorithm; the Pareto front holds feasible points only
- Design matrices: full-factorial grid, Latin hypercube, Poisson disk or an imported matrix, with
  an optional nominal anchor point as row 0
- A per-cell outcome on every evaluation, so a point that did not compute is never a number
- Surrogate regressors and a weighted mixture of them
- Benchmarks with known answers (ZDT1, ZDT2, the constrained BNH) and the IGD and 2-D
  hypervolume metrics


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
## Configuration

optimi-lab reads no configuration file: every setting is an argument. The old `usr/` config tree
is retired -- per-machine settings across the family live in the lab-commons machine file
(`lab_commons.config`), and a leftover `usr/local/config.toml` is refused by the architecture
tests with that remedy.

## Usage

```python
import numpy as np

from optimi_lab import (
    Constraint, ConstraintSet, Evaluation, Objective, ObjectiveSet, SampleSpec, Variable, VariableSet, optimize,
)
from optimi_lab.intelligent_algorithm import nsga2

space = VariableSet([Variable('x', 0.0, 1.0), Variable('y', 0.0, 1.0)])
objectives = ObjectiveSet([Objective('cost'), Objective('gain', 'maximize')])
constraints = ConstraintSet([Constraint('x_cap', '<=', 0.8)])


def evaluate(points: np.ndarray) -> Evaluation:
    values = np.column_stack([points.sum(axis=1), points[:, 0] * points[:, 1]])
    return Evaluation(points, values, constraints=points[:, :1])


record = optimize(
    space=space,
    objectives=objectives,
    constraints=constraints,
    evaluate=evaluate,
    build_proposer=nsga2(space, seed=0),
    spec=SampleSpec(n_samples=20, seed=0),
    n_batches=10,
)
front = record.pareto()  # feasible, non-dominated points, in your own frame
```

A design of experiments is the same call with `n_batches=1` and the sample you want, e.g.
`SampleSpec(kind='grid', levels=(5, 5), anchor=(0.5, 0.5))`.


## Contributing

Contributions are welcome — please open an issue to discuss large changes before submitting
a pull request. Add tests for new features and follow the existing code style.

## License

This project is licensed under the Apache License 2.0 — see the `LICENSE` file for details.

## Contact

For questions or collaboration, open an issue or contact the maintainers through the repository.
