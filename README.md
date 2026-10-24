# Optimi-Lab

Multi-objective optimization and surrogate models, as pure numerics.

## Overview

Optimi-Lab is a library: it optimizes over declared variables, objectives and constraints, and
returns arrays. It reads no file, writes no file, configures no logger and depends on numpy,
scipy, numerics-lab and `lab_commons.outcomes` only, so it can sit inside another program without
reconfiguring it. Every surrogate model is self-implemented (`optimi_lab.model`).

## Key Features

- Layered (plan II.7): `problem` -> `oracle` -> `archive` -> `metric` -> `solver` -> `run`
- Variables: `Real` (linear or log), `Integer`, `Categorical` (metric only when embedded), `Field`,
  with conditional `active_if`; objectives with a fixed normalisation; five constraint kinds
- Every result is a `lab_commons.outcomes.Cell`: a status (DECIDED or RETRYABLE), values only when
  it computed, a reason otherwise -- a licence miss is never an infeasible design
- One append-only record per run; an interrupted run resumes by replaying it
- Population solvers NSGA-II, NSGA-III, MODE, MOEA/D, MOPSO; IGD and exact hypervolume
- Surrogate regressors and a weighted mixture of them

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
uv sync --frozen --group dev # for developer
```
## Configuration

optimi-lab reads no configuration file: every setting is an argument. The old `usr/` config tree
is retired -- per-machine settings across the family live in the lab-commons machine file
(`lab_commons.config`), and a leftover `usr/<machine>/config.toml` is refused by the architecture
tests with that remedy.

## Usage

```python
from pathlib import Path

from lab_commons.outcomes import Record

from optimi_lab import (
    BlackBox,
    Budget,
    Capability,
    Objective,
    Problem,
    Real,
    SampleSpec,
    Sense,
    Space,
    from_function,
    run,
)
from optimi_lab.solver import nsga2

problem = Problem(
    Space((Real('x', 0.0, 1.0), Real('y', 0.0, 1.0))),
    (Objective('cost'), Objective('gain', Sense.MAXIMIZE)),
    constraints=(BlackBox('x_out', '<=', 0.8),),
)


def evaluate(action):
    x, y = action.design['x'], action.design['y']
    return {'cost': x + y, 'gain': x * y, 'x_out': x}


oracle = from_function(evaluate, Capability(outputs=('cost', 'gain', 'x_out')))
solver = nsga2(problem, oracle.capability, spec=SampleSpec(n_samples=20, seed=0))
outcome = run(problem, oracle, solver, Record(Path('run.jsonl')), Budget(evaluations=200))
front = outcome.archive.front()  # feasible, non-dominated, observed results
```

Run the same call again over the same `run.jsonl` with a larger budget and it resumes by replay.


## Contributing

Contributions are welcome — please open an issue to discuss large changes before submitting
a pull request. Add tests for new features and follow the existing code style.

## License

This project is licensed under the Apache License 2.0 — see the `LICENSE` file for details.

## Contact

For questions or collaboration, open an issue or contact the maintainers through the repository.
