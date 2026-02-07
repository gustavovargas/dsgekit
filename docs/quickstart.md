# Quick Start

This guide walks through the full baseline pipeline in `dsgekit`:

1. Load a model (`.mod`, `.yaml`, or `dict`)
2. Linearize around steady state
3. Solve the linear rational expectations system
4. Run IRFs/simulation/moments
5. Convert to state-space and run Kalman filter

## Requirements

- Python `>= 3.13`
- `numpy`, `scipy`, `pandas` (installed automatically with `dsgekit`)

## Installation

```bash
pip install dsgekit
```

For plotting:

```bash
pip install dsgekit[plot]
```

## Minimal Pipeline

```python
from dsgekit import load_model
from dsgekit.filters import kalman_filter
from dsgekit.simulate import irf, moments, simulate
from dsgekit.solvers import solve_linear
from dsgekit.transforms import linearize, to_state_space

# 1) Load
model, cal, ss = load_model("model.yaml")

# 2) Linearize
lin = linearize(model, ss, cal)

# 3) Solve (Blanchard-Kahn checked by default)
solution = solve_linear(lin)

# 4) Analysis
irfs = irf(solution, "e", periods=20)
sim = simulate(solution, cal, n_periods=200, seed=42)
m = moments(solution, cal, max_lag=8)

print(irfs.data.head())
print(sim.data.head())
print(m.summary())

# 5) State-space + filtering
ss_model = to_state_space(solution, cal, observables=["y"])
kf = kalman_filter(ss_model, sim.data[["y"]])
print(kf.summary())
```

## Deterministic Perfect Foresight

```python
import numpy as np

from dsgekit.solvers import solve_perfect_foresight

shock_path = np.zeros(12)
shock_path[0] = 0.05

pf = solve_perfect_foresight(
    model,
    ss,
    cal,
    n_periods=12,
    shocks={"e": shock_path},
    initial_state={"y": 0.0},
    terminal_state={"y": 0.0},
)

print(pf.summary())
print(pf.path.head())
```

## Forecast and Historical Decomposition

```python
from dsgekit.filters import (
    forecast,
    historical_decomposition,
    kalman_filter,
    kalman_smoother,
)

ss_model = to_state_space(solution, cal, observables=["y"])
kf = kalman_filter(ss_model, sim.data[["y"]])
sm = kalman_smoother(ss_model, kf)

fc = forecast(ss_model, kf, steps=12, smoother_result=sm)
print(fc.out_of_sample_observables.head())

hd = historical_decomposition(ss_model, sm)
print(hd.summary())
```

## First-Order Solver for Non-Linear Models

If you want a direct first-order perturbation path (via the derivative stack),
use `solve_first_order`:

```python
from dsgekit import load_model
from dsgekit.solvers import solve_first_order

model, cal, ss = load_model("tests/fixtures/models/rbc.mod")
solution = solve_first_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)
print(solution.summary())
```

## MAP Estimation with Priors

```python
from dsgekit.estimation import estimate_map
from dsgekit.model.calibration import EstimatedParam

cal.estimated_params = [
    EstimatedParam.from_dict(
        {
            "type": "param",
            "name": "rho",
            "init": 0.5,
            "lower": 0.01,
            "upper": 0.99,
            "prior": {"distribution": "normal", "mean": 0.8, "std": 0.1},
        }
    )
]

map_result = estimate_map(
    model,
    ss,
    cal,
    sim.data[["y"]],
    observables=["y"],
    param_names=None,  # inferred from estimated_params
    bounds=None,       # inferred from estimated_params
)
print(map_result.summary())
```

## MCMC Sampling

```python
from dsgekit.estimation import estimate_mcmc

mcmc = estimate_mcmc(
    model,
    ss,
    cal,
    sim.data[["y"]],
    observables=["y"],
    param_names=None,  # inferred from estimated_params
    n_draws=2000,
    burn_in=500,
    thin=5,
    proposal_scale=0.04,
    seed=42,
)
print(mcmc.summary())
diag = mcmc.diagnostics()
print(diag.summary())
print(mcmc.trace_dict()["rho"][:10])  # ready for trace plots
```

## Marginal Data Density

```python
from dsgekit.estimation import estimate_mdd_harmonic_mean, estimate_mdd_laplace

laplace = estimate_mdd_laplace(
    model,
    ss,
    cal,
    sim.data[["y"]],
    observables=["y"],
    param_names=None,
)

harmonic = estimate_mdd_harmonic_mean(
    model,
    ss,
    cal,
    sim.data[["y"]],
    observables=["y"],
    param_names=None,
    mcmc_result=mcmc,
    max_samples=500,
    seed=7,
)

print(laplace.log_mdd, harmonic.log_mdd)
```

## If BK Conditions Fail

If `solve_linear()` raises `IndeterminacyError` or `NoStableSolutionError`, you can still inspect diagnostics:

```python
from dsgekit.solvers import diagnose_bk, solve_linear

solution = solve_linear(lin, check_bk=False)
diag = diagnose_bk(solution)
print(diag.summary())
print(diag.trace())  # Canonical reproducible trace (includes trace_id)
```

## CLI Quick Start

```bash
dsgekit info model.yaml
dsgekit solve model.mod
dsgekit irf model.yaml -s e
dsgekit simulate model.mod
dsgekit estimation model.mod --method mle --params rho
dsgekit forecast model.yaml -p 12
dsgekit decompose model.yaml --train-periods 120
dsgekit run model.yaml
```

## Verify Local Setup

Run this from the repository root:

```bash
python -c "
from dsgekit import load_model
from dsgekit.transforms import linearize
from dsgekit.solvers import solve_linear
from dsgekit.simulate import irf
model, cal, ss = load_model('tests/fixtures/models/ar1.yaml')
solution = solve_linear(linearize(model, ss, cal))
print(irf(solution, 'e', periods=5).data)
print('OK')
"
```

For `.mod` migration notes and tutorials, see `docs/mod_workflow_migration.md`.
