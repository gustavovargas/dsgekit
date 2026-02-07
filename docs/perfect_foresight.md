# Perfect Foresight Solver

`dsgekit` includes a deterministic perfect-foresight solver for non-linear models:

- `solve_perfect_foresight(...)`

It solves a full transition path with known shocks and boundary conditions:

- fixed initial state (`y_0`)
- fixed terminal state (`y_{T+1}`)
- deterministic shocks `u_1..u_T`

When explicit inputs are omitted, defaults come from parsed model blocks:

- `initial_state` defaults to `initval` with `histval(var,0)` overrides
- `terminal_state` defaults to `initval` with `endval` overrides
- `shocks` defaults to deterministic entries declared in `shocks` via
  `periods ...; values ...;`

## Example

```python
import numpy as np

from dsgekit import load_model
from dsgekit.solvers import solve_perfect_foresight

model, cal, ss = load_model("tests/fixtures/models/ar1.yaml")

n_periods = 12
shock_path = np.zeros(n_periods)
shock_path[0] = 0.05

pf = solve_perfect_foresight(
    model,
    ss,
    cal,
    n_periods=n_periods,
    shocks={"e": shock_path},
    initial_state={"y": 0.0},
    terminal_state={"y": 0.0},
    tol=1e-10,
    max_iter=40,
)

print(pf.summary())
print(pf.path.head())
print(pf.residuals.abs().max())
```

## Anticipated vs Unanticipated (News Shocks)

You can define shock events explicitly and build a deterministic path:

```python
from dsgekit.solvers import (
    anticipated_shock,
    build_news_shock_path,
    solve_perfect_foresight,
    unanticipated_shock,
)

events = [
    unanticipated_shock("e", period=1, value=0.05),
    anticipated_shock("e", announcement_period=1, horizon=3, value=0.10),
]

shock_path = build_news_shock_path(
    n_periods=12,
    shock_names=model.shock_names,
    events=events,
)

pf = solve_perfect_foresight(
    model,
    ss,
    cal,
    n_periods=12,
    shocks=shock_path,
)
```

Semantics:

- unanticipated shock: `announcement_period == impact_period`
- anticipated/news shock: `impact_period = announcement_period + horizon`

## Notes

- Current scope supports models with timings up to one lag and one lead.
- The algorithm uses Newton iterations with backtracking line search.
- Linear solve mode supports `linear_solver="auto"|"dense"|"sparse"`.
  Auto switches to sparse block-tridiagonal assembly when
  `n_periods * n_variables >= sparse_threshold` (default `200`).
- Optional JIT backend supports `jit_backend="none"|"numba"` for dense block
  assembly (`numba` requires `pip install dsgekit[speed]`).
- If convergence fails, the solver raises `SolverError` by default (`raise_on_fail=False` returns a non-converged result instead).
