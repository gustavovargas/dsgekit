# OccBin-lite (One Constraint)

`dsgekit` includes an OccBin-lite deterministic solver for one occasionally
binding constraint:

- `solve_occbin_lite(...)`

It combines:

1. Newton perfect-foresight solve for a fixed regime sequence
2. Regime update from a simple constraint rule
3. Iteration to a fixed point

## API

```python
import numpy as np

from dsgekit import load_model
from dsgekit.solvers import solve_occbin_lite

model, cal, ss = load_model("tests/fixtures/models/zlb_toy.yaml")
n_periods = 16

shocks = np.zeros((n_periods, model.n_shocks))
e_i = model.shock_names.index("e_i")
shocks[:5, e_i] = [-0.22, -0.14, -0.09, -0.05, -0.02]

res = solve_occbin_lite(
    model,
    ss,
    cal,
    n_periods=n_periods,
    shocks=shocks,
    switch_equation="effective_rate_floor",
    relaxed_equation="i = i_shadow",
    binding_equation="i = 0.0",
    constraint_var="i_shadow",
    constraint_operator="<=",
    constraint_value=0.0,
    constraint_timing=0,
)

print(res.summary())
print(res.binding_regime)
print(res.path[["i_shadow", "i"]].head())
```

## Parameters specific to OccBin-lite

- `switch_equation`: equation name or index to switch across regimes.
- `relaxed_equation`: equation used when not binding (optional; defaults to model equation).
- `binding_equation`: equation used when binding.
- `constraint_var`: variable used to define the binding condition.
- `constraint_operator`: one of `"<"`, `"<="`, `">"`, `">="`.
- `constraint_value`: threshold for the constraint.
- `constraint_timing`: timing of `constraint_var` used in the rule (`-1`, `0`, `1`).
- `linear_solver`: `auto`, `dense`, or `sparse` for Newton linear systems.
- `jit_backend`: `none` or `numba` for dense Jacobian assembly
  (`numba` requires `pip install dsgekit[speed]`).
- `sparse_threshold`: in `auto`, switch to sparse when
  `n_periods * n_variables >= sparse_threshold` (default `200`).

## Output

`OccBinResult` contains:

- `path`: solved endogenous path.
- `binding_regime`: boolean regime sequence by period.
- `residuals`: final residual matrix for the solved piecewise system.
- convergence metadata (`n_regime_iterations`, `n_newton_iterations`, etc.).

## Scope

- Current implementation supports one occasionally binding restriction.
- Timing support matches the perfect-foresight solver (`max_lag <= 1`, `max_lead <= 1`).
- Multi-constraint / multi-regime generalized OccBin is out of scope for this stage.
