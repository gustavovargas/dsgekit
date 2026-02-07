# `.mod` Workflow Migration

This guide is for users moving existing `.mod` workflows to `dsgekit`.

`dsgekit` is an independent project (no affiliation or endorsement by third-party tool maintainers).
If you are migrating from Dynare, use this page as a practical map of equivalent workflow steps.

It focuses on:

- what maps directly
- what differs
- practical workarounds
- end-to-end migration tutorials

## Command Mapping

| Previous command style | `dsgekit` |
|---|---|
| simulation command | `dsgekit simulate model.mod` |
| `estimation(...)` | `dsgekit estimation model.mod --method mle|map|mcmc` |
| `forecast` | `dsgekit forecast model.mod -p <horizon>` |
| `model_diagnostics`/BK checks | `solve_linear(...)` + `diagnose_bk(...)` |

## `.mod` Block Compatibility

Supported now:

- `var`, `varexo`, `parameters`, `model`
- `initval`, `histval`, `endval`
- `shocks` (`stderr`, variance shorthand, deterministic `periods/values`)
- `varobs`, `estimated_params`, `steady_state_model`
- basic macros: `@#define`, `@#if/@#elseif/@#else/@#endif`, `@#include`

Not yet fully equivalent:

- full macro language and advanced options
- `extended_path`
- general OccBin multi-regime workflows (one-constraint `solve_occbin_lite(...)` is available)
- full command-level parity for all options

## Key Behavioral Differences

1. `perfect_foresight` horizon is explicit:

- You always pass `n_periods`.

1. Deterministic transition defaults:

- If not provided explicitly, `solve_perfect_foresight(...)` uses:
  - initial boundary from `initval` with `histval(var,0)` overrides
  - terminal boundary from `initval` with `endval` overrides
  - deterministic shocks from `shocks` block `periods/values`

1. Estimation defaults:

- `param_names=None` means infer from `estimated_params`.
- MAP/MCMC require priors by default; use `--allow-missing-priors` only when needed.

## Common Workarounds

### 1) Unsupported simulation options

Use Python API for full control:

```python
from dsgekit import load_model
from dsgekit.simulate import irf, moments, simulate
from dsgekit.solvers import solve_linear
from dsgekit.transforms import linearize

model, cal, ss = load_model("my_model.mod")
solution = solve_linear(linearize(model, ss, cal))

sim = simulate(solution, cal, n_periods=500, seed=42, burn_in=100)
irfs = irf(solution, "e", periods=40)
m = moments(solution, cal, max_lag=12)
```

### 2) Deterministic/news shocks

Either define inside `.mod`:

```mod
shocks;
    var e; stderr 0.0;
    var e;
        periods 1:4;
        values 0.05;
end;
```

Or build event schedules in Python with anticipated/unanticipated helpers.

## Tutorial 1: Port a Linear `.mod`

Goal: load, solve, inspect BK, run IRFs.

1. Keep your `.mod` file and load it directly.
1. Solve and inspect diagnostics:

```python
from dsgekit import load_model
from dsgekit.simulate import irf
from dsgekit.solvers import diagnose_bk, solve_linear
from dsgekit.transforms import linearize

model, cal, ss = load_model("tests/fixtures/models/nk.mod")
solution = solve_linear(linearize(model, ss, cal))
diag = diagnose_bk(solution)

print(diag.summary())
print(irf(solution, "e_d", periods=20).data.head())
```

1. Optional baseline check against your reference outputs:

```python
from dsgekit.diagnostics import run_linear_compat

report = run_linear_compat(
    baseline_source="tests/fixtures/baselines/ar1_linear_baseline.json",
)
print(report.summary())
```

## Tutorial 2: Deterministic Transition + News Shocks

Goal: replicate a deterministic transition workflow with anticipated and unanticipated shocks.

```python
from dsgekit import load_model
from dsgekit.solvers import (
    anticipated_shock,
    build_news_shock_path,
    solve_perfect_foresight,
    unanticipated_shock,
)

model, cal, ss = load_model("tests/fixtures/models/ar1_deterministic_transition.mod")

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

print(pf.summary())
print(pf.shocks.head())
print(pf.path.head())
```

## Migration Checklist

- Parser: `.mod` loads without manual edits.
- Determinacy: BK diagnostics are sensible.
- Transition: deterministic path reproduces expected impulse/transitions.
- Estimation: `estimated_params` entries map to intended MLE/MAP/MCMC setup.
- Regression: keep a reference baseline JSON for line-by-line matrix/IRF comparison.
