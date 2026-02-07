# Linear Baseline Compatibility Harness

`dsgekit` includes a baseline-driven harness for linear compatibility checks:

- Solution matrices (`T`, `R`)
- BK counts (`n_stable`, `n_unstable`, `n_predetermined`)
- IRFs by shock
- Batch regression suite with Markdown/JSON dashboard outputs

Legal and attribution note: see `docs/legal_and_attribution.md`.

Use:

```python
from dsgekit.diagnostics import run_linear_compat

report = run_linear_compat(
    baseline_source="tests/fixtures/baselines/ar1_linear_baseline.json",
)
print(report.summary())
```

You can also pass a model explicitly:

```python
report = run_linear_compat(
    baseline_source="tests/fixtures/baselines/ar1_linear_baseline.json",
    model_source="tests/fixtures/models/ar1.mod",
)
```

## Regression Suite (Batch)

Run multiple baselines in one shot and get an aggregate report:

```python
from dsgekit.diagnostics import (
    expand_baseline_sources,
    run_linear_regression_suite,
)

baseline_sources = expand_baseline_sources(["tests/fixtures/baselines"])
suite = run_linear_regression_suite(baseline_sources=baseline_sources)

print(suite.summary())
print(suite.dashboard_markdown())
```

## CLI (CI-friendly)

```bash
dsgekit baseline_regression \
  --baselines tests/fixtures/baselines \
  --dashboard baseline_regression_dashboard.md \
  --json-output baseline_regression_report.json
```

Exit code is `0` when all baselines pass, `1` otherwise.

## Baseline Schema

Minimal JSON fields:

- `name`: baseline label
- `model_path`: optional model path (used when `model_source` is omitted)
- `periods`: IRF horizon
- `tolerances`:
  - `matrix_atol`, `matrix_rtol`
  - `irf_atol`, `irf_rtol`
- `solution`:
  - `var_names`, `shock_names`
  - `n_stable`, `n_unstable`, `n_predetermined`
  - `T`, `R` (2D arrays)
- `irfs`:
  - mapping `shock_name -> {"columns": [...], "data": [[...], ...]}`

See `tests/fixtures/baselines/ar1_linear_baseline.json` as reference.
