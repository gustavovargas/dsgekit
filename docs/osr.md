# OSR (Rule Sweeps)

`dsgekit` includes an OSR-style grid-search optimizer for parameterized policy rules:

- API: `osr_grid_search(...)`
- CLI: `dsgekit osr ...`

Current objective:

- weighted sum of unconditional variances:
  - `loss = sum_i w_i * Var(x_i)`

## Python API

```python
from dsgekit import load_model
from dsgekit.solvers import osr_grid_search

model, cal, ss = load_model("tests/fixtures/models/ar1.yaml")

result = osr_grid_search(
    model,
    ss,
    cal,
    parameter_grid={
        "rho": [0.0, 0.5, 0.9],
    },
    loss_weights={
        "y": 1.0,
    },
    require_determinate=True,
)

print(result.summary())
print(result.to_frame(include_failed=False).head())
print("Best:", result.best.parameters if result.best else None)
```

## CLI

```bash
dsgekit osr tests/fixtures/models/ar1.yaml \
  --grid rho=0.0,0.5,0.9 \
  --loss y=1.0 \
  --top 5 \
  --output osr_candidates.csv
```

Grid syntax:

- explicit values: `name=v1,v2,v3`
- linear range: `name=start:stop:num` (inclusive, `num >= 2`)

Examples:

- `--grid phi_pi=1.0:3.0:9`
- `--grid phi_y=0.0,0.125,0.25`

## Notes

- By default, non-determinate candidates are rejected.
- Use `--allow-indeterminate` to keep them in the candidate set.
- Use `--include-failed` to include rejected/error candidates in output tables.
