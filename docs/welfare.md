# Welfare Metrics

`dsgekit` provides compact welfare evaluation utilities over simulated paths:

- `evaluate_welfare(...)`: expected utility and quadratic-loss aggregates for one scenario
- `compare_welfare(...)`: ranked comparison across scenarios

## Single-Scenario Welfare

```python
from dsgekit import load_model
from dsgekit.simulate import evaluate_welfare, simulate
from dsgekit.solvers import solve_linear
from dsgekit.transforms import linearize

model, cal, ss = load_model("tests/fixtures/models/ar1.yaml")
solution = solve_linear(linearize(model, ss, cal))
sim = simulate(solution, cal, n_periods=200, seed=42, burn_in=100)

result = evaluate_welfare(
    sim,
    variables=["y"],
    targets={"y": 0.0},
    quadratic_loss_weights={"y": 1.0},
    linear_utility_weights={"y": 0.0},
    discount=0.99,
    name="baseline_policy",
)

print(result.summary())
```

Per-period utility uses:

- `u_t = c + a' x_t - 0.5 * x_t' Q x_t`
- `L_t = x_t' Q x_t`

where `x_t` are deviations from `targets`.

## Scenario Comparison

```python
from dsgekit.simulate import compare_welfare

comparison = compare_welfare(
    {
        "baseline": baseline_result,
        "alternative": alternative_result,
    },
    baseline="baseline",
    metric="discounted_utility",  # or discounted_loss / mean_utility / mean_loss
)

print(comparison.summary())
```

Rules:

- utility metrics rank higher values as better
- loss metrics rank lower values as better

## Notes

- `loss_matrix` can be provided directly as a full quadratic matrix (`numpy` or `pandas`), or use diagonal `quadratic_loss_weights`.
- `data` can be a `pandas.DataFrame` or any object exposing `.data` as a DataFrame (e.g., `SimulationResult`).
- `discount` must be in `(0, 1]`.
