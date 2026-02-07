# First-Order Perturbation Solver

`dsgekit` provides a direct first-order stochastic solver for non-linear models:

- `linearize_first_order(...)`: build first-order matrices from the derivative stack
- `solve_first_order(...)`: compute policy matrices `T`, `R` with BK/QZ checks

## API

```python
from dsgekit import load_model
from dsgekit.solvers import linearize_first_order, solve_first_order

model, cal, ss = load_model("tests/fixtures/models/rbc.mod")

approx = linearize_first_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)
solution = solve_first_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)

print(approx.linear_system.summary())
print(solution.summary())
```

## Notes

- `derivative_backend` supports:
  - `numeric` (finite differences; default)
  - `sympy` (symbolic; requires `pip install dsgekit[sym]`)
- The current implementation supports models with max one lag and one lead.
- For equivalent systems, `solve_first_order(...)` is regression-tested against
  the existing `linearize(...) + solve_linear(...)` pipeline.
