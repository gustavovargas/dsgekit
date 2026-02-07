# Third-Order Perturbation (Pruning)

`dsgekit` includes a third-order perturbation solver with pruning-oriented simulation:

- `solve_third_order(...)`
- `simulate_pruned_third_order(...)`
- `simulate_pruned_third_order_path(...)`

## Scope (current)

- Supports non-linear models with timings up to one lag and no leads (`max_lead=0`).
- Uses Jacobian/Hessian/third-order derivatives from the derivative stack and an implicit-function expansion around steady state.
- Returns second-order and third-order policy tensors over `z_t = [y_{t-1}, u_t]`.

## API

```python
import numpy as np

from dsgekit import load_model
from dsgekit.simulate import simulate_pruned_third_order
from dsgekit.solvers import solve_third_order

model, cal, ss = load_model("tests/fixtures/models/ar1.yaml")
to = solve_third_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)

sim = simulate_pruned_third_order(
    to,
    cal,
    n_periods=200,
    seed=42,
    initial_state=np.zeros(model.n_variables),
)

print(to.summary())
print(sim.data.head())
```

## Shock-path simulation

```python
import numpy as np

from dsgekit.simulate import simulate_pruned_third_order_path

shock_path = np.zeros((8, len(to.shock_names)))
shock_path[0, 0] = 0.1
shock_path[3, 0] = -0.05

path = simulate_pruned_third_order_path(
    to,
    shock_path,
    initial_state=np.array([0.2]),
)
print(path.data)
```

## Pruning recursion

Simulation follows:

- `x1_t = T x1_{t-1} + R u_t`
- `x2_t = T x2_{t-1} + 0.5 * G2([x1_{t-1}, u_t], [x1_{t-1}, u_t])`
- `x3_t = T x3_{t-1} + G2([x1_{t-1}, u_t], [x2_{t-1}, 0]) + (1/6) * G3([x1_{t-1}, u_t], [x1_{t-1}, u_t], [x1_{t-1}, u_t])`
- `y_t = x1_t + x2_t + x3_t`

This decomposition keeps higher-order terms out of lower-order recursions and improves numerical stability in long runs.
