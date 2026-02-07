# Second-Order Perturbation (Pruning)

`dsgekit` includes a second-order perturbation solver with pruning-oriented simulation:

- `solve_second_order(...)`
- `simulate_pruned_second_order(...)`
- `simulate_pruned_second_order_path(...)`
- `girf_pruned_second_order(...)`

## Scope (current)

- Supports non-linear models with timings up to one lag and no leads (`max_lead=0`).
- Uses the derivative stack Hessian + implicit-function expansion around steady state.
- Returns a second-order policy tensor over `z_t = [y_{t-1}, u_t]`.

## API

```python
import numpy as np

from dsgekit import load_model
from dsgekit.simulate import simulate_pruned_second_order
from dsgekit.solvers import solve_second_order

model, cal, ss = load_model("tests/fixtures/models/ar1.yaml")
so = solve_second_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)

sim = simulate_pruned_second_order(
    so,
    cal,
    n_periods=200,
    seed=42,
    initial_state=np.zeros(model.n_variables),
)

print(so.summary())
print(sim.data.head())
```

## State-dependent trajectories

You can also simulate from an explicit shock path (instead of drawing shocks):

```python
import numpy as np

from dsgekit.simulate import simulate_pruned_second_order_path

shock_path = np.zeros((8, len(so.shock_names)))
shock_path[0, 0] = 0.1
shock_path[3, 0] = -0.05

path = simulate_pruned_second_order_path(
    so,
    shock_path,
    initial_state=np.array([0.2]),
)
print(path.data)
```

## Generalized IRFs (GIRFs)

For non-linear models, responses depend on the current state. `girf_pruned_second_order`
computes Monte Carlo generalized IRFs by comparing shocked vs baseline paths with
shared future innovations:

```python
import numpy as np

from dsgekit.simulate import girf_pruned_second_order

girf0 = girf_pruned_second_order(
    so,
    cal,
    "e",
    periods=12,
    shock_size=0.1,
    n_draws=1000,
    seed=123,
    initial_state=np.array([0.0]),
)
girf1 = girf_pruned_second_order(
    so,
    cal,
    "e",
    periods=12,
    shock_size=0.1,
    n_draws=1000,
    seed=123,
    initial_state=np.array([1.0]),
)

print(girf0.data.head())
print(girf1.data.head())
```

## Pruning recursion

Simulation follows:

- `x1_t = T x1_{t-1} + R u_t`
- `x2_t = T x2_{t-1} + 0.5 * G2([x1_{t-1}, u_t], [x1_{t-1}, u_t])`
- `y_t = x1_t + x2_t`

This avoids feeding second-order states into the non-linear term and helps
stability in long simulations.
