# Derivative Stack

`dsgekit` provides a unified derivative API for model residual equations:

- Jacobian
- Hessian
- Third-order derivative tensor

via `DerivativeStack` in `dsgekit.derivatives`.

## Backends

- `numeric` (default): central finite differences
- `sympy` (optional): symbolic derivatives (requires `dsgekit[sym]`)

## Example

```python
from dsgekit import load_model
from dsgekit.derivatives import DerivativeStack, shock_coord, var_coord
from dsgekit.model.equations import EvalContext

model, cal, ss = load_model("tests/fixtures/models/ar1.mod")

ctx = EvalContext(parameters=cal.parameters.copy())
ctx.set_variable("y", -1, ss.values["y"])
ctx.set_variable("y", 0, ss.values["y"])
ctx.set_variable("y", 1, ss.values["y"])
ctx.set_shock("e", 0.0)

coords = [var_coord("y", -1), var_coord("y", 0), shock_coord("e")]
stack = DerivativeStack("numeric")

J = stack.jacobian(model, ctx, coords)
H = stack.hessian(model, ctx, coords)
T3 = stack.third_order(model, ctx, coords)
```
