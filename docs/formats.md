# Model Formats

`dsgekit` supports three model input formats:

- `.mod` subset
- YAML (`.yaml` / `.yml`)
- Python `dict` (same schema as YAML)

All formats are loaded through:

```python
from dsgekit import load_model
model, cal, ss = load_model("my_model.yaml")
```

## `.mod` Format (`.mod` subset)

This subset is intended to ease migration from Dynare-style `.mod` workflows.

### Supported blocks

- `var ...;`
- `varexo ...;`
- `parameters ...;`
- `model; ... end;`
- `initval; ... end;`
- `shocks; ... end;`
- `varobs ...;`
- `estimated_params; ... end;`
- `histval; ... end;`
- `endval; ... end;`
- `steady_state_model; ... end;`

### Example

```mod
// AR(1) example
var y;
varexo e;
parameters rho;

rho = 0.9;

model;
    [ar1] y = rho * y(-1) + e;
end;

initval;
    y = 0;
end;

shocks;
    var e; stderr 0.01;
end;
```

### Notes

- Equation names are optional (`[name]`).
- Timing syntax uses `.mod` notation (`x(-1)`, `x(+1)`).
- Power operator `^` is supported in numeric expressions and equations.
- In `shocks`, both forms are supported:
  - `var e; stderr 0.01;`
  - `var e = 0.0001;` (interpreted as variance, converted to `stderr = sqrt(variance)`)
- Deterministic shock paths are also supported in `shocks`:
  - `var e; periods 1:4; values 0.05;` (broadcast one value across periods)
  - `var e; periods 1 3; values 0.1 -0.2;` (one value per period)

### Not supported

- Advanced options outside the supported subset

Unsupported features raise `UnsupportedFormatFeatureError`.

## YAML Format

### Example

```yaml
name: AR1

variables:
  - y: Output

shocks:
  - e: Technology shock

parameters:
  rho: 0.9

equations:
  - name: ar1
    expr: y = rho * y(-1) + e

steady_state:
  y: 0

shocks_config:
  e:
    stderr: 0.01
```

### Accepted variants

- `variables` and `shocks` can be either:
  - list of strings (`["y", "c"]`)
  - list of maps with descriptions (`- y: Output`)
- `equations` can be:
  - list of strings
  - list of `{name, expr}` objects
- `steady_state` or `initval` are both accepted

## Python `dict` Format

The `dict` format uses the same schema as YAML:

```python
source = {
    "name": "AR1",
    "variables": ["y"],
    "shocks": ["e"],
    "parameters": {"rho": 0.9},
    "equations": [{"name": "ar1", "expr": "y = rho * y(-1) + e"}],
    "steady_state": {"y": 0.0},
    "shocks_config": {"e": {"stderr": 0.01}},
}

model, cal, ss = load_model(source)
```

## Format Selection

Auto-detection by extension:

- `.mod` -> `.mod` parser
- `.yaml` / `.yml` -> YAML parser

You can override explicitly:

```python
model, cal, ss = load_model("model.mod", format="mod")
model, cal, ss = load_model("model.yaml", format="yaml")
```

For `dict`, `format` must be omitted or set to `"dict"`.
