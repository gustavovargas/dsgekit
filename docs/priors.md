# Priors DSL and Estimated Parameters

`dsgekit` supports a priors DSL for estimable parameters (`estimated_params`) with:

- Distributions: `normal`, `beta`, `gamma`, `inv_gamma`
- Aliases: `*_pdf` variants are accepted (`normal_pdf`, `beta_pdf`, ...)
- Validation of distribution parameters (`mean`, `std`) and entry references

## Python API

```python
from dsgekit.io.formats.python_api import ModelBuilder
from dsgekit.model import PriorSpec

model, cal, ss = (
    ModelBuilder("ar1")
    .var("y")
    .varexo("e")
    .param("rho", 0.9)
    .equation("y = rho * y(-1) + e")
    .initval(y=0.0)
    .shock_stderr(e=0.01)
    .estimate_param(
        "rho",
        init=0.9,
        lower=0.0,
        upper=0.99,
        prior=PriorSpec.beta(mean=0.8, std=0.05),
    )
    .estimate_stderr(
        "e",
        init=0.01,
        lower=0.001,
        upper=0.1,
        prior=PriorSpec.inv_gamma(mean=0.02, std=0.01),
    )
    .build()
)
```

## `.mod` estimated_params

```mod
estimated_params;
    rho, 0.9, 0.0, 0.99, beta_pdf, 0.8, 0.05;
    stderr e, 0.01, 0.001, 0.1, inv_gamma_pdf, 0.02, 0.01;
end;
```

## YAML / dict format

```yaml
estimated_params:
  - type: param
    name: rho
    init: 0.9
    lower: 0.0
    upper: 0.99
    prior:
      distribution: beta
      mean: 0.8
      std: 0.05
  - type: stderr
    name: e
    init: 0.01
    lower: 0.001
    upper: 0.1
    prior: inv_gamma
    prior_mean: 0.02
    prior_std: 0.01
```

## Serialization

- `Calibration.to_dict()` emits both:
  - Legacy fields (`prior_shape`, `prior_mean`, `prior_std`)
  - Nested prior object (`prior: {distribution, mean, std}`)
- `Calibration.from_dict()` accepts both formats.

## MAP Estimation

Use priors directly in posterior mode optimization:

```python
from dsgekit.estimation import estimate_map

result = estimate_map(
    model,
    ss,
    cal,                # includes estimated_params with priors
    data,
    observables=["y"],
    param_names=None,   # infer from calibration.estimated_params
    bounds=None,        # infer bounds from calibration.estimated_params
)

print(result.summary())
```
