# Marginal Data Density

`dsgekit` provides two estimators for log marginal data density (log MDD):

- `estimate_mdd_laplace(...)`
- `estimate_mdd_harmonic_mean(...)` (optional, high-variance estimator)

## Laplace Approximation

```python
from dsgekit.estimation import estimate_mdd_laplace

laplace = estimate_mdd_laplace(
    model,
    ss,
    cal,
    data,
    observables=["y"],
    param_names=None,   # inferred from estimated_params
    hessian_eps=1e-4,
)

print(laplace.log_mdd)
print(laplace.summary())
```

## Harmonic Mean (from MCMC draws)

```python
from dsgekit.estimation import (
    estimate_mcmc,
    estimate_mdd_harmonic_mean,
)

mcmc = estimate_mcmc(
    model,
    ss,
    cal,
    data,
    observables=["y"],
    param_names=None,
    n_draws=3000,
    burn_in=600,
    thin=2,
    proposal_scale=0.04,
    seed=42,
)

harmonic = estimate_mdd_harmonic_mean(
    model,
    ss,
    cal,
    data,
    observables=["y"],
    param_names=None,
    mcmc_result=mcmc,
    max_samples=1000,  # optional subsampling for speed
    seed=7,
)

print(harmonic.log_mdd)
print(harmonic.summary())
```

## Notes

- For harmonic mean, `mcmc_result.param_names` must match the estimation parameter order.
- Harmonic mean is often unstable; use it as a secondary diagnostic against Laplace.
