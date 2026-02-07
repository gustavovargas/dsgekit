# MCMC Metropolis-Hastings

`dsgekit` includes random-walk Metropolis-Hastings posterior sampling:

- `sample_posterior_mh(...)`
- `estimate_mcmc(...)` (alias)
- diagnostics via `MCMCResult.diagnostics()`

## Example

```python
from dsgekit.estimation import estimate_mcmc
from dsgekit.model.calibration import EstimatedParam

cal.estimated_params = [
    EstimatedParam.from_dict(
        {
            "type": "param",
            "name": "rho",
            "init": 0.7,
            "lower": 0.01,
            "upper": 0.99,
            "prior": {"distribution": "normal", "mean": 0.8, "std": 0.12},
        }
    )
]

mcmc = estimate_mcmc(
    model,
    ss,
    cal,
    data,
    observables=["y"],
    param_names=None,   # inferred from estimated_params
    n_draws=5000,
    burn_in=1000,
    thin=5,
    proposal_scale=0.04,
    seed=42,
)

print(mcmc.summary())
print(mcmc.posterior_mean())
print(mcmc.diagnostics().summary())

# Trace vectors ready for plotting
traces = mcmc.trace_dict(post_burn=True)
print(traces["rho"][:5])
```

## Controls

- `n_draws`: total MH iterations
- `burn_in`: discarded prefix of the chain
- `thin`: keep every `thin` sample after burn-in
- `seed`: reproducible random generator seed
- `proposal_scale`: scalar or per-parameter proposal std

## Diagnostics

`mcmc.diagnostics()` returns:

- `acceptance_rate`: empirical acceptance ratio
- `ess`: effective sample size per parameter
- `r_hat`: split-chain R-hat per parameter
- `notes`: warnings/notes (for example, if R-hat is unavailable)

R-hat uses split-chain diagnostics on the saved posterior samples. If too few samples are available, R-hat is reported as `nan`.
