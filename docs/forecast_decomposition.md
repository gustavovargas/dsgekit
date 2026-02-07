# Forecast and Historical Decomposition

`dsgekit` provides linear state-space forecasting and shock decomposition tools:

- `forecast(...)`
- `historical_decomposition(...)`

Both APIs work with `to_state_space(...)`, `kalman_filter(...)`, and `kalman_smoother(...)`.

## Forecast

```python
from dsgekit.filters import forecast, kalman_filter, kalman_smoother
from dsgekit.transforms import to_state_space

ss_model = to_state_space(solution, cal, observables=["y"])
kf = kalman_filter(ss_model, data)
sm = kalman_smoother(ss_model, kf)

fc = forecast(
    ss_model,
    kf,
    steps=12,            # out-of-sample horizon
    smoother_result=sm,  # use final smoothed state (default source)
)

print(fc.summary())
print(fc.in_sample_observables.head())      # one-step in-sample forecasts
print(fc.out_of_sample_observables.head())  # h-step forecasts
print(fc.intervals(alpha=0.05).head())      # Gaussian prediction intervals
```

## Historical Decomposition

```python
from dsgekit.filters import historical_decomposition

hd = historical_decomposition(
    ss_model,
    sm,
    include_initial=True,
    include_residual=True,
    include_steady_state=True,
)

print(hd.summary())

# Component contribution for one shock in observables
print(hd.component("e", observables=True).head())

# Multi-component table (columns: component x observable)
print(hd.observable_contributions.head())
```

## CLI

```bash
# 12-step forecast using synthetic training data
dsgekit forecast model.yaml -p 12 --train-periods 120 --seed 42

# Historical decomposition using a CSV file with observable columns
dsgekit decompose model.yaml --data data.csv --observables y,pi
```
