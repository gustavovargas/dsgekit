# Examples

Run these from repository root with the virtual environment active.

## AR(1) Full Pipeline

```bash
python examples/ar1_pipeline.py
```

Covers:

- load -> linearize -> solve
- IRFs, simulation, moments
- state-space + Kalman filter/smoother
- MLE, MAP, MCMC estimation (+ ESS/R-hat diagnostics), marginal data density, forecast, and historical decomposition
- Deterministic perfect-foresight transition paths

## NK Forward-Looking Model

```bash
python examples/nk_forward_looking.py
```

Covers:

- forward-looking setup and BK failure on default solve
- diagnostics with `check_bk=False`
- IRFs for all shocks

## Reference Model Zoo (RBC, NK, SOE, Fiscal, ZLB toy)

```bash
python examples/reference_model_zoo.py
```

Covers:

- load and solve for 5 reference models
- BK diagnostics summary per model
- short IRF snapshot for one representative shock

## Nonlinear GIRFs (Second-Order Pruning)

```bash
python examples/nonlinear_girf.py
```

Covers:

- second-order perturbation solve on a non-linear AR(1) toy model
- generalized IRFs (GIRFs) at two different initial states
- simulation on an explicit shock trajectory (state-dependent path)

## OccBin-lite on ZLB toy

```bash
python examples/occbin_zlb_toy.py
```

Covers:

- two-regime deterministic solve with one occasionally binding restriction
- relaxed (`i = i_shadow`) vs binding (`i = 0`) policy-rate equation
- regime path output (`binding_regime`) with solved trajectory

## Newton JIT Benchmark

```bash
python examples/benchmark_newton_jit.py --periods 240 --eq 8 --vars 8 --repeats 12
```

Covers:

- dense block Jacobian assembly baseline (pure Python path)
- optional `numba` JIT dense assembly path (if `dsgekit[speed]` is installed)
- median runtime and speedup reporting

## Block Sparse Newton Benchmark

```bash
python examples/benchmark_block_sparse_newton.py --periods 240 --vars 8 --zero-prob 0.90 --repeats 4
```

Covers:

- dense vs sparse block Jacobian assembly + linear solve benchmark
- medium/large horizon configurations with controlled sparsity
- median runtime and total speedup reporting
