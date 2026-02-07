# dsgekit

A Python toolkit for Dynamic Stochastic General Equilibrium (DSGE) models.

## Features

- **Model specification**: Define models via Python API, YAML, or `.mod` files
- **Non-linear equations**: Full support for `exp`, `log`, `sqrt`, `abs`, trig functions, `pow`, `min`/`max`, and power operators in model equations
- **`.mod` compatibility subset**: Macro preprocessor (`@#define`, `@#if`, `@#include`), `varobs`, `estimated_params`, `histval`, `endval`, `steady_state_model` (helps migration from Dynare-style workflows)
- **Linear baseline harness**: Compare solution matrices and IRFs against stored reference baselines with explicit tolerances
- **Priors DSL**: Validated `normal`/`beta`/`gamma`/`inv_gamma` priors for `estimated_params` across `.mod`, YAML/dict, and Python API
- **Unified derivative stack**: Jacobian/Hessian/third-order tensors with switchable `numeric` and optional `sympy` backends
- **First-order perturbation solver**: Direct stochastic policy solution for non-linear models around steady state (`solve_first_order`)
- **Second-order perturbation + pruning**: Quadratic policy tensor, state-dependent trajectories, and GIRFs for backward-looking models (`solve_second_order`, `simulate_pruned_second_order`, `simulate_pruned_second_order_path`, `girf_pruned_second_order`)
- **Third-order perturbation + pruning**: Cubic policy tensor with pruned stochastic simulation for backward-looking models (`solve_third_order`, `simulate_pruned_third_order`, `simulate_pruned_third_order_path`)
- **Perfect foresight solver**: Deterministic non-linear transition paths with fixed initial/terminal states (`solve_perfect_foresight`)
- **OccBin-lite (one constraint)**: Deterministic two-regime solver for occasionally binding constraints (`solve_occbin_lite`)
- **Block-sparse + optional JIT Newton linear algebra**: Sparse time-block Jacobian assembly/solve plus opt-in `numba` dense assembly (`linear_solver="auto|dense|sparse"`, `jit_backend="none|numba"`) for deterministic perfect-foresight and OccBin-lite workflows
- **News shocks API**: Define anticipated/unanticipated shock events and build deterministic paths (`anticipated_shock`, `unanticipated_shock`, `build_news_shock_path`)
- **Linear solution**: Blanchard-Kahn / Gensys solver with QZ decomposition
- **State-space**: Automatic conversion to state-space form for filtering
- **Simulation**: Stochastic simulation, impulse response functions, moments, FEVD
- **Estimation**: Kalman filter, RTS smoother, maximum likelihood (MLE), posterior mode (MAP), MCMC Metropolis-Hastings with ESS/R-hat diagnostics and trace extraction, marginal data density (Laplace + optional harmonic mean)
- **Forecasting and decomposition**: In-sample/out-of-sample forecasts and historical shock decomposition from smoothed states
- **OSR rule sweeps**: Grid-search optimizer for policy-rule parameters with weighted-variance objective (`osr_grid_search`, `dsgekit osr`)
- **Ramsey LQ policy**: Linear-quadratic commitment solver and closed-loop path simulation (`solve_ramsey_lq`, `solve_ramsey_from_linear_solution`, `simulate_ramsey_path`)
- **Discretionary LQ policy**: Receding-horizon discretionary solver and closed-loop path simulation (`solve_discretion_lq`, `solve_discretion_from_linear_solution`, `simulate_discretion_path`)
- **Welfare metrics**: Expected utility and quadratic-loss evaluation with scenario comparison (`evaluate_welfare`, `compare_welfare`)

## Requirements

- Python >= 3.13

## Installation

```bash
pip install dsgekit
```

With optional dependencies:

```bash
pip install dsgekit[plot]      # matplotlib for plots
pip install dsgekit[sym]       # sympy for symbolic derivatives
pip install dsgekit[all]       # all optional dependencies
```

## Development Setup

```bash
# Install Python 3.13 (Ubuntu)
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update && sudo apt install -y python3.13 python3.13-venv python3.13-dev

# Clone and setup
git clone https://github.com/gustavovargas/dsgekit.git
cd dsgekit
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plot]"
```

## Quick Start

```python
from dsgekit import load_model
from dsgekit.simulate import irf
from dsgekit.solvers import solve_first_order, solve_linear
from dsgekit.transforms import linearize

# Load model + calibration + steady state
model, cal, ss = load_model("my_model.mod")

# Linearize and solve
solution = solve_linear(linearize(model, ss, cal))

# Alternative: first-order perturbation solver for non-linear models
solution_nl = solve_first_order(model, ss, cal)

# Compute IRFs
irfs = irf(solution, "e", periods=40)

print(irfs.data.head())
```

See:

- `docs/quickstart.md` for end-to-end workflow
- `docs/formats.md` for `.mod`, YAML, and `dict` schemas
- `docs/priors.md` for priors and `estimated_params` DSL
- `docs/mcmc.md` for posterior sampling controls and usage
- `docs/marginal_data_density.md` for model-evidence estimators
- `docs/forecast_decomposition.md` for forecasting and historical decomposition
- `docs/osr.md` for OSR-style policy-rule sweeps (API and CLI)
- `docs/ramsey_lq.md` for Ramsey linear-quadratic commitment policy
- `docs/discretion_lq.md` for discretionary linear-quadratic policy workflows
- `docs/welfare.md` for expected-utility/quadratic-loss metrics and scenario comparison
- `docs/second_order_perturbation.md` for quadratic perturbation and pruned simulation
- `docs/third_order_perturbation.md` for cubic perturbation and pruned simulation
- `docs/perfect_foresight.md` for deterministic transition-path solving
- `docs/occbin_lite.md` for one-constraint occasionally binding regime solving
- `docs/mod_workflow_migration.md` for migration tips and `.mod` workflow tutorials
- `docs/legal_and_attribution.md` for non-affiliation notice and third-party attributions

## Examples

```bash
python examples/ar1_pipeline.py
python examples/nk_forward_looking.py
python examples/reference_model_zoo.py
python examples/nonlinear_girf.py
python examples/occbin_zlb_toy.py
```

More details in `examples/README.md`.

## CLI

```bash
dsgekit simulate model.mod
dsgekit estimation model.mod --method mle --params rho
dsgekit forecast model.mod -p 12
dsgekit osr model.mod --grid phi_pi=1.0:3.0:9 --loss pi=1.0 --loss y=0.5
dsgekit baseline_regression --baselines path/to/baselines
```

## Status

**Alpha** - Under active development. API may change.

## Legal Notice

- `dsgekit` is an independent project and is not affiliated with third-party tool maintainers.
- Third-party references in this repository are limited to compatibility, migration, and attribution context.
- See `docs/legal_and_attribution.md`, `docs/brand_naming_policy.md`, `docs/release_legal_gate.md`, and `THIRD_PARTY_NOTICES.md` for attribution, naming policy, release-gate process, and dependency-license inventory details.

## License

MIT
