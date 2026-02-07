# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0rc1] - 2026-02-07

### Added
- Initial project structure
- Core model representation (IR, symbols, equations)
- Linear solver (Gensys/QZ)
- IRFs and simulation
- Basic `.mod` parser
- State-space representation for Kalman filtering (`StateSpace`, `to_state_space`)
- New exceptions: `StateSpaceError`, `InvalidObservableError`
- Kalman filter with missing-observations support and Lyapunov initialization
- RTS (Rauch-Tung-Striebel) Kalman smoother
- Maximum likelihood estimation (MLE) with Hessian-based standard errors, AIC/BIC
- `.mod` macro preprocessor: `@#define`, `@#if/@#elseif/@#else/@#endif`, `@#include` (DYN-A01)
- Extended `.mod` block support: `varobs`, `estimated_params`, `histval`, `endval`, `steady_state_model` (DYN-A02)
- Non-linear expression support: `ln`, `abs`, `sin`, `cos`, `tan`, `pow`, `min`, `max` helpers; centralized `KNOWN_FUNCTIONS`/`EQUATION_FUNCTIONS` registry; all three parsers (`.mod`, YAML, Python API) support all 11 built-in functions (DYN-A03)
- RBC model fixture with correct analytical steady state
- BK diagnostics with reproducible traces (`trace_id`, canonical trace payload) and companion-adjusted stability counting (DYN-A04)
- Linear baseline compatibility harness: baseline-driven runner for solution + IRFs with tolerance-based checks (`run_linear_compat`) and AR(1) reference baseline fixture (DYN-A05)
- Hardened core-linear CI workflow with split jobs (`parser`, `solver`, `filters`, `smoke-nk`), per-job timeout budgets, and duration reporting in step summaries (DYN-A06)
- Unified derivative stack API for equation systems (`DerivativeStack`): Jacobian, Hessian, and third-order tensors with switchable backends (`numeric`, optional `sympy`) (DYN-B01)
- First-order perturbation solver for non-linear models (`linearize_first_order`, `solve_first_order`) using the derivative stack and BK/QZ solution path, with parity tests vs legacy `linearize + solve_linear` (DYN-B02)
- Priors DSL for `estimated_params` (`normal`/`beta`/`gamma`/`inv_gamma`) with canonical alias handling, validation, Python API builders (`estimate_param`, `estimate_stderr`, `estimate_corr`), parser checks, and round-trip serialization support (DYN-C01)
- Posterior mode (MAP) optimization with prior log-density support (`estimate_map`), automatic inference of estimable names/bounds from `calibration.estimated_params`, robust initial-point clipping to bounds, and reproducible convergence tests on AR(1) pipelines (DYN-C02)
- Metropolis-Hastings posterior sampling (`sample_posterior_mh` / `estimate_mcmc`) with configurable draws, burn-in, thinning, proposal scale, and seed control; includes reproducibility and parameter-recovery integration tests (DYN-C03)
- MCMC diagnostics (`MCMCDiagnostics`) with per-parameter effective sample size (ESS), split-chain R-hat, acceptance-rate reporting, and trace extraction via `MCMCResult.trace_dict()` (DYN-C04)
- Marginal data density estimators for Bayesian model comparison: Laplace approximation (`estimate_mdd_laplace`) and optional harmonic mean estimator over posterior draws (`estimate_mdd_harmonic_mean`) with dedicated tests and docs (DYN-C05)
- Forecasting and historical decomposition tools on top of Kalman filter/smoother: new `forecast(...)` and `historical_decomposition(...)` APIs, plus CLI commands `dsgekit forecast` and `dsgekit decompose` with integration and CLI coverage tests (DYN-C06)
- Deterministic perfect-foresight solver for non-linear transition paths (`solve_perfect_foresight`) with Newton + line search, boundary conditions (`initial_state`/`terminal_state`), deterministic shock paths, and canonical convergence tests (AR1 and lead-lag toy model) (DYN-D01)
- Welfare metrics API: expected utility + quadratic loss evaluation for scenario paths (`evaluate_welfare`) and ranked cross-scenario comparisons (`compare_welfare`), with dedicated docs and tests (DYN-E04)
- OSR-style policy-rule sweeps: grid-search optimizer for parameterized rule coefficients (`osr_grid_search`) plus CLI command (`dsgekit osr`) and CSV ranking output (DYN-E03)
- Ramsey LQ commitment solver: discounted Riccati-based optimal policy (`solve_ramsey_lq`), reduced-form wrapper (`solve_ramsey_from_linear_solution`), and closed-loop path simulation (`simulate_ramsey_path`) with canonical tests (DYN-E01)
- Discretionary LQ policy solver: receding-horizon feedback policy (`solve_discretion_lq`), reduced-form wrapper (`solve_discretion_from_linear_solution`), and closed-loop simulation (`simulate_discretion_path`) with dedicated docs and tests (DYN-E02)
- Third-order perturbation for backward-looking models: cubic policy tensors via implicit derivatives (`solve_third_order`) plus pruned simulation APIs (`simulate_pruned_third_order`, `simulate_pruned_third_order_path`) with dedicated regression/stability tests (DYN-B04)

### Changed
- `.mod` and Python API parsers now use a centralized function registry instead of hardcoded lists
- BK checks now run for no-lead systems too, so explosive AR models raise `NoStableSolutionError` by default
- NK/RBC fixtures now classify as determinate under the corrected BK counting logic (DYN-A04 regression coverage)
- `linearize()` now reuses a mutable evaluation context during finite-difference Jacobian construction (less allocation overhead, faster repeated linearizations)
- `build_objective()` now memoizes repeated `theta` evaluations with bounded LRU cache (enabled by default; configurable via `cache` / `cache_max_size`)
- `_compute_std_errors()` now caches Hessian evaluation points and uses a diagonal-specific finite-difference path, reducing redundant objective evaluations

### Removed
- Dead `_convert_timing()` function from `.mod` parser
