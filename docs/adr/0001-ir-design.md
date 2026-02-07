# ADR 0001: Canonical IR-Centered Architecture

- Status: Accepted
- Date: 2026-02-05
- Decision Makers: dsgekit maintainers

## Context

`dsgekit` must support multiple model input formats (`.mod`, YAML, Python `dict`/API) while keeping:

- one solver interface
- one simulation/filtering/estimation pipeline
- consistent diagnostics across formats

Without a shared internal model representation, each parser would need custom logic for linearization, solving, filtering, and reporting, creating duplicated behavior and inconsistent results.

## Decision

Use a canonical intermediate representation (`ModelIR`) as the core contract:

1. Parsers only translate input into `ModelIR` + `Calibration` + `SteadyState`.
2. Core pipeline consumes this representation:
   - `linearize(ModelIR, SteadyState, Calibration)`
   - `solve_linear(LinearizedSystem)`
   - `to_state_space(LinearSolution, Calibration, observables)`
   - filtering/estimation/reporting on top of these outputs.
3. Timing semantics are encoded explicitly in IR (`LeadLagStructure`) to classify:
   - predetermined variables
   - forward-looking variables
   - static variables

## Rationale

- Single source of truth for model structure and timing.
- Feature work (new parser, diagnostics, estimators) composes around shared data structures.
- Testing is simpler: parser tests verify conversion; pipeline tests verify behavior once at IR level.
- Better maintainability than parser-specific downstream code paths.

## Consequences

### Positive

- Consistent behavior across `.mod`, YAML, and dict/API sources.
- Lower long-term maintenance cost.
- Clear extension points for future Bayesian estimation and richer parser support.

### Negative

- Parsers must normalize syntax differences upfront.
- Extra abstraction layer adds initial complexity.

## Alternatives Considered

1. Parser-specific execution paths.
   - Rejected: high duplication and drift risk.
2. Solver-centric API with ad-hoc parser adapters.
   - Rejected: weak typing/contracts for downstream components.
3. Symbolic-first core (SymPy/JAX) for everything.
   - Rejected for now: heavier dependencies and slower iteration than numeric-first core.

## Notes

- Current architecture in code aligns with this ADR: `parser -> IR -> solver -> state-space/filter/estimation`.
- Future ADRs may refine:
  - higher-order timing support
  - symbolic derivative plug-ins
  - Bayesian estimation workflow.
