"""Pruned second-order stochastic simulation and GIRFs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dsgekit.model.calibration import Calibration
    from dsgekit.solvers.nonlinear.second_order import SecondOrderSolution


@dataclass
class SecondOrderSimulationResult:
    """Results from second-order pruned simulation."""

    data: pd.DataFrame
    shocks: pd.DataFrame
    first_order_component: pd.DataFrame
    second_order_component: pd.DataFrame
    n_periods: int
    seed: int | None = None

    def __getitem__(self, var_name: str) -> NDArray[np.float64]:
        return self.data[var_name].values


@dataclass
class GeneralizedIRFResult:
    """Results from generalized impulse response computation."""

    data: pd.DataFrame
    baseline_mean: pd.DataFrame
    shocked_mean: pd.DataFrame
    shock_name: str
    shock_size: float
    periods: int
    n_draws: int
    seed: int | None = None

    def __getitem__(self, var_name: str) -> NDArray[np.float64]:
        return self.data[var_name].values


def _coerce_initial_state(
    initial_state: NDArray[np.float64] | None,
    n_vars: int,
) -> NDArray[np.float64]:
    if initial_state is None:
        return np.zeros(n_vars, dtype=np.float64)
    init = np.asarray(initial_state, dtype=np.float64).reshape(-1)
    if init.shape[0] != n_vars:
        raise ValueError(f"initial_state has length {init.shape[0]}, expected {n_vars}")
    return init.copy()


def _coerce_shocks(
    shocks: NDArray[np.float64] | pd.DataFrame,
    expected_names: list[str],
) -> NDArray[np.float64]:
    n_shocks = len(expected_names)
    if isinstance(shocks, pd.DataFrame):
        missing = [name for name in expected_names if name not in shocks.columns]
        if missing:
            raise ValueError(f"Shock DataFrame is missing columns: {missing}")
        return shocks.loc[:, expected_names].to_numpy(dtype=np.float64, copy=True)

    shock_array = np.asarray(shocks, dtype=np.float64)
    if shock_array.ndim != 2:
        raise ValueError(
            f"shocks must be 2D with shape (n_periods, n_shocks), got ndim={shock_array.ndim}"
        )
    if shock_array.shape[1] != n_shocks:
        raise ValueError(
            f"shocks has {shock_array.shape[1]} columns, expected {n_shocks}"
        )
    return shock_array.copy()


def _simulate_pruned_arrays(
    solution: SecondOrderSolution,
    shocks: NDArray[np.float64],
    initial_state: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    T = solution.T
    R = solution.R
    n_periods, n_shocks = shocks.shape
    n_vars = T.shape[0]

    if n_shocks != R.shape[1]:
        raise ValueError(f"shocks has {n_shocks} columns, expected {R.shape[1]}")

    x1 = _coerce_initial_state(initial_state, n_vars)
    x2 = np.zeros(n_vars, dtype=np.float64)

    paths_total = np.zeros((n_periods, n_vars), dtype=np.float64)
    paths_fo = np.zeros((n_periods, n_vars), dtype=np.float64)
    paths_so = np.zeros((n_periods, n_vars), dtype=np.float64)

    for t in range(n_periods):
        u_t = shocks[t, :]
        x1_prev = x1.copy()
        quad_t = solution.quadratic_effect(x1_prev, u_t)

        x1 = T @ x1_prev + R @ u_t
        x2 = T @ x2 + quad_t

        y_t = x1 + x2
        paths_total[t, :] = y_t
        paths_fo[t, :] = x1
        paths_so[t, :] = x2

    return paths_total, paths_fo, paths_so


def _shock_cholesky(calibration: Calibration, shock_names: list[str]) -> NDArray[np.float64]:
    shock_cov = calibration.shock_cov_matrix(shock_names)
    if np.allclose(shock_cov, 0.0):
        return np.zeros_like(shock_cov)
    return np.linalg.cholesky(shock_cov)


def simulate_pruned_second_order_path(
    solution: SecondOrderSolution,
    shocks: NDArray[np.float64] | pd.DataFrame,
    *,
    initial_state: NDArray[np.float64] | None = None,
) -> SecondOrderSimulationResult:
    """Run second-order pruned simulation for a user-provided shock trajectory."""
    shock_array = _coerce_shocks(shocks, solution.shock_names)
    paths_total, paths_fo, paths_so = _simulate_pruned_arrays(
        solution,
        shock_array,
        initial_state,
    )

    n_periods = shock_array.shape[0]
    index = range(n_periods)
    data_df = pd.DataFrame(paths_total, index=index, columns=solution.var_names)
    data_df.index.name = "period"
    fo_df = pd.DataFrame(paths_fo, index=index, columns=solution.var_names)
    fo_df.index.name = "period"
    so_df = pd.DataFrame(paths_so, index=index, columns=solution.var_names)
    so_df.index.name = "period"
    shocks_df = pd.DataFrame(shock_array, index=index, columns=solution.shock_names)
    shocks_df.index.name = "period"

    return SecondOrderSimulationResult(
        data=data_df,
        shocks=shocks_df,
        first_order_component=fo_df,
        second_order_component=so_df,
        n_periods=n_periods,
        seed=None,
    )


def simulate_pruned_second_order(
    solution: SecondOrderSolution,
    calibration: Calibration,
    n_periods: int = 100,
    seed: int | None = None,
    initial_state: NDArray[np.float64] | None = None,
    burn_in: int = 0,
) -> SecondOrderSimulationResult:
    """Run second-order simulation with pruning.

    Uses:
      x1_t = T x1_{t-1} + R u_t
      x2_t = T x2_{t-1} + 0.5 * G2([x1_{t-1}, u_t], [x1_{t-1}, u_t])
      y_t  = x1_t + x2_t
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    n_shocks = solution.n_shocks
    total_periods = n_periods + burn_in

    shock_chol = _shock_cholesky(calibration, solution.shock_names)
    z = rng.standard_normal((total_periods, n_shocks))
    shocks = z @ shock_chol.T
    paths_total, paths_fo, paths_so = _simulate_pruned_arrays(
        solution,
        shocks,
        initial_state,
    )

    paths_total = paths_total[burn_in:, :]
    paths_fo = paths_fo[burn_in:, :]
    paths_so = paths_so[burn_in:, :]
    shocks = shocks[burn_in:, :]

    index = range(n_periods)
    data_df = pd.DataFrame(paths_total, index=index, columns=solution.var_names)
    data_df.index.name = "period"
    fo_df = pd.DataFrame(paths_fo, index=index, columns=solution.var_names)
    fo_df.index.name = "period"
    so_df = pd.DataFrame(paths_so, index=index, columns=solution.var_names)
    so_df.index.name = "period"
    shocks_df = pd.DataFrame(shocks, index=index, columns=solution.shock_names)
    shocks_df.index.name = "period"

    return SecondOrderSimulationResult(
        data=data_df,
        shocks=shocks_df,
        first_order_component=fo_df,
        second_order_component=so_df,
        n_periods=n_periods,
        seed=seed,
    )


def girf_pruned_second_order(
    solution: SecondOrderSolution,
    calibration: Calibration,
    shock_name: str,
    *,
    periods: int = 40,
    shock_size: float = 1.0,
    n_draws: int = 500,
    seed: int | None = None,
    initial_state: NDArray[np.float64] | None = None,
) -> GeneralizedIRFResult:
    """Compute state-dependent GIRFs using pruned second-order simulation.

    For each Monte Carlo draw, future shocks are shared between baseline and
    shocked paths. The shocked path adds `shock_size` to `shock_name` at period 0.
    The returned GIRF is the average path difference across draws.
    """
    if periods <= 0:
        raise ValueError(f"periods must be positive, got {periods}")
    if n_draws <= 0:
        raise ValueError(f"n_draws must be positive, got {n_draws}")

    try:
        shock_idx = solution.shock_names.index(shock_name)
    except ValueError as err:
        raise ValueError(
            f"Shock '{shock_name}' not found. Available: {solution.shock_names}"
        ) from err

    rng = np.random.default_rng(seed)
    n_shocks = solution.n_shocks
    n_vars = solution.n_variables
    shock_chol = _shock_cholesky(calibration, solution.shock_names)
    state0 = _coerce_initial_state(initial_state, n_vars)

    baseline_acc = np.zeros((periods, n_vars), dtype=np.float64)
    shocked_acc = np.zeros((periods, n_vars), dtype=np.float64)

    for _ in range(n_draws):
        z = rng.standard_normal((periods, n_shocks))
        draw_shocks = z @ shock_chol.T

        shocked_shocks = draw_shocks.copy()
        shocked_shocks[0, shock_idx] += shock_size

        baseline_paths, _, _ = _simulate_pruned_arrays(solution, draw_shocks, state0)
        shocked_paths, _, _ = _simulate_pruned_arrays(solution, shocked_shocks, state0)
        baseline_acc += baseline_paths
        shocked_acc += shocked_paths

    baseline_mean = baseline_acc / n_draws
    shocked_mean = shocked_acc / n_draws
    girf_data = shocked_mean - baseline_mean

    index = range(periods)
    columns = solution.var_names
    girf_df = pd.DataFrame(girf_data, index=index, columns=columns)
    girf_df.index.name = "period"
    baseline_df = pd.DataFrame(baseline_mean, index=index, columns=columns)
    baseline_df.index.name = "period"
    shocked_df = pd.DataFrame(shocked_mean, index=index, columns=columns)
    shocked_df.index.name = "period"

    return GeneralizedIRFResult(
        data=girf_df,
        baseline_mean=baseline_df,
        shocked_mean=shocked_df,
        shock_name=shock_name,
        shock_size=shock_size,
        periods=periods,
        n_draws=n_draws,
        seed=seed,
    )
