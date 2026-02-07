"""Forecasting and historical decomposition from state-space results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from dsgekit.exceptions import FilterError

if TYPE_CHECKING:
    from dsgekit.filters.kalman import KalmanResult
    from dsgekit.filters.smoother import SmootherResult
    from dsgekit.transforms.statespace import StateSpace


def _obs_steady_state_vector(state_space: StateSpace) -> NDArray[np.float64]:
    return np.array(
        [state_space.steady_state.get(name, 0.0) for name in state_space.obs_names],
        dtype=np.float64,
    )


def _build_multiindex_frame(
    payload: dict[str, NDArray[np.float64]],
    *,
    index: pd.Index,
    columns: list[str],
    first_level_name: str,
    second_level_name: str,
) -> pd.DataFrame:
    parts = {
        name: pd.DataFrame(values, index=index, columns=columns)
        for name, values in payload.items()
    }
    out = pd.concat(parts, axis=1)
    out.columns.names = [first_level_name, second_level_name]
    return out


@dataclass
class ForecastResult:
    """In-sample and out-of-sample forecasts from a filtered state-space model."""

    in_sample_observables: pd.DataFrame
    out_of_sample_states: pd.DataFrame
    out_of_sample_observables: pd.DataFrame
    out_of_sample_state_cov: NDArray[np.float64]
    out_of_sample_obs_cov: NDArray[np.float64]
    horizon: int
    source: str

    def __getitem__(self, observable: str) -> NDArray[np.float64]:
        return self.out_of_sample_observables[observable].values

    def intervals(self, alpha: float = 0.05) -> pd.DataFrame:
        """Gaussian prediction intervals for out-of-sample observables."""
        if not (0.0 < alpha < 1.0):
            raise FilterError(f"alpha must be in (0, 1), got {alpha}")
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        std = np.sqrt(np.maximum(0.0, np.diagonal(self.out_of_sample_obs_cov, axis1=1, axis2=2)))
        lower = self.out_of_sample_observables.values - z * std
        upper = self.out_of_sample_observables.values + z * std

        frames: dict[str, pd.DataFrame] = {}
        for i, name in enumerate(self.out_of_sample_observables.columns):
            frames[name] = pd.DataFrame(
                {
                    "lower": lower[:, i],
                    "upper": upper[:, i],
                },
                index=self.out_of_sample_observables.index,
            )
        out = pd.concat(frames, axis=1)
        out.columns.names = ["observable", "bound"]
        return out

    def summary(self) -> str:
        lines = [
            "Forecast",
            "=" * 50,
            f"  Source state:     {self.source}",
            f"  In-sample rows:   {self.in_sample_observables.shape[0]}",
            f"  Out-of-sample h:  {self.horizon}",
            f"  Observables:      {', '.join(self.out_of_sample_observables.columns)}",
        ]
        return "\n".join(lines)


@dataclass
class HistoricalDecompositionResult:
    """Shock-wise historical decomposition on smoothed states/observables."""

    state_contributions: pd.DataFrame
    observable_contributions: pd.DataFrame
    reconstructed_states: pd.DataFrame
    reconstructed_observables: pd.DataFrame
    target_states: pd.DataFrame
    target_observables: pd.DataFrame
    inferred_shocks: pd.DataFrame
    components: list[str]
    max_abs_state_error: float
    max_abs_obs_error: float

    def component(self, name: str, *, observables: bool = True) -> pd.DataFrame:
        """Extract one component contribution table."""
        table = self.observable_contributions if observables else self.state_contributions
        if name not in table.columns.get_level_values(0):
            raise FilterError(f"Component '{name}' not found")
        return table[name]

    def summary(self) -> str:
        lines = [
            "Historical Decomposition",
            "=" * 50,
            f"  Periods:             {self.target_states.shape[0]}",
            f"  Components:          {', '.join(self.components)}",
            f"  Max |state error|:   {self.max_abs_state_error:.3e}",
            f"  Max |obs error|:     {self.max_abs_obs_error:.3e}",
        ]
        return "\n".join(lines)


def forecast(
    state_space: StateSpace,
    kalman_result: KalmanResult,
    *,
    steps: int = 12,
    smoother_result: SmootherResult | None = None,
    include_steady_state: bool = True,
) -> ForecastResult:
    """Compute in-sample one-step forecasts and out-of-sample projections."""
    if steps < 1:
        raise FilterError(f"steps must be >= 1, got {steps}")

    ss = state_space
    kf = kalman_result

    if kf.predicted_states.shape[1] != ss.n_states:
        raise FilterError(
            f"State dimension mismatch: kalman_result has {kf.predicted_states.shape[1]} "
            f"states, state_space has {ss.n_states}"
        )

    in_sample_obs = kf.predicted_states.values @ ss.Z.T
    if include_steady_state:
        in_sample_obs = in_sample_obs + _obs_steady_state_vector(ss)[np.newaxis, :]
    in_sample_df = pd.DataFrame(
        in_sample_obs,
        index=kf.predicted_states.index,
        columns=ss.obs_names,
    )
    in_sample_df.index.name = "period"

    if smoother_result is not None:
        state = smoother_result.smoothed_states.iloc[-1].values.astype(np.float64, copy=True)
        cov = smoother_result.smoothed_cov[-1].astype(np.float64, copy=True)
        source = "smoothed"
    else:
        state = kf.filtered_states.iloc[-1].values.astype(np.float64, copy=True)
        cov = kf.filtered_cov[-1].astype(np.float64, copy=True)
        source = "filtered"

    states = np.zeros((steps, ss.n_states), dtype=np.float64)
    states_cov = np.zeros((steps, ss.n_states, ss.n_states), dtype=np.float64)
    obs = np.zeros((steps, ss.n_obs), dtype=np.float64)
    obs_cov = np.zeros((steps, ss.n_obs, ss.n_obs), dtype=np.float64)

    for h in range(steps):
        state = ss.T @ state
        cov = ss.T @ cov @ ss.T.T + ss.Q
        cov = 0.5 * (cov + cov.T)

        states[h, :] = state
        states_cov[h, :, :] = cov
        obs[h, :] = ss.Z @ state
        obs_cov[h, :, :] = ss.Z @ cov @ ss.Z.T + ss.H
        obs_cov[h, :, :] = 0.5 * (obs_cov[h, :, :] + obs_cov[h, :, :].T)

    if include_steady_state:
        obs = obs + _obs_steady_state_vector(ss)[np.newaxis, :]

    horizon_index = pd.RangeIndex(start=1, stop=steps + 1, name="horizon")
    state_df = pd.DataFrame(states, index=horizon_index, columns=ss.state_names)
    obs_df = pd.DataFrame(obs, index=horizon_index, columns=ss.obs_names)

    return ForecastResult(
        in_sample_observables=in_sample_df,
        out_of_sample_states=state_df,
        out_of_sample_observables=obs_df,
        out_of_sample_state_cov=states_cov,
        out_of_sample_obs_cov=obs_cov,
        horizon=steps,
        source=source,
    )


def historical_decomposition(
    state_space: StateSpace,
    smoother_result: SmootherResult,
    *,
    include_initial: bool = True,
    include_residual: bool = True,
    include_steady_state: bool = True,
) -> HistoricalDecompositionResult:
    """Decompose smoothed trajectories into shock contributions."""
    ss = state_space
    sm = smoother_result

    target_states = sm.smoothed_states.values.astype(np.float64, copy=False)
    n_periods, n_states = target_states.shape
    n_shocks = ss.n_shocks
    if n_periods == 0:
        raise FilterError("smoother_result contains no periods")
    if n_states != ss.n_states:
        raise FilterError(
            f"State dimension mismatch: smoother_result has {n_states} states, "
            f"state_space has {ss.n_states}"
        )

    inferred_shocks = np.zeros((n_periods, n_shocks), dtype=np.float64)
    if n_shocks > 0:
        for t in range(n_periods):
            prev = target_states[t - 1] if t > 0 else np.zeros(n_states, dtype=np.float64)
            rhs = target_states[t] - ss.T @ prev
            eps, *_ = np.linalg.lstsq(ss.R, rhs, rcond=None)
            inferred_shocks[t, :] = eps

    state_parts: dict[str, NDArray[np.float64]] = {}

    for j, shock_name in enumerate(ss.shock_names):
        contrib = np.zeros((n_periods, n_states), dtype=np.float64)
        impact = ss.R[:, j]
        for t in range(n_periods):
            prev = contrib[t - 1] if t > 0 else np.zeros(n_states, dtype=np.float64)
            contrib[t, :] = ss.T @ prev + impact * inferred_shocks[t, j]
        state_parts[shock_name] = contrib

    if include_initial:
        initial = np.zeros((n_periods, n_states), dtype=np.float64)
        sum_t0 = np.zeros(n_states, dtype=np.float64)
        for values in state_parts.values():
            sum_t0 = sum_t0 + values[0]
        initial[0, :] = target_states[0] - sum_t0
        for t in range(1, n_periods):
            initial[t, :] = ss.T @ initial[t - 1]
        state_parts["initial"] = initial

    reconstructed_states = np.zeros((n_periods, n_states), dtype=np.float64)
    for values in state_parts.values():
        reconstructed_states = reconstructed_states + values

    if include_residual:
        residual = target_states - reconstructed_states
        state_parts["residual"] = residual
        reconstructed_states = reconstructed_states + residual

    obs_parts = {
        name: values @ ss.Z.T
        for name, values in state_parts.items()
    }
    reconstructed_obs = reconstructed_states @ ss.Z.T
    target_obs = target_states @ ss.Z.T

    if include_steady_state:
        steady_obs = np.tile(_obs_steady_state_vector(ss), (n_periods, 1))
        obs_parts["steady_state"] = steady_obs
        reconstructed_obs = reconstructed_obs + steady_obs
        target_obs = target_obs + steady_obs

    idx = sm.smoothed_states.index
    state_contrib_df = _build_multiindex_frame(
        state_parts,
        index=idx,
        columns=ss.state_names,
        first_level_name="component",
        second_level_name="state",
    )
    obs_contrib_df = _build_multiindex_frame(
        obs_parts,
        index=idx,
        columns=ss.obs_names,
        first_level_name="component",
        second_level_name="observable",
    )

    reconstructed_states_df = pd.DataFrame(
        reconstructed_states,
        index=idx,
        columns=ss.state_names,
    )
    reconstructed_obs_df = pd.DataFrame(
        reconstructed_obs,
        index=idx,
        columns=ss.obs_names,
    )
    target_states_df = sm.smoothed_states.copy()
    target_obs_df = pd.DataFrame(target_obs, index=idx, columns=ss.obs_names)
    shocks_df = pd.DataFrame(inferred_shocks, index=idx, columns=ss.shock_names)

    max_abs_state_error = float(np.max(np.abs(target_states - reconstructed_states)))
    max_abs_obs_error = float(np.max(np.abs(target_obs - reconstructed_obs)))

    return HistoricalDecompositionResult(
        state_contributions=state_contrib_df,
        observable_contributions=obs_contrib_df,
        reconstructed_states=reconstructed_states_df,
        reconstructed_observables=reconstructed_obs_df,
        target_states=target_states_df,
        target_observables=target_obs_df,
        inferred_shocks=shocks_df,
        components=list(state_parts.keys()),
        max_abs_state_error=max_abs_state_error,
        max_abs_obs_error=max_abs_obs_error,
    )

