"""Welfare metrics from simulated or deterministic paths.

Provides compact, reproducible welfare summaries based on:
- period utility (linear term minus quadratic loss)
- discounted and non-discounted aggregates
- scenario comparisons
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

WelfareMetric = Literal[
    "discounted_utility",
    "discounted_loss",
    "mean_utility",
    "mean_loss",
]


def _as_dataframe(data: pd.DataFrame | object) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    maybe_data = data.data if hasattr(data, "data") else None
    if isinstance(maybe_data, pd.DataFrame):
        return maybe_data
    raise TypeError("data must be a pandas DataFrame or an object with a pandas .data DataFrame")


def _resolve_variables(
    data: pd.DataFrame,
    *,
    variables: list[str] | None,
    targets: dict[str, float] | None,
    linear_utility_weights: dict[str, float] | None,
    quadratic_loss_weights: dict[str, float] | None,
    loss_matrix: NDArray[np.float64] | pd.DataFrame | None,
) -> list[str]:
    if variables is not None:
        resolved = list(variables)
    elif isinstance(loss_matrix, pd.DataFrame):
        resolved = list(loss_matrix.index)
    elif quadratic_loss_weights:
        resolved = list(quadratic_loss_weights)
    elif linear_utility_weights:
        resolved = list(linear_utility_weights)
    elif targets:
        resolved = list(targets)
    else:
        resolved = list(data.columns)

    if not resolved:
        raise ValueError("variables cannot be empty")

    unknown = [name for name in resolved if name not in data.columns]
    if unknown:
        raise ValueError(f"variables not found in data columns: {unknown}")

    return resolved


def _build_loss_matrix(
    variables: list[str],
    *,
    quadratic_loss_weights: dict[str, float] | None,
    loss_matrix: NDArray[np.float64] | pd.DataFrame | None,
) -> NDArray[np.float64]:
    n = len(variables)
    if loss_matrix is None:
        if quadratic_loss_weights is None:
            mat = np.eye(n, dtype=np.float64)
        else:
            mat = np.zeros((n, n), dtype=np.float64)
            for i, var_name in enumerate(variables):
                weight = float(quadratic_loss_weights.get(var_name, 0.0))
                if weight < 0.0:
                    raise ValueError(f"quadratic loss weight must be >= 0 for '{var_name}'")
                mat[i, i] = weight
    elif isinstance(loss_matrix, pd.DataFrame):
        missing_rows = [name for name in variables if name not in loss_matrix.index]
        missing_cols = [name for name in variables if name not in loss_matrix.columns]
        if missing_rows or missing_cols:
            raise ValueError(
                "loss_matrix DataFrame must contain all variables as both index and columns"
            )
        mat = loss_matrix.loc[variables, variables].to_numpy(dtype=np.float64, copy=True)
    else:
        mat = np.asarray(loss_matrix, dtype=np.float64)
        if mat.shape != (n, n):
            raise ValueError(f"loss_matrix must have shape {(n, n)}, got {mat.shape}")

    if not np.all(np.isfinite(mat)):
        raise ValueError("loss_matrix must contain only finite values")

    # Use the symmetric part to keep x'Qx well-defined even if caller passes
    # slightly non-symmetric numeric input.
    return 0.5 * (mat + mat.T)


def _discount_vector(n_periods: int, discount: float) -> NDArray[np.float64]:
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1")
    if not np.isfinite(discount) or discount <= 0.0 or discount > 1.0:
        raise ValueError(f"discount must be in (0, 1], got {discount}")
    return np.power(discount, np.arange(n_periods, dtype=np.float64))


@dataclass
class WelfareResult:
    """Welfare metrics for one simulated path or deterministic scenario."""

    name: str
    variables: list[str]
    n_periods: int
    discount: float
    mean_utility: float
    mean_loss: float
    discounted_utility: float
    discounted_loss: float
    discounted_mean_utility: float
    discounted_mean_loss: float
    utility_series: pd.Series
    loss_series: pd.Series

    def summary(self) -> str:
        lines = [
            f"Welfare Summary ({self.name})",
            "=" * 50,
            f"  Periods:                  {self.n_periods}",
            f"  Discount factor:          {self.discount:.4f}",
            f"  Mean utility:             {self.mean_utility:.6f}",
            f"  Mean quadratic loss:      {self.mean_loss:.6f}",
            f"  Discounted utility sum:   {self.discounted_utility:.6f}",
            f"  Discounted loss sum:      {self.discounted_loss:.6f}",
            f"  Discounted mean utility:  {self.discounted_mean_utility:.6f}",
            f"  Discounted mean loss:     {self.discounted_mean_loss:.6f}",
        ]
        return "\n".join(lines)


@dataclass
class WelfareComparison:
    """Comparison table across scenarios."""

    metric: WelfareMetric
    baseline: str
    table: pd.DataFrame

    def summary(self) -> str:
        lines = [
            f"Welfare Comparison ({self.metric})",
            "=" * 50,
            f"  Baseline: {self.baseline}",
            "",
            self.table.to_string(float_format=lambda x: f"{x:.6f}"),
        ]
        return "\n".join(lines)


def evaluate_welfare(
    data: pd.DataFrame | object,
    *,
    variables: list[str] | None = None,
    targets: dict[str, float] | None = None,
    linear_utility_weights: dict[str, float] | None = None,
    quadratic_loss_weights: dict[str, float] | None = None,
    loss_matrix: NDArray[np.float64] | pd.DataFrame | None = None,
    utility_constant: float = 0.0,
    discount: float = 0.99,
    name: str = "scenario",
) -> WelfareResult:
    """Compute utility/loss welfare metrics for one scenario path.

    Period utility is:
        u_t = c + a' x_t - 0.5 * x_t' Q x_t

    where x_t are variable deviations from targets.

    Quadratic loss is:
        L_t = x_t' Q x_t
    """
    data_df = _as_dataframe(data)
    if data_df.empty:
        raise ValueError("data must contain at least one period")

    resolved_vars = _resolve_variables(
        data_df,
        variables=variables,
        targets=targets,
        linear_utility_weights=linear_utility_weights,
        quadratic_loss_weights=quadratic_loss_weights,
        loss_matrix=loss_matrix,
    )
    q_mat = _build_loss_matrix(
        resolved_vars,
        quadratic_loss_weights=quadratic_loss_weights,
        loss_matrix=loss_matrix,
    )

    x = data_df[resolved_vars].to_numpy(dtype=np.float64, copy=True)
    if targets:
        target_vec = np.array([float(targets.get(var_name, 0.0)) for var_name in resolved_vars])
        x = x - target_vec

    linear_weights = np.array(
        [
            0.0 if linear_utility_weights is None else float(linear_utility_weights.get(var_name, 0.0))
            for var_name in resolved_vars
        ],
        dtype=np.float64,
    )

    loss_values = np.einsum("ti,ij,tj->t", x, q_mat, x)
    utility_values = float(utility_constant) + (x @ linear_weights) - 0.5 * loss_values

    weights = _discount_vector(len(data_df), discount=discount)
    weight_sum = float(np.sum(weights))

    discounted_utility = float(np.dot(weights, utility_values))
    discounted_loss = float(np.dot(weights, loss_values))

    utility_series = pd.Series(utility_values, index=data_df.index, name="utility")
    loss_series = pd.Series(loss_values, index=data_df.index, name="quadratic_loss")

    return WelfareResult(
        name=name,
        variables=resolved_vars,
        n_periods=int(len(data_df)),
        discount=float(discount),
        mean_utility=float(np.mean(utility_values)),
        mean_loss=float(np.mean(loss_values)),
        discounted_utility=discounted_utility,
        discounted_loss=discounted_loss,
        discounted_mean_utility=discounted_utility / weight_sum,
        discounted_mean_loss=discounted_loss / weight_sum,
        utility_series=utility_series,
        loss_series=loss_series,
    )


def compare_welfare(
    scenarios: dict[str, WelfareResult],
    *,
    baseline: str | None = None,
    metric: WelfareMetric = "discounted_utility",
) -> WelfareComparison:
    """Compare welfare metrics across scenarios.

    Utility metrics are ranked descending (higher is better).
    Loss metrics are ranked ascending (lower is better).
    """
    if not scenarios:
        raise ValueError("scenarios must contain at least one WelfareResult")

    names = list(scenarios)
    base_name = names[0] if baseline is None else baseline
    if base_name not in scenarios:
        raise ValueError(f"baseline '{base_name}' not found in scenarios")

    values = {name: float(getattr(result, metric)) for name, result in scenarios.items()}
    base_value = values[base_name]
    is_utility = metric in {"discounted_utility", "mean_utility"}

    rows: list[dict[str, float | int | str]] = []
    for name, value in values.items():
        delta_vs_baseline = value - base_value
        improvement = delta_vs_baseline if is_utility else -delta_vs_baseline
        rows.append(
            {
                "scenario": name,
                "value": value,
                "delta_vs_baseline": delta_vs_baseline,
                "improvement_vs_baseline": improvement,
            }
        )

    table = pd.DataFrame(rows).set_index("scenario")
    table = table.sort_values("value", ascending=not is_utility)
    table["rank"] = np.arange(1, len(table) + 1, dtype=int)
    table = table[["rank", "value", "delta_vs_baseline", "improvement_vs_baseline"]]

    return WelfareComparison(metric=metric, baseline=base_name, table=table)
