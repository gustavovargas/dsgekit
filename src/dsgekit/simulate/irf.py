"""Impulse Response Functions (IRFs) for DSGE models.

Computes the dynamic response of model variables to a one-time shock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dsgekit.solvers.linear import LinearSolution


@dataclass
class IRFResult:
    """Results from impulse response function computation.

    Attributes:
        data: DataFrame with IRFs (rows=periods, columns=variables)
        shock_name: Name of the shock
        shock_size: Size of the shock (usually 1 std dev)
        periods: Number of periods computed
    """

    data: pd.DataFrame
    shock_name: str
    shock_size: float
    periods: int

    def __getitem__(self, var_name: str) -> NDArray[np.float64]:
        """Get IRF for a specific variable."""
        return self.data[var_name].values

    def plot(
        self,
        variables: list[str] | None = None,
        figsize: tuple[float, float] = (10, 6),
        title: str | None = None,
    ):
        """Plot IRFs.

        Args:
            variables: Variables to plot (default: all)
            figsize: Figure size
            title: Plot title

        Returns:
            matplotlib Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install dsgekit[plot]"
            ) from err

        if variables is None:
            variables = list(self.data.columns)

        n_vars = len(variables)
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for i, var in enumerate(variables):
            ax = axes[i]
            ax.plot(self.data.index, self.data[var], "b-", linewidth=1.5)
            ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
            ax.set_title(var)
            ax.set_xlabel("Period")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(variables), len(axes)):
            axes[i].set_visible(False)

        if title is None:
            title = f"IRF to {self.shock_name} shock"
        fig.suptitle(title)
        fig.tight_layout()

        return fig


def irf(
    solution: LinearSolution,
    shock_name: str,
    periods: int = 40,
    shock_size: float = 1.0,
) -> IRFResult:
    """Compute impulse response functions.

    Args:
        solution: LinearSolution from solver
        shock_name: Name of the shock to impulse
        periods: Number of periods to compute
        shock_size: Size of shock (default: 1 standard deviation)

    Returns:
        IRFResult with IRFs for all variables
    """
    # Find shock index
    try:
        shock_idx = solution.shock_names.index(shock_name)
    except ValueError as err:
        raise ValueError(
            f"Shock '{shock_name}' not found. "
            f"Available: {solution.shock_names}"
        ) from err

    T = solution.T
    R = solution.R
    n_vars = T.shape[0]

    # Initialize storage
    responses = np.zeros((periods, n_vars))

    # Initial shock
    shock_vec = np.zeros(len(solution.shock_names))
    shock_vec[shock_idx] = shock_size

    # Period 0: impact
    y = R @ shock_vec
    responses[0, :] = y

    # Subsequent periods: propagation through T
    for t in range(1, periods):
        y = T @ y
        responses[t, :] = y

    # Create DataFrame
    df = pd.DataFrame(
        responses,
        index=range(periods),
        columns=solution.var_names,
    )
    df.index.name = "period"

    return IRFResult(
        data=df,
        shock_name=shock_name,
        shock_size=shock_size,
        periods=periods,
    )


def irf_all_shocks(
    solution: LinearSolution,
    periods: int = 40,
    shock_size: float = 1.0,
) -> dict[str, IRFResult]:
    """Compute IRFs for all shocks.

    Args:
        solution: LinearSolution from solver
        periods: Number of periods
        shock_size: Size of each shock

    Returns:
        Dictionary mapping shock_name -> IRFResult
    """
    results = {}
    for shock_name in solution.shock_names:
        results[shock_name] = irf(
            solution,
            shock_name,
            periods=periods,
            shock_size=shock_size,
        )
    return results


def irf_to_dataframe(
    irfs: dict[str, IRFResult],
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Convert multiple IRFs to a single DataFrame.

    Args:
        irfs: Dictionary of IRFResult from irf_all_shocks
        variables: Variables to include (default: all)

    Returns:
        DataFrame with MultiIndex columns (shock, variable)
    """
    dfs = []
    for shock_name, irf_result in irfs.items():
        df = irf_result.data.copy()
        if variables is not None:
            df = df[variables]
        df.columns = pd.MultiIndex.from_product(
            [[shock_name], df.columns],
            names=["shock", "variable"],
        )
        dfs.append(df)

    return pd.concat(dfs, axis=1)
