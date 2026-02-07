"""Stochastic simulation for DSGE models.

Generates sample paths from the solved model given random shocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dsgekit.model.calibration import Calibration
    from dsgekit.solvers.linear import LinearSolution


@dataclass
class SimulationResult:
    """Results from stochastic simulation.

    Attributes:
        data: DataFrame with simulated paths (rows=periods, columns=variables)
        shocks: DataFrame with shock realizations
        n_periods: Number of periods simulated
        seed: Random seed used (if any)
    """

    data: pd.DataFrame
    shocks: pd.DataFrame
    n_periods: int
    seed: int | None = None

    def __getitem__(self, var_name: str) -> NDArray[np.float64]:
        """Get simulated path for a specific variable."""
        return self.data[var_name].values

    def plot(
        self,
        variables: list[str] | None = None,
        figsize: tuple[float, float] = (12, 6),
        title: str = "Simulation",
    ):
        """Plot simulated paths.

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
        fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)

        if n_vars == 1:
            axes = [axes]

        for ax, var in zip(axes, variables, strict=True):
            ax.plot(self.data.index, self.data[var], "b-", linewidth=0.8)
            ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
            ax.set_ylabel(var)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Period")
        fig.suptitle(title)
        fig.tight_layout()

        return fig


def simulate(
    solution: LinearSolution,
    calibration: Calibration,
    n_periods: int = 100,
    seed: int | None = None,
    initial_state: NDArray[np.float64] | None = None,
    burn_in: int = 0,
) -> SimulationResult:
    """Run stochastic simulation.

    Simulates the model:
        y_t = T * y_{t-1} + R * u_t

    where u_t ~ N(0, Σ) with Σ from calibration.

    Args:
        solution: LinearSolution from solver
        calibration: Calibration with shock covariances
        n_periods: Number of periods to simulate
        seed: Random seed for reproducibility
        initial_state: Initial values (default: zeros = steady state)
        burn_in: Periods to discard at start

    Returns:
        SimulationResult with simulated paths
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    T = solution.T
    R = solution.R
    n_vars = T.shape[0]
    n_shocks = R.shape[1]

    total_periods = n_periods + burn_in

    # Get shock covariance and Cholesky factor
    shock_cov = calibration.shock_cov_matrix(solution.shock_names)

    # Handle case where all shocks have zero variance
    if np.allclose(shock_cov, 0):
        shock_chol = np.zeros_like(shock_cov)
    else:
        shock_chol = np.linalg.cholesky(shock_cov)

    # Generate shocks: u_t = chol @ z_t where z_t ~ N(0, I)
    z = rng.standard_normal((total_periods, n_shocks))
    shocks = z @ shock_chol.T

    # Initialize
    if initial_state is None:
        y = np.zeros(n_vars)
    else:
        y = initial_state.copy()

    # Simulate
    paths = np.zeros((total_periods, n_vars))

    for t in range(total_periods):
        y = T @ y + R @ shocks[t, :]
        paths[t, :] = y

    # Discard burn-in
    paths = paths[burn_in:, :]
    shocks = shocks[burn_in:, :]

    # Create DataFrames
    data_df = pd.DataFrame(
        paths,
        index=range(n_periods),
        columns=solution.var_names,
    )
    data_df.index.name = "period"

    shocks_df = pd.DataFrame(
        shocks,
        index=range(n_periods),
        columns=solution.shock_names,
    )
    shocks_df.index.name = "period"

    return SimulationResult(
        data=data_df,
        shocks=shocks_df,
        n_periods=n_periods,
        seed=seed,
    )


def simulate_many(
    solution: LinearSolution,
    calibration: Calibration,
    n_simulations: int = 100,
    n_periods: int = 100,
    seed: int | None = None,
    burn_in: int = 100,
) -> list[SimulationResult]:
    """Run multiple independent simulations.

    Useful for Monte Carlo analysis.

    Args:
        solution: LinearSolution from solver
        calibration: Calibration with shock covariances
        n_simulations: Number of independent simulations
        n_periods: Periods per simulation
        seed: Base random seed
        burn_in: Burn-in periods per simulation

    Returns:
        List of SimulationResult
    """
    results = []

    for i in range(n_simulations):
        sim_seed = seed + i if seed is not None else None
        result = simulate(
            solution=solution,
            calibration=calibration,
            n_periods=n_periods,
            seed=sim_seed,
            burn_in=burn_in,
        )
        results.append(result)

    return results
