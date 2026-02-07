"""Plotting functions for DSGE model results.

Provides functions to create visualizations for:
- Impulse Response Functions (IRFs)
- Forecast Error Variance Decomposition (FEVD)
- Simulations
- Autocorrelations
- Eigenvalues (unit circle)
- Moments comparison

Requires matplotlib. Install with: pip install dsgekit[plot]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dsgekit.simulate.irf import IRFResult
    from dsgekit.simulate.moments import MomentsResult
    from dsgekit.simulate.simulate import SimulationResult
    from dsgekit.solvers.linear import LinearSolution


def _get_matplotlib():
    """Get matplotlib.pyplot, raising informative error if not available."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as err:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install dsgekit[plot] or pip install matplotlib"
        ) from err


def _get_figure_and_axes(
    n_plots: int,
    n_cols: int = 3,
    figsize: tuple[float, float] | None = None,
    sharex: bool = True,
    sharey: bool = False,
):
    """Create figure with subplot grid."""
    plt = _get_matplotlib()

    n_rows = (n_plots + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, sharex=sharex, sharey=sharey, squeeze=False
    )
    axes_flat = axes.flatten()

    return fig, axes_flat, n_rows, n_cols


# ============================================================================
# IRF Plots
# ============================================================================


def plot_irf(
    irf_result: IRFResult,
    variables: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    n_cols: int = 3,
    title: str | None = None,
    color: str = "steelblue",
    linewidth: float = 1.5,
    show_zero: bool = True,
    grid: bool = True,
) -> Any:
    """Plot impulse response functions.

    Args:
        irf_result: IRFResult from irf computation
        variables: Variables to plot (None = all)
        figsize: Figure size (width, height)
        n_cols: Number of columns in subplot grid
        title: Overall figure title
        color: Line color
        linewidth: Line width
        show_zero: Show horizontal zero line
        grid: Show grid

    Returns:
        matplotlib Figure
    """
    _get_matplotlib()

    if variables is None:
        variables = list(irf_result.data.columns)

    n_vars = len(variables)
    fig, axes, n_rows, _ = _get_figure_and_axes(n_vars, n_cols, figsize)

    for i, var in enumerate(variables):
        ax = axes[i]
        ax.plot(
            irf_result.data.index,
            irf_result.data[var],
            color=color,
            linewidth=linewidth,
        )
        if show_zero:
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_title(var)
        ax.set_xlabel("Period")
        if grid:
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    if title is None:
        title = f"IRF to {irf_result.shock_name} shock"
    fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_irf_comparison(
    irfs: dict[str, IRFResult],
    variables: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    n_cols: int = 3,
    title: str | None = None,
    colors: list[str] | None = None,
    linewidth: float = 1.5,
    show_zero: bool = True,
    grid: bool = True,
    legend: bool = True,
) -> Any:
    """Plot IRF comparison across multiple shocks.

    Args:
        irfs: Dictionary of shock_name -> IRFResult
        variables: Variables to plot (None = all)
        figsize: Figure size
        n_cols: Number of columns in subplot grid
        title: Overall figure title
        colors: Colors for each shock
        linewidth: Line width
        show_zero: Show horizontal zero line
        grid: Show grid
        legend: Show legend

    Returns:
        matplotlib Figure
    """
    plt = _get_matplotlib()

    # Get all variables if not specified
    if variables is None:
        first_irf = next(iter(irfs.values()))
        variables = list(first_irf.data.columns)

    shock_names = list(irfs.keys())
    n_shocks = len(shock_names)
    n_vars = len(variables)

    if colors is None:
        cmap = plt.cm.get_cmap("tab10")
        colors = [cmap(i) for i in range(n_shocks)]

    fig, axes, _, _ = _get_figure_and_axes(n_vars, n_cols, figsize)

    for i, var in enumerate(variables):
        ax = axes[i]
        for j, (shock_name, irf_result) in enumerate(irfs.items()):
            ax.plot(
                irf_result.data.index,
                irf_result.data[var],
                color=colors[j],
                linewidth=linewidth,
                label=shock_name,
            )
        if show_zero:
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_title(var)
        ax.set_xlabel("Period")
        if grid:
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    if legend and n_shocks > 1:
        # Add legend to first subplot
        axes[0].legend(loc="best", fontsize="small")

    if title is None:
        title = "IRF Comparison"
    fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_irf_single_variable(
    irfs: dict[str, IRFResult],
    variable: str,
    figsize: tuple[float, float] = (8, 5),
    title: str | None = None,
    colors: list[str] | None = None,
    linewidth: float = 1.5,
    show_zero: bool = True,
    grid: bool = True,
) -> Any:
    """Plot IRFs for a single variable from multiple shocks.

    Args:
        irfs: Dictionary of shock_name -> IRFResult
        variable: Variable to plot
        figsize: Figure size
        title: Plot title
        colors: Colors for each shock
        linewidth: Line width
        show_zero: Show horizontal zero line
        grid: Show grid

    Returns:
        matplotlib Figure
    """
    plt = _get_matplotlib()

    shock_names = list(irfs.keys())
    n_shocks = len(shock_names)

    if colors is None:
        cmap = plt.cm.get_cmap("tab10")
        colors = [cmap(i) for i in range(n_shocks)]

    fig, ax = plt.subplots(figsize=figsize)

    for j, (shock_name, irf_result) in enumerate(irfs.items()):
        ax.plot(
            irf_result.data.index,
            irf_result.data[variable],
            color=colors[j],
            linewidth=linewidth,
            label=shock_name,
        )

    if show_zero:
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Period")
    ax.set_ylabel(variable)
    ax.legend(loc="best")

    if grid:
        ax.grid(True, alpha=0.3)

    if title is None:
        title = f"IRF of {variable}"
    ax.set_title(title)

    fig.tight_layout()
    return fig


# ============================================================================
# FEVD Plots
# ============================================================================


def plot_fevd(
    fevd_df: pd.DataFrame,
    variable: str,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    colors: list[str] | None = None,
    kind: str = "area",
) -> Any:
    """Plot Forecast Error Variance Decomposition.

    Args:
        fevd_df: DataFrame from compute_fevd (MultiIndex: horizon, variable)
        variable: Variable to plot FEVD for
        figsize: Figure size
        title: Plot title
        colors: Colors for each shock
        kind: Plot type: 'area' (stacked area) or 'bar' (stacked bar)

    Returns:
        matplotlib Figure
    """
    plt = _get_matplotlib()

    # Extract data for variable
    df = fevd_df.xs(variable, level="variable")

    fig, ax = plt.subplots(figsize=figsize)

    if kind == "area":
        df.plot.area(ax=ax, alpha=0.8, color=colors)
    elif kind == "bar":
        df.plot.bar(ax=ax, stacked=True, color=colors, width=0.8)
    else:
        raise ValueError(f"Unknown plot kind: {kind}. Use 'area' or 'bar'.")

    ax.set_xlabel("Horizon")
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="Shock", loc="best")

    if title is None:
        title = f"FEVD of {variable}"
    ax.set_title(title)

    fig.tight_layout()
    return fig


def plot_fevd_all(
    fevd_df: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
    n_cols: int = 2,
    title: str | None = None,
    colors: list[str] | None = None,
    kind: str = "area",
) -> Any:
    """Plot FEVD for all variables.

    Args:
        fevd_df: DataFrame from compute_fevd
        figsize: Figure size
        n_cols: Number of columns in subplot grid
        title: Overall figure title
        colors: Colors for each shock
        kind: Plot type: 'area' or 'bar'

    Returns:
        matplotlib Figure
    """
    # Get unique variables
    variables = fevd_df.index.get_level_values("variable").unique().tolist()
    n_vars = len(variables)

    fig, axes, _, _ = _get_figure_and_axes(n_vars, n_cols, figsize, sharex=False)

    for i, var in enumerate(variables):
        ax = axes[i]
        df = fevd_df.xs(var, level="variable")

        if kind == "area":
            df.plot.area(ax=ax, alpha=0.8, color=colors, legend=False)
        else:
            df.plot.bar(ax=ax, stacked=True, color=colors, width=0.8, legend=False)

        ax.set_title(var)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("%")
        ax.set_ylim(0, 100)

    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    # Add single legend
    shock_names = fevd_df.columns.tolist()
    fig.legend(
        shock_names,
        title="Shock",
        loc="upper right",
        bbox_to_anchor=(1, 1),
    )

    if title is None:
        title = "Forecast Error Variance Decomposition"
    fig.suptitle(title)
    fig.tight_layout()

    return fig


# ============================================================================
# Simulation Plots
# ============================================================================


def plot_simulation(
    simulation: SimulationResult,
    variables: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    n_cols: int = 2,
    title: str | None = None,
    color: str = "steelblue",
    linewidth: float = 0.8,
    show_mean: bool = True,
    grid: bool = True,
) -> Any:
    """Plot simulated time series.

    Args:
        simulation: SimulationResult from simulate
        variables: Variables to plot (None = all)
        figsize: Figure size
        n_cols: Number of columns in subplot grid
        title: Overall figure title
        color: Line color
        linewidth: Line width
        show_mean: Show horizontal line at mean (usually 0)
        grid: Show grid

    Returns:
        matplotlib Figure
    """
    if variables is None:
        variables = list(simulation.data.columns)

    n_vars = len(variables)
    fig, axes, _, _ = _get_figure_and_axes(n_vars, n_cols, figsize, sharex=True)

    for i, var in enumerate(variables):
        ax = axes[i]
        ax.plot(
            simulation.data.index,
            simulation.data[var],
            color=color,
            linewidth=linewidth,
        )
        if show_mean:
            ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_title(var)
        ax.set_xlabel("Period")
        if grid:
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    if title is None:
        title = "Simulation"
    fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_simulation_distribution(
    simulation: SimulationResult,
    variables: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    n_cols: int = 3,
    title: str | None = None,
    bins: int = 30,
    color: str = "steelblue",
    kde: bool = True,
) -> Any:
    """Plot distribution of simulated values.

    Args:
        simulation: SimulationResult
        variables: Variables to plot
        figsize: Figure size
        n_cols: Number of columns
        title: Overall figure title
        bins: Number of histogram bins
        color: Histogram color
        kde: Overlay kernel density estimate (requires scipy)

    Returns:
        matplotlib Figure
    """
    if variables is None:
        variables = list(simulation.data.columns)

    n_vars = len(variables)
    fig, axes, _, _ = _get_figure_and_axes(n_vars, n_cols, figsize, sharex=False)

    for i, var in enumerate(variables):
        ax = axes[i]
        data = simulation.data[var].dropna()

        ax.hist(data, bins=bins, color=color, alpha=0.7, density=True)

        if kde:
            try:
                from scipy import stats

                kde_x = np.linspace(data.min(), data.max(), 100)
                kde_y = stats.gaussian_kde(data)(kde_x)
                ax.plot(kde_x, kde_y, color="darkred", linewidth=1.5)
            except ImportError:
                pass  # scipy not available, skip KDE

        ax.set_title(var)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    if title is None:
        title = "Distribution of Simulated Values"
    fig.suptitle(title)
    fig.tight_layout()

    return fig


# ============================================================================
# Autocorrelation Plots
# ============================================================================


def plot_autocorrelation(
    moments: MomentsResult,
    variables: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    n_cols: int = 3,
    title: str | None = None,
    color: str = "steelblue",
    show_confidence: bool = True,
    confidence_level: float = 0.95,
    n_obs: int | None = None,
) -> Any:
    """Plot autocorrelation functions.

    Args:
        moments: MomentsResult with autocorrelation data
        variables: Variables to plot
        figsize: Figure size
        n_cols: Number of columns
        title: Overall figure title
        color: Bar color
        show_confidence: Show confidence bands
        confidence_level: Confidence level for bands
        n_obs: Number of observations (for confidence bands)

    Returns:
        matplotlib Figure
    """
    if moments.autocorrelation is None:
        raise ValueError("MomentsResult has no autocorrelation data")

    acf_df = moments.autocorrelation

    if variables is None:
        variables = list(acf_df.columns)

    n_vars = len(variables)
    fig, axes, _, _ = _get_figure_and_axes(n_vars, n_cols, figsize, sharex=True)

    for i, var in enumerate(variables):
        ax = axes[i]
        lags = acf_df.index.values
        acf_values = acf_df[var].values

        ax.bar(lags, acf_values, color=color, alpha=0.7, width=0.6)
        ax.axhline(y=0, color="black", linewidth=0.5)

        if show_confidence and n_obs is not None:
            from scipy import stats

            z = stats.norm.ppf((1 + confidence_level) / 2)
            ci = z / np.sqrt(n_obs)
            ax.axhline(y=ci, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.axhline(y=-ci, color="red", linestyle="--", linewidth=0.5, alpha=0.7)

        ax.set_title(var)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_ylim(-1.1, 1.1)

    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    if title is None:
        title = "Autocorrelation Functions"
    fig.suptitle(title)
    fig.tight_layout()

    return fig


# ============================================================================
# Eigenvalue Plots
# ============================================================================


def plot_eigenvalues(
    solution: LinearSolution,
    figsize: tuple[float, float] = (7, 7),
    title: str | None = None,
    stable_color: str = "steelblue",
    unstable_color: str = "crimson",
    marker_size: int = 80,
    show_unit_circle: bool = True,
) -> Any:
    """Plot eigenvalues in the complex plane with unit circle.

    Args:
        solution: LinearSolution with eigenvalues
        figsize: Figure size
        title: Plot title
        stable_color: Color for stable eigenvalues (|λ| < 1)
        unstable_color: Color for unstable eigenvalues (|λ| >= 1)
        marker_size: Marker size
        show_unit_circle: Show unit circle

    Returns:
        matplotlib Figure
    """
    plt = _get_matplotlib()

    eigenvalues = solution.eigenvalues
    finite_mask = np.isfinite(eigenvalues)
    finite_eigs = eigenvalues[finite_mask]

    fig, ax = plt.subplots(figsize=figsize)

    # Unit circle
    if show_unit_circle:
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=1, alpha=0.5)

    # Classify eigenvalues
    moduli = np.abs(finite_eigs)
    stable_mask = moduli < 1.0

    # Plot stable eigenvalues
    stable_eigs = finite_eigs[stable_mask]
    if len(stable_eigs) > 0:
        ax.scatter(
            np.real(stable_eigs),
            np.imag(stable_eigs),
            c=stable_color,
            s=marker_size,
            marker="o",
            label=f"Stable ({len(stable_eigs)})",
            zorder=5,
        )

    # Plot unstable eigenvalues
    unstable_eigs = finite_eigs[~stable_mask]
    if len(unstable_eigs) > 0:
        ax.scatter(
            np.real(unstable_eigs),
            np.imag(unstable_eigs),
            c=unstable_color,
            s=marker_size,
            marker="x",
            label=f"Unstable ({len(unstable_eigs)})",
            zorder=5,
        )

    # Axes
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Set limits to show unit circle
    max_mod = max(1.2, np.max(moduli) * 1.1) if len(moduli) > 0 else 1.2
    ax.set_xlim(-max_mod, max_mod)
    ax.set_ylim(-max_mod, max_mod)

    if title is None:
        status = "DETERMINATE" if solution.is_determinate() else "INDETERMINATE"
        title = f"Eigenvalues ({status})"
    ax.set_title(title)

    fig.tight_layout()
    return fig


# ============================================================================
# Utility Functions
# ============================================================================


def save_figure(
    fig: Any,
    filename: str,
    dpi: int = 150,
    transparent: bool = False,
) -> None:
    """Save figure to file.

    Args:
        fig: matplotlib Figure
        filename: Output filename (extension determines format)
        dpi: Resolution in dots per inch
        transparent: Transparent background
    """
    fig.savefig(filename, dpi=dpi, transparent=transparent, bbox_inches="tight")


def show_figures() -> None:
    """Display all figures (calls plt.show())."""
    plt = _get_matplotlib()
    plt.show()


def close_figures() -> None:
    """Close all figures."""
    plt = _get_matplotlib()
    plt.close("all")
