"""Table generation for DSGE model results.

Provides functions to create formatted tables for:
- Parameters
- Steady state values
- Shock configuration
- Moments (variance, std dev)
- Correlation matrices
- Autocorrelations
- IRFs
- FEVD
- Eigenvalues
- Model summary
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dsgekit.model.calibration import Calibration
    from dsgekit.model.ir import ModelIR
    from dsgekit.model.steady_state import SteadyState
    from dsgekit.simulate.irf import IRFResult
    from dsgekit.simulate.moments import MomentsResult
    from dsgekit.solvers.linear import LinearSolution


class TableFormat(Enum):
    """Output format for tables."""

    PLAIN = "plain"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"


@dataclass
class TableFormatter:
    """Configuration for table formatting.

    Attributes:
        format: Output format (plain, markdown, latex, html)
        precision: Number of decimal places
        max_rows: Maximum rows to display (None = all)
        title: Optional table title
    """

    format: TableFormat = TableFormat.PLAIN
    precision: int = 4
    max_rows: int | None = None
    title: str | None = None

    def format_df(self, df: pd.DataFrame, title: str | None = None) -> str:
        """Format a DataFrame according to settings."""
        title = title or self.title

        if self.max_rows is not None and len(df) > self.max_rows:
            df = df.head(self.max_rows)

        if self.format == TableFormat.PLAIN:
            return self._format_plain(df, title)
        elif self.format == TableFormat.MARKDOWN:
            return self._format_markdown(df, title)
        elif self.format == TableFormat.LATEX:
            return self._format_latex(df, title)
        elif self.format == TableFormat.HTML:
            return self._format_html(df, title)
        else:
            return self._format_plain(df, title)

    def _format_plain(self, df: pd.DataFrame, title: str | None) -> str:
        """Format as plain text table."""
        lines = []
        if title:
            lines.append(title)
            lines.append("=" * len(title))

        # Format with fixed precision
        formatted = df.to_string(
            float_format=lambda x: f"{x:.{self.precision}f}" if pd.notna(x) else "",
        )
        lines.append(formatted)
        return "\n".join(lines)

    def _format_markdown(self, df: pd.DataFrame, title: str | None) -> str:
        """Format as Markdown table."""
        lines = []
        if title:
            lines.append(f"### {title}")
            lines.append("")

        # Reset index if named
        if df.index.name:
            df = df.reset_index()

        # Try to use to_markdown if tabulate is available
        try:
            lines.append(df.to_markdown(floatfmt=f".{self.precision}f"))
        except ImportError:
            # Fallback: generate markdown manually
            lines.append(self._manual_markdown(df))

        return "\n".join(lines)

    def _manual_markdown(self, df: pd.DataFrame) -> str:
        """Generate markdown table without tabulate."""
        lines = []

        # Header
        headers = [str(col) for col in df.columns]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Rows
        for _, row in df.iterrows():
            cells = []
            for val in row:
                if isinstance(val, float):
                    cells.append(f"{val:.{self.precision}f}")
                else:
                    cells.append(str(val))
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def _format_latex(self, df: pd.DataFrame, title: str | None) -> str:
        """Format as LaTeX table."""
        latex = df.to_latex(
            float_format=f"%.{self.precision}f",
            caption=title,
            escape=False,
        )
        return latex

    def _format_html(self, df: pd.DataFrame, title: str | None) -> str:
        """Format as HTML table."""
        html_parts = []
        if title:
            html_parts.append(f"<h3>{title}</h3>")

        html_parts.append(
            df.to_html(
                float_format=lambda x: f"{x:.{self.precision}f}",
                classes=["dsgekit-table"],
            )
        )
        return "\n".join(html_parts)


# Default formatter
_default_formatter = TableFormatter()


def parameters_table(
    calibration: Calibration,
    formatter: TableFormatter | None = None,
) -> pd.DataFrame:
    """Create table of parameter values.

    Args:
        calibration: Model calibration
        formatter: Optional formatter for string output

    Returns:
        DataFrame with parameter names and values
    """
    data = {
        "Parameter": list(calibration.parameters.keys()),
        "Value": list(calibration.parameters.values()),
    }
    df = pd.DataFrame(data)
    df = df.sort_values("Parameter").reset_index(drop=True)
    return df


def steady_state_table(
    steady_state: SteadyState,
    formatter: TableFormatter | None = None,
) -> pd.DataFrame:
    """Create table of steady state values.

    Args:
        steady_state: Steady state solution
        formatter: Optional formatter

    Returns:
        DataFrame with variable names and steady state values
    """
    data = {
        "Variable": list(steady_state.values.keys()),
        "Steady State": list(steady_state.values.values()),
    }

    # Add residuals if available
    if steady_state.residuals:
        residuals = [
            steady_state.residuals.get(f"eq_{i+1}", 0.0)
            for i in range(len(steady_state.values))
        ]
        # Match residuals by order if names don't match
        if len(residuals) == len(steady_state.values):
            data["Residual"] = residuals

    df = pd.DataFrame(data)
    df = df.sort_values("Variable").reset_index(drop=True)
    return df


def shocks_table(
    calibration: Calibration,
    formatter: TableFormatter | None = None,
) -> pd.DataFrame:
    """Create table of shock configuration.

    Args:
        calibration: Model calibration
        formatter: Optional formatter

    Returns:
        DataFrame with shock names, std dev, and variance
    """
    shocks = list(calibration.shock_stderr.keys())
    stderrs = [calibration.shock_stderr[s] for s in shocks]
    variances = [s**2 for s in stderrs]

    df = pd.DataFrame(
        {
            "Shock": shocks,
            "Std.Dev.": stderrs,
            "Variance": variances,
        }
    )
    df = df.sort_values("Shock").reset_index(drop=True)
    return df


def moments_table(
    moments: MomentsResult,
    formatter: TableFormatter | None = None,
) -> pd.DataFrame:
    """Create table of unconditional moments.

    Args:
        moments: MomentsResult from moments computation
        formatter: Optional formatter

    Returns:
        DataFrame with variable, std dev, variance
    """
    variables = list(moments.variance.keys())

    df = pd.DataFrame(
        {
            "Variable": variables,
            "Std.Dev.": [moments.std_dev[v] for v in variables],
            "Variance": [moments.variance[v] for v in variables],
        }
    )
    return df


def correlation_table(
    moments: MomentsResult,
    formatter: TableFormatter | None = None,
) -> pd.DataFrame:
    """Create correlation matrix table.

    Args:
        moments: MomentsResult
        formatter: Optional formatter

    Returns:
        DataFrame with correlation matrix
    """
    return moments.correlation.copy()


def autocorrelation_table(
    moments: MomentsResult,
    max_lag: int | None = None,
    formatter: TableFormatter | None = None,
) -> pd.DataFrame:
    """Create autocorrelation table.

    Args:
        moments: MomentsResult
        max_lag: Maximum lag to include (None = all)
        formatter: Optional formatter

    Returns:
        DataFrame with autocorrelations (rows=lags, columns=variables)
    """
    if moments.autocorrelation is None:
        raise ValueError("MomentsResult has no autocorrelation data")

    df = moments.autocorrelation.copy()
    df.index.name = "Lag"

    if max_lag is not None:
        df = df.loc[:max_lag]

    return df


def irf_table(
    irf_result: IRFResult,
    variables: list[str] | None = None,
    periods: list[int] | None = None,
    formatter: TableFormatter | None = None,
) -> pd.DataFrame:
    """Create IRF table.

    Args:
        irf_result: IRFResult from irf computation
        variables: Variables to include (None = all)
        periods: Periods to include (None = all)
        formatter: Optional formatter

    Returns:
        DataFrame with IRFs (rows=periods, columns=variables)
    """
    df = irf_result.data.copy()

    if variables is not None:
        df = df[variables]

    if periods is not None:
        df = df.loc[periods]

    return df


def irf_comparison_table(
    irfs: dict[str, IRFResult],
    variable: str,
    periods: list[int] | None = None,
    formatter: TableFormatter | None = None,
) -> pd.DataFrame:
    """Create IRF comparison table across shocks for a single variable.

    Args:
        irfs: Dict of shock_name -> IRFResult
        variable: Variable to compare
        periods: Periods to include (None = all)
        formatter: Optional formatter

    Returns:
        DataFrame with columns=shocks, rows=periods
    """
    data = {}
    for shock_name, irf_result in irfs.items():
        data[shock_name] = irf_result.data[variable]

    df = pd.DataFrame(data)
    df.index.name = "Period"

    if periods is not None:
        df = df.loc[periods]

    return df


def fevd_table(
    fevd_df: pd.DataFrame,
    variable: str | None = None,
    horizons: list[int] | None = None,
    formatter: TableFormatter | None = None,
) -> pd.DataFrame:
    """Create FEVD table.

    Args:
        fevd_df: DataFrame from compute_fevd
        variable: Single variable to show (None = all)
        horizons: Horizons to include (None = all)
        formatter: Optional formatter

    Returns:
        DataFrame with FEVD percentages
    """
    df = fevd_df.copy()

    if variable is not None:
        df = df.xs(variable, level="variable")

    if horizons is not None:
        if variable is not None:
            df = df.loc[horizons]
        else:
            df = df.loc[horizons]

    return df


def eigenvalues_table(
    solution: LinearSolution,
    formatter: TableFormatter | None = None,
) -> pd.DataFrame:
    """Create table of eigenvalues with stability analysis.

    Args:
        solution: LinearSolution with eigenvalues
        formatter: Optional formatter

    Returns:
        DataFrame with eigenvalue information
    """
    eigenvalues = solution.eigenvalues

    # Filter and sort
    finite_mask = np.isfinite(eigenvalues)
    finite_eigs = eigenvalues[finite_mask]

    sorted_idx = np.argsort(np.abs(finite_eigs))
    sorted_eigs = finite_eigs[sorted_idx]

    data = {
        "Modulus": np.abs(sorted_eigs),
        "Real": np.real(sorted_eigs),
        "Imaginary": np.imag(sorted_eigs),
        "Stable": np.abs(sorted_eigs) < 1.0,
    }

    df = pd.DataFrame(data)
    df.index = range(1, len(df) + 1)
    df.index.name = "#"

    # Add info about infinite eigenvalues
    n_infinite = np.sum(~finite_mask)
    if n_infinite > 0:
        # Add as metadata via attrs
        df.attrs["n_infinite"] = n_infinite

    return df


def model_summary(
    model: ModelIR,
    calibration: Calibration,
    steady_state: SteadyState,
    solution: LinearSolution | None = None,
    formatter: TableFormatter | None = None,
) -> pd.DataFrame:
    """Create model summary table.

    Args:
        model: ModelIR
        calibration: Calibration
        steady_state: SteadyState
        solution: Optional LinearSolution for eigenvalue info
        formatter: Optional formatter

    Returns:
        DataFrame with model summary information
    """
    rows = [
        ("Model name", model.name),
        ("Variables", len(model.variable_names)),
        ("Shocks", len(model.shock_names)),
        ("Parameters", len(model.parameter_names)),
        ("Equations", len(model.equations)),
        ("Steady state valid", steady_state.is_valid()),
    ]

    if solution is not None:
        rows.extend(
            [
                ("Stable eigenvalues", solution.n_stable),
                ("Unstable eigenvalues", solution.n_unstable),
                ("Predetermined vars", solution.n_predetermined),
                ("Determinate", solution.is_determinate()),
            ]
        )

    df = pd.DataFrame(rows, columns=["Property", "Value"])
    return df


def blanchard_kahn_summary(
    solution: LinearSolution,
    formatter: TableFormatter | None = None,
) -> pd.DataFrame:
    """Create Blanchard-Kahn conditions summary.

    Args:
        solution: LinearSolution
        formatter: Optional formatter

    Returns:
        DataFrame with BK conditions info
    """
    n_stable = solution.n_stable
    n_unstable = solution.n_unstable
    n_predetermined = solution.n_predetermined
    n_forward = solution.n_variables - n_predetermined

    is_satisfied = n_stable == n_predetermined

    if n_stable < n_predetermined:
        status = "NO STABLE SOLUTION"
        message = f"Need {n_predetermined} stable eigenvalues, have {n_stable}"
    elif n_stable > n_predetermined:
        status = "INDETERMINACY"
        message = f"Need {n_predetermined} stable eigenvalues, have {n_stable}"
    else:
        status = "DETERMINATE"
        message = "Blanchard-Kahn conditions satisfied"

    rows = [
        ("Status", status),
        ("Message", message),
        ("Predetermined variables", n_predetermined),
        ("Forward-looking variables", n_forward),
        ("Stable eigenvalues", n_stable),
        ("Unstable eigenvalues", n_unstable),
        ("Condition satisfied", is_satisfied),
    ]

    df = pd.DataFrame(rows, columns=["Property", "Value"])
    return df


# ============================================================================
# Formatted string output functions
# ============================================================================


def format_parameters(
    calibration: Calibration,
    format: TableFormat = TableFormat.PLAIN,
    precision: int = 4,
) -> str:
    """Format parameters table as string."""
    formatter = TableFormatter(format=format, precision=precision)
    df = parameters_table(calibration)
    return formatter.format_df(df, "Parameters")


def format_steady_state(
    steady_state: SteadyState,
    format: TableFormat = TableFormat.PLAIN,
    precision: int = 6,
) -> str:
    """Format steady state table as string."""
    formatter = TableFormatter(format=format, precision=precision)
    df = steady_state_table(steady_state)
    return formatter.format_df(df, "Steady State")


def format_moments(
    moments: MomentsResult,
    format: TableFormat = TableFormat.PLAIN,
    precision: int = 4,
) -> str:
    """Format moments table as string."""
    formatter = TableFormatter(format=format, precision=precision)
    df = moments_table(moments)
    return formatter.format_df(df, "Unconditional Moments")


def format_model_summary(
    model: ModelIR,
    calibration: Calibration,
    steady_state: SteadyState,
    solution: LinearSolution | None = None,
    format: TableFormat = TableFormat.PLAIN,
) -> str:
    """Format model summary as string."""
    formatter = TableFormatter(format=format, precision=0)
    df = model_summary(model, calibration, steady_state, solution)
    return formatter.format_df(df, "Model Summary")
