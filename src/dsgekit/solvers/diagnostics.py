"""Blanchard-Kahn diagnostics for linear rational expectations models.

Provides structured analysis of the Blanchard-Kahn conditions:
determinacy, indeterminacy, or no stable solution.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from dsgekit.solvers.linear.gensys import LinearSolution


@dataclass
class EigenInfo:
    """Metadata for a single eigenvalue.

    Attributes:
        value: The (possibly complex) eigenvalue.
        modulus: Absolute value |λ|.
        is_stable: True if |λ| < 1 and root is effective for BK counting.
        is_real: True if imaginary part ≈ 0.
        is_effective: True if root is included in BK counting.
    """

    value: complex
    modulus: float
    is_stable: bool
    is_real: bool
    is_effective: bool

    @property
    def kind(self) -> str:
        if not self.is_effective:
            return "excluded"
        return "stable" if self.is_stable else "unstable"


@dataclass
class BKDiagnostics:
    """Structured Blanchard-Kahn diagnostics.

    Attributes:
        status: ``"determinate"``, ``"indeterminate"``, or
            ``"no_stable_solution"``.
        n_stable: Number of stable eigenvalues (|λ| < 1).
        n_unstable: Number of unstable eigenvalues (|λ| ≥ 1).
        n_predetermined: Number of predetermined (state) variables.
        n_forward: Number of forward-looking variables.
        eigenvalues: Per-eigenvalue metadata, sorted by modulus.
        condition_met: True when n_stable == n_predetermined.
        message: Human-readable diagnostic message.
        solver_meta: Solver metadata used for BK counting.
        trace_id: Deterministic short hash of trace payload.
        trace_lines: Canonical trace payload lines (without trace_id).
    """

    status: str
    n_stable: int
    n_unstable: int
    n_predetermined: int
    n_forward: int
    eigenvalues: list[EigenInfo]
    condition_met: bool
    message: str
    solver_meta: dict[str, Any]
    trace_id: str
    trace_lines: tuple[str, ...]

    def summary(self) -> str:
        """Formatted diagnostic report."""
        label = self.status.upper().replace("_", " ")
        n_stable_raw = self.solver_meta.get("n_stable_raw")
        n_zero_removed = self.solver_meta.get("n_zero_removed")
        lines = [
            "Blanchard-Kahn Diagnostics",
            "=" * 50,
            f"  Status:              {label}",
            f"  Predetermined vars:  {self.n_predetermined}",
            f"  Forward-looking:     {self.n_forward}",
            f"  Stable eigenvalues:  {self.n_stable} (need {self.n_predetermined})",
            f"  Unstable eigenvalues: {self.n_unstable}",
            "",
        ]
        if isinstance(n_stable_raw, int) and isinstance(n_zero_removed, int):
            lines.extend([
                f"  Stable roots (raw):  {n_stable_raw}",
                f"  Structural zeros removed: {n_zero_removed}",
                "",
            ])

        if self.eigenvalues:
            lines.append(
                f"  {'#':<4} {'Modulus':<12} {'Real':<12} {'Imag':<12} {'Type':<10}"
            )
            lines.append(f"  {'-' * 4} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 10}")
            for i, e in enumerate(self.eigenvalues, 1):
                typ = e.kind
                im = np.imag(e.value)
                re = np.real(e.value)
                mod_str = f"{e.modulus:.6f}" if np.isfinite(e.modulus) else "inf"
                re_str = f"{re:.6f}" if np.isfinite(re) else "inf"
                im_str = f"{im:.6f}" if np.isfinite(im) else "inf"
                lines.append(
                    f"  {i:<4} {mod_str:<12} {re_str:<12} {im_str:<12} {typ:<10}"
                )
            lines.append("")

        mark = "OK" if self.condition_met else "FAIL"
        eq = "==" if self.condition_met else "!="
        lines.append(
            f"  Condition: n_stable ({self.n_stable}) "
            f"{eq} n_predetermined ({self.n_predetermined})  [{mark}]"
        )

        if not self.condition_met:
            lines.append("")
            lines.append(f"  {self.message}")

        lines.append("")
        lines.append(f"  Trace ID: {self.trace_id}")

        return "\n".join(lines)

    def trace(self) -> str:
        """Return canonical reproducible trace payload with trace id."""
        return "\n".join([f"BK_TRACE_ID={self.trace_id}", *self.trace_lines])


def _format_trace_float(x: float) -> str:
    if np.isposinf(x):
        return "inf"
    if np.isneginf(x):
        return "-inf"
    return f"{x:.12e}"


def _build_trace_lines(
    status: str,
    n_stable: int,
    n_unstable: int,
    n_predetermined: int,
    n_forward: int,
    eig_infos: list[EigenInfo],
    solver_meta: dict[str, Any],
) -> tuple[str, ...]:
    lines: list[str] = [
        "BK_TRACE_V1",
        f"status={status}",
        f"n_stable={n_stable}",
        f"n_unstable={n_unstable}",
        f"n_predetermined={n_predetermined}",
        f"n_forward={n_forward}",
    ]

    for key in sorted(solver_meta):
        value = solver_meta[key]
        if isinstance(value, float):
            val_str = _format_trace_float(float(value))
        else:
            val_str = str(value)
        lines.append(f"meta.{key}={val_str}")

    for i, e in enumerate(eig_infos, 1):
        re = _format_trace_float(float(np.real(e.value)))
        im = _format_trace_float(float(np.imag(e.value)))
        mod = _format_trace_float(float(e.modulus))
        lines.append(
            f"eig[{i}]=re:{re},im:{im},mod:{mod},"
            f"stable:{int(e.is_stable)},effective:{int(e.is_effective)}"
        )

    return tuple(lines)


def diagnose_bk(solution: LinearSolution) -> BKDiagnostics:
    """Analyse Blanchard-Kahn conditions for a solved model.

    This function never raises — it always returns a diagnostic
    result, even for indeterminate or explosive models.

    Args:
        solution: A ``LinearSolution`` from ``solve_linear()``.
            Can be obtained with ``check_bk=False`` to inspect
            problematic models.

    Returns:
        BKDiagnostics with full eigenvalue analysis and status.
    """
    eigs_raw = solution.eigenvalues
    solver_meta = dict(solution.bk_meta or {})
    zero_tol = float(solver_meta.get("zero_tol", 1e-12))
    n_zero_removed = int(solver_meta.get("n_zero_removed", 0))

    n_pre = solution.n_predetermined
    n_vars = solution.n_variables
    n_fwd = n_vars - n_pre

    finite_idx = np.where(np.isfinite(eigs_raw))[0]
    finite_idx_sorted = sorted(
        finite_idx.tolist(),
        key=lambda i: (
            float(np.abs(eigs_raw[i])),
            float(np.real(eigs_raw[i])),
            float(np.imag(eigs_raw[i])),
        ),
    )
    removable_idx = [
        i for i in finite_idx_sorted
        if float(np.abs(eigs_raw[i])) <= zero_tol
    ]
    excluded_idx = set(removable_idx[:max(0, n_zero_removed)])

    eig_infos = []
    for i, ev in enumerate(eigs_raw):
        mod = float(np.abs(ev))
        is_finite = bool(np.isfinite(ev))
        is_effective = bool(is_finite and (i not in excluded_idx))
        eig_infos.append(
            EigenInfo(
                value=complex(ev),
                modulus=mod,
                is_stable=bool(is_effective and (mod < 1.0)),
                is_real=bool(is_finite and abs(np.imag(ev)) < 1e-10),
                is_effective=is_effective,
            )
        )
    eig_infos.sort(
        key=lambda e: (
            e.modulus if np.isfinite(e.modulus) else np.inf,
            float(np.real(e.value)) if np.isfinite(np.real(e.value)) else np.inf,
            float(np.imag(e.value)) if np.isfinite(np.imag(e.value)) else np.inf,
            0 if not e.is_effective else 1 if e.is_stable else 2,
        )
    )

    computed_stable = int(sum(1 for e in eig_infos if e.is_stable))
    computed_unstable = int(sum(1 for e in eig_infos if e.is_effective and not e.is_stable))
    n_stable = int(solver_meta.get("n_stable_effective", computed_stable))
    n_unstable = int(solver_meta.get("n_unstable_effective", computed_unstable))

    # Determine status
    condition_met = n_stable == n_pre

    if condition_met:
        status = "determinate"
        message = (
            "The Blanchard-Kahn condition is satisfied. "
            "The model has a unique stable solution."
        )
    elif n_stable < n_pre:
        status = "no_stable_solution"
        deficit = n_pre - n_stable
        message = (
            f"Too few stable eigenvalues: found {n_stable}, need {n_pre}. "
            f"The model is explosive ({deficit} stable eigenvalue(s) missing). "
            "Consider checking parameter values — some may place the model "
            "outside the determinacy region."
        )
    else:
        status = "indeterminate"
        excess = n_stable - n_pre
        message = (
            f"Too many stable eigenvalues: found {n_stable}, need {n_pre}. "
            f"The equilibrium is not unique ({excess} excess stable eigenvalue(s)). "
            "This may indicate missing equilibrium selection conditions "
            "or parameter values in the indeterminacy region."
        )

    trace_lines = _build_trace_lines(
        status=status,
        n_stable=n_stable,
        n_unstable=n_unstable,
        n_predetermined=n_pre,
        n_forward=n_fwd,
        eig_infos=eig_infos,
        solver_meta=solver_meta,
    )
    trace_payload = "\n".join(trace_lines)
    trace_id = hashlib.sha256(trace_payload.encode("utf-8")).hexdigest()[:16]

    return BKDiagnostics(
        status=status,
        n_stable=n_stable,
        n_unstable=n_unstable,
        n_predetermined=n_pre,
        n_forward=n_fwd,
        eigenvalues=eig_infos,
        condition_met=condition_met,
        message=message,
        solver_meta=solver_meta,
        trace_id=trace_id,
        trace_lines=trace_lines,
    )
