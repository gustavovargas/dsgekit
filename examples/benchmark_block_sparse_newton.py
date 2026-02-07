"""Benchmark block-dense vs block-sparse Newton linear algebra."""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy import sparse

from dsgekit.solvers.nonlinear._newton_linear import (
    BlockTridiagonalJacobianBuilder,
    solve_newton_step,
)


def _build_blocks(
    *,
    n_periods: int,
    n_vars: int,
    zero_prob: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_blocks = rng.normal(scale=0.05, size=(n_periods, n_vars, n_vars))
    c_blocks = rng.normal(scale=0.05, size=(n_periods, n_vars, n_vars))
    if zero_prob > 0.0:
        a_blocks *= rng.uniform(size=a_blocks.shape) > zero_prob
        c_blocks *= rng.uniform(size=c_blocks.shape) > zero_prob

    b_blocks = np.zeros((n_periods, n_vars, n_vars), dtype=np.float64)
    eye = np.eye(n_vars, dtype=np.float64)
    for t in range(n_periods):
        b_t = eye + rng.normal(scale=0.02, size=(n_vars, n_vars))
        if zero_prob > 0.0:
            b_t *= rng.uniform(size=(n_vars, n_vars)) > zero_prob
            b_t += eye
        b_blocks[t] = b_t

    return (
        a_blocks.astype(np.float64),
        b_blocks.astype(np.float64),
        c_blocks.astype(np.float64),
    )


def _assemble_and_solve(
    *,
    mode: str,
    a_blocks: np.ndarray,
    b_blocks: np.ndarray,
    c_blocks: np.ndarray,
    rhs: np.ndarray,
    repeats: int,
) -> tuple[float, float, float, int, float]:
    total_ms: list[float] = []
    assembly_ms: list[float] = []
    solve_ms: list[float] = []
    nnz_last = 0
    residual_inf_last = 0.0

    for _ in range(repeats):
        t0 = time.perf_counter()
        builder = BlockTridiagonalJacobianBuilder(
            n_periods=a_blocks.shape[0],
            n_eq=a_blocks.shape[1],
            n_vars=a_blocks.shape[2],
            use_sparse=(mode == "sparse"),
        )
        for t in range(a_blocks.shape[0]):
            builder.add_period_blocks(
                period=t,
                a_t=a_blocks[t],
                b_t=b_blocks[t],
                c_t=c_blocks[t],
            )
        assembled = builder.build()
        t1 = time.perf_counter()

        delta = solve_newton_step(
            jacobian=assembled.matrix,
            rhs=rhs,
            solver_mode=mode,  # type: ignore[arg-type]
        )
        t2 = time.perf_counter()

        jac = assembled.matrix
        if sparse.issparse(jac):
            residual = jac @ delta - rhs
        else:
            residual = np.asarray(jac, dtype=np.float64) @ delta - rhs
        residual_inf_last = float(np.max(np.abs(residual)))

        total_ms.append((t2 - t0) * 1000.0)
        assembly_ms.append((t1 - t0) * 1000.0)
        solve_ms.append((t2 - t1) * 1000.0)
        nnz_last = int(assembled.nnz)

    return (
        float(np.median(total_ms)),
        float(np.median(assembly_ms)),
        float(np.median(solve_ms)),
        nnz_last,
        residual_inf_last,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark dense vs sparse block Newton linear algebra.",
    )
    parser.add_argument("--periods", type=int, default=240, help="Number of periods.")
    parser.add_argument("--vars", type=int, default=8, help="Variables per period.")
    parser.add_argument(
        "--zero-prob",
        type=float,
        default=0.90,
        help="Probability that each coefficient is zeroed.",
    )
    parser.add_argument("--repeats", type=int, default=4, help="Benchmark repeats.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    a_blocks, b_blocks, c_blocks = _build_blocks(
        n_periods=args.periods,
        n_vars=args.vars,
        zero_prob=args.zero_prob,
        rng=rng,
    )
    rhs = rng.normal(size=args.periods * args.vars).astype(np.float64)

    dense = _assemble_and_solve(
        mode="dense",
        a_blocks=a_blocks,
        b_blocks=b_blocks,
        c_blocks=c_blocks,
        rhs=rhs,
        repeats=args.repeats,
    )
    sparse_result = _assemble_and_solve(
        mode="sparse",
        a_blocks=a_blocks,
        b_blocks=b_blocks,
        c_blocks=c_blocks,
        rhs=rhs,
        repeats=args.repeats,
    )

    dense_total, dense_asm, dense_solve, nnz, dense_resid = dense
    sparse_total, sparse_asm, sparse_solve, _, sparse_resid = sparse_result
    speedup = dense_total / sparse_total if sparse_total > 0.0 else float("inf")

    print(
        "config "
        f"periods={args.periods} vars={args.vars} zero_prob={args.zero_prob} "
        f"repeats={args.repeats} nnz={nnz}"
    )
    print(
        f"dense total_ms={dense_total:.3f} "
        f"assembly_ms={dense_asm:.3f} solve_ms={dense_solve:.3f} "
        f"residual_inf={dense_resid:.3e}"
    )
    print(
        f"sparse total_ms={sparse_total:.3f} "
        f"assembly_ms={sparse_asm:.3f} solve_ms={sparse_solve:.3f} "
        f"residual_inf={sparse_resid:.3e}"
    )
    print(f"speedup_total={speedup:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
