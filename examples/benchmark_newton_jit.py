"""Benchmark dense block Jacobian assembly with optional numba JIT backend."""

from __future__ import annotations

import argparse
import time

import numpy as np

from dsgekit.exceptions import SolverError
from dsgekit.solvers.nonlinear._newton_linear import BlockTridiagonalJacobianBuilder


def _random_block(
    rng: np.random.Generator,
    n_eq: int,
    n_vars: int,
    zero_prob: float,
) -> np.ndarray:
    block = rng.normal(size=(n_eq, n_vars)).astype(np.float64)
    if zero_prob > 0.0:
        keep = rng.uniform(size=(n_eq, n_vars)) > zero_prob
        block *= keep
    return block


def _build_dense_jacobian(
    *,
    n_periods: int,
    n_eq: int,
    n_vars: int,
    jit_backend: str,
    seed: int,
    zero_prob: float,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    builder = BlockTridiagonalJacobianBuilder(
        n_periods=n_periods,
        n_eq=n_eq,
        n_vars=n_vars,
        use_sparse=False,
        jit_backend=jit_backend,  # type: ignore[arg-type]
    )
    for t in range(n_periods):
        a_t = _random_block(rng, n_eq, n_vars, zero_prob=zero_prob)
        b_t = _random_block(rng, n_eq, n_vars, zero_prob=zero_prob)
        c_t = _random_block(rng, n_eq, n_vars, zero_prob=zero_prob)
        builder.add_period_blocks(period=t, a_t=a_t, b_t=b_t, c_t=c_t)
    assembled = builder.build()
    return np.asarray(assembled.matrix, dtype=np.float64), int(assembled.nnz)


def _benchmark_mode(
    *,
    mode: str,
    n_periods: int,
    n_eq: int,
    n_vars: int,
    zero_prob: float,
    repeats: int,
    seed: int,
) -> tuple[float, int]:
    elapsed_ms: list[float] = []
    nnz_last = 0
    # Warm-up call. For numba this triggers compilation once.
    _build_dense_jacobian(
        n_periods=n_periods,
        n_eq=n_eq,
        n_vars=n_vars,
        jit_backend=mode,
        seed=seed,
        zero_prob=zero_prob,
    )
    for i in range(repeats):
        t0 = time.perf_counter()
        _, nnz_last = _build_dense_jacobian(
            n_periods=n_periods,
            n_eq=n_eq,
            n_vars=n_vars,
            jit_backend=mode,
            seed=seed + i + 1,
            zero_prob=zero_prob,
        )
        t1 = time.perf_counter()
        elapsed_ms.append((t1 - t0) * 1000.0)
    return float(np.median(elapsed_ms)), nnz_last


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark dense block Jacobian assembly (python vs numba JIT).",
    )
    parser.add_argument("--periods", type=int, default=240, help="Number of periods.")
    parser.add_argument("--eq", type=int, default=8, help="Equations per period.")
    parser.add_argument("--vars", type=int, default=8, help="Variables per period.")
    parser.add_argument(
        "--zero-prob",
        type=float,
        default=0.6,
        help="Probability each coefficient is set to zero.",
    )
    parser.add_argument("--repeats", type=int, default=12, help="Benchmark repeats.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = parser.parse_args()

    dense_ms, nnz = _benchmark_mode(
        mode="none",
        n_periods=args.periods,
        n_eq=args.eq,
        n_vars=args.vars,
        zero_prob=args.zero_prob,
        repeats=args.repeats,
        seed=args.seed,
    )
    print(
        f"dense/python median_ms={dense_ms:.3f} "
        f"(periods={args.periods}, eq={args.eq}, vars={args.vars}, nnz={nnz})"
    )

    try:
        numba_ms, _ = _benchmark_mode(
            mode="numba",
            n_periods=args.periods,
            n_eq=args.eq,
            n_vars=args.vars,
            zero_prob=args.zero_prob,
            repeats=args.repeats,
            seed=args.seed,
        )
    except SolverError as exc:
        print(f"dense/numba unavailable: {exc}")
        return 0

    speedup = dense_ms / numba_ms if numba_ms > 0.0 else float("inf")
    print(f"dense/numba median_ms={numba_ms:.3f}")
    print(f"speedup={speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
