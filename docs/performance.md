# Performance and JIT

`dsgekit` includes an optional JIT path for deterministic Newton solvers:

- `linear_solver`: `auto`, `dense`, `sparse`
- `jit_backend`: `none`, `numba` (only applies to dense block Jacobian assembly)

`jit_backend="numba"` requires:

```bash
pip install dsgekit[speed]
```

## Benchmark Script

Use the synthetic benchmark to compare dense Python assembly vs dense numba JIT:

```bash
python examples/benchmark_newton_jit.py --periods 240 --eq 8 --vars 8 --repeats 12
```

The script reports median assembly time and speedup.

Notes:

- It benchmarks Jacobian assembly only (not full model residual/Jacobian evaluation).
- First numba call includes compilation warm-up and is excluded from measured repeats.
- Speedup is shape-dependent (`periods`, `eq`, `vars`, sparsity) and can be <1x or >1x.

## Block Sparse Benchmark (F01)

Use:

```bash
python examples/benchmark_block_sparse_newton.py --periods 240 --vars 8 --zero-prob 0.90 --repeats 4
```

Measured in this environment (2026-02-07):

- `periods=180`, `vars=8`, `zero_prob=0.90`: `dense 58.350 ms`, `sparse 7.110 ms`, `8.21x`.
- `periods=240`, `vars=8`, `zero_prob=0.90`: `dense 121.249 ms`, `sparse 9.400 ms`, `12.90x`.
- `periods=320`, `vars=10`, `zero_prob=0.90`: `dense 473.905 ms`, `sparse 15.798 ms`, `30.00x`.

These runs provide the measurable medium/large-model speedup target for `DYN-F01`.
