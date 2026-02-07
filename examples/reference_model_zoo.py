"""Reference model zoo smoke example.

Run from repository root:
    python examples/reference_model_zoo.py
"""

from __future__ import annotations

from pathlib import Path

from dsgekit import load_model
from dsgekit.simulate import irf
from dsgekit.solvers import diagnose_bk, solve_linear
from dsgekit.transforms import linearize


def _run_model(repo_root: Path, label: str, relative_path: str, shock_name: str) -> None:
    model_path = repo_root / relative_path
    model, cal, ss = load_model(model_path)
    solution = solve_linear(linearize(model, ss, cal))
    diag = diagnose_bk(solution)

    print(f"== {label} ==")
    print(f"Model file: {relative_path}")
    print(f"BK status: {diag.status}")
    print(solution.summary())
    print(f"IRF head to shock '{shock_name}':")
    print(irf(solution, shock_name, periods=6).data.head().to_string())
    print()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    print("Reference model zoo")
    print("===================")
    print()

    models = [
        ("RBC", "tests/fixtures/models/rbc.mod", "e"),
        ("NK", "tests/fixtures/models/nk.yaml", "e_d"),
        ("SOE", "tests/fixtures/models/soe.yaml", "e_a"),
        ("Fiscal", "tests/fixtures/models/fiscal.yaml", "e_g"),
        ("ZLB toy (shadow-rate approximation)", "tests/fixtures/models/zlb_toy.yaml", "e_i"),
    ]

    for label, relative_path, shock_name in models:
        _run_model(repo_root, label, relative_path, shock_name)


if __name__ == "__main__":
    main()
