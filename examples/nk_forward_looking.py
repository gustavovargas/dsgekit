"""Forward-looking NK example with BK diagnostics.

Run from repository root:
    python examples/nk_forward_looking.py
"""

from __future__ import annotations

from pathlib import Path

from dsgekit import load_model
from dsgekit.exceptions import IndeterminacyError
from dsgekit.simulate import irf
from dsgekit.solvers import diagnose_bk, solve_linear
from dsgekit.transforms import linearize


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "tests" / "fixtures" / "models" / "nk.yaml"

    model, cal, ss = load_model(model_path)
    lin = linearize(model, ss, cal)

    print("== NK 3-equation model ==")
    print(f"Variables: {model.variable_names}")
    print(f"Shocks: {model.shock_names}")
    print(
        f"Predetermined: {model.predetermined_variable_names}, "
        f"Forward-looking: {model.forward_looking_variable_names}"
    )
    print()

    try:
        solve_linear(lin)
    except IndeterminacyError as exc:
        print("BK check raised as expected:")
        print(str(exc))
        print()

    solution = solve_linear(lin, check_bk=False)
    diag = diagnose_bk(solution)

    print("== BK diagnostics (check_bk=False) ==")
    print(diag.summary())
    print()

    print("== IRF head for each shock ==")
    for shock_name in model.shock_names:
        responses = irf(solution, shock_name, periods=8).data
        print(f"Shock: {shock_name}")
        print(responses.head().to_string())
        print()


if __name__ == "__main__":
    main()
