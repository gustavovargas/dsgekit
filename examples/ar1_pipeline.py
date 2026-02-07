"""End-to-end AR(1) example: solve, simulate, moments, filter/smoother, MLE, MAP.

Run from repository root:
    python examples/ar1_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dsgekit import load_model
from dsgekit.estimation import estimate_map, estimate_mle
from dsgekit.filters import kalman_filter, kalman_smoother
from dsgekit.model.calibration import EstimatedParam
from dsgekit.simulate import irf, moments, simulate
from dsgekit.solvers import solve_linear
from dsgekit.transforms import linearize, to_state_space


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "tests" / "fixtures" / "models" / "ar1.yaml"

    model, cal, ss = load_model(model_path)
    lin = linearize(model, ss, cal)
    solution = solve_linear(lin)

    print("== Linear solution ==")
    print(f"T = {solution.T}")
    print(f"R = {solution.R}")
    print()

    irfs = irf(solution, "e", periods=10)
    print("== IRF(y) to shock e ==")
    print(irfs.data["y"].to_string())
    print()

    sim = simulate(solution, cal, n_periods=200, seed=42)
    m = moments(solution, cal, max_lag=5)
    print("== Simulated data (head) ==")
    print(sim.data.head().to_string())
    print()
    print("== Moments summary ==")
    print(m.summary())
    print()

    ss_model = to_state_space(solution, cal, observables=["y"])
    kf = kalman_filter(ss_model, sim.data[["y"]])
    sm = kalman_smoother(ss_model, kf)
    print("== Filter/smoother ==")
    print(kf.summary())
    print(sm.summary())
    print()

    mle = estimate_mle(
        model,
        ss,
        cal,
        sim.data[["y"]],
        observables=["y"],
        param_names=["rho"],
        bounds={"rho": (0.01, 0.99)},
        compute_se=False,
    )
    print("== MLE ==")
    print(mle.summary())

    cal_map = cal.copy()
    cal_map.estimated_params = [
        EstimatedParam.from_dict(
            {
                "type": "param",
                "name": "rho",
                "init": 0.5,
                "lower": 0.01,
                "upper": 0.99,
                "prior": {"distribution": "normal", "mean": 0.8, "std": 0.1},
            }
        )
    ]
    map_res = estimate_map(
        model,
        ss,
        cal_map,
        sim.data[["y"]],
        observables=["y"],
        param_names=None,
        bounds=None,
        compute_se=False,
    )
    print("== MAP ==")
    print(map_res.summary())


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    main()
