"""Reference model-zoo tests (RBC, NK, SOE, fiscal, ZLB toy)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.simulate import irf, simulate
from dsgekit.solvers import diagnose_bk, solve_linear
from dsgekit.transforms import linearize

REFERENCE_MODELS = [
    pytest.param("rbc.mod", 4, 1, id="rbc"),
    pytest.param("nk.yaml", 3, 3, id="nk"),
    pytest.param("soe.yaml", 3, 2, id="soe"),
    pytest.param("fiscal.yaml", 3, 2, id="fiscal"),
    pytest.param("zlb_toy.yaml", 4, 3, id="zlb-toy"),
]


@pytest.mark.parametrize(("fixture_name", "n_vars", "n_shocks"), REFERENCE_MODELS)
def test_reference_model_zoo_load_and_solve(models_dir, fixture_name, n_vars, n_shocks):
    """All reference models should load and produce a determinate linear solution."""
    model, cal, ss = load_model(models_dir / fixture_name)

    assert model.n_variables == n_vars
    assert model.n_shocks == n_shocks
    assert model.n_equations == n_vars

    lin = linearize(model, ss, cal)
    solution = solve_linear(lin)
    diag = diagnose_bk(solution)

    assert solution.T.shape == (n_vars, n_vars)
    assert solution.R.shape == (n_vars, n_shocks)
    assert diag.status == "determinate"
    assert solution.n_stable == model.n_predetermined
    assert np.all(np.isfinite(solution.T))
    assert np.all(np.isfinite(solution.R))


@pytest.mark.parametrize(("fixture_name", "n_vars", "_"), REFERENCE_MODELS)
def test_reference_model_zoo_dynamics(models_dir, fixture_name, n_vars, _):
    """IRFs and simulations should be finite for all reference models."""
    model, cal, ss = load_model(models_dir / fixture_name)
    solution = solve_linear(linearize(model, ss, cal))

    irf_result = irf(solution, model.shock_names[0], periods=8)
    assert irf_result.data.shape == (8, n_vars)
    assert np.all(np.isfinite(irf_result.data.values))

    sim = simulate(solution, cal, n_periods=40, seed=404)
    assert sim.data.shape == (40, n_vars)
    assert np.all(np.isfinite(sim.data.values))


def test_reference_model_zoo_example_executes():
    """Smoke test: the model-zoo example script runs without crashing."""
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "reference_model_zoo.py"

    env = os.environ.copy()
    src_path = str(repo_root / "src")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{src_path}:{current_pythonpath}"
        if current_pythonpath
        else src_path
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "Reference model zoo" in completed.stdout
