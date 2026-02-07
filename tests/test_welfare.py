"""Tests for welfare metrics and scenario comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsgekit.simulate import (
    SimulationResult,
    compare_welfare,
    evaluate_welfare,
)


def test_evaluate_welfare_quadratic_closed_form():
    data = pd.DataFrame({"y": [1.0, -1.0, 2.0]})
    result = evaluate_welfare(
        data,
        variables=["y"],
        quadratic_loss_weights={"y": 1.0},
        discount=1.0,
        name="closed_form",
    )

    np.testing.assert_allclose(result.loss_series.values, [1.0, 1.0, 4.0], atol=1e-12)
    np.testing.assert_allclose(result.utility_series.values, [-0.5, -0.5, -2.0], atol=1e-12)
    assert result.mean_loss == pytest.approx(2.0, abs=1e-12)
    assert result.mean_utility == pytest.approx(-1.0, abs=1e-12)
    assert result.discounted_loss == pytest.approx(6.0, abs=1e-12)
    assert result.discounted_utility == pytest.approx(-3.0, abs=1e-12)
    assert "Welfare Summary" in result.summary()


def test_evaluate_welfare_with_targets_linear_term_and_discount():
    data = pd.DataFrame({"y": [2.0, 2.0], "pi": [1.0, -1.0]})
    result = evaluate_welfare(
        data,
        variables=["y", "pi"],
        targets={"y": 1.0, "pi": 0.0},
        linear_utility_weights={"y": 1.0, "pi": -2.0},
        quadratic_loss_weights={"y": 2.0, "pi": 0.5},
        discount=0.9,
        name="targets_linear",
    )

    # Utility path:
    # t0 gaps=(1,1):  1 - 2 - 0.5*(2 + 0.5) = -2.25
    # t1 gaps=(1,-1): 1 + 2 - 0.5*(2 + 0.5) =  1.75
    np.testing.assert_allclose(result.utility_series.values, [-2.25, 1.75], atol=1e-12)
    np.testing.assert_allclose(result.loss_series.values, [2.5, 2.5], atol=1e-12)
    assert result.discounted_utility == pytest.approx(-0.675, abs=1e-12)
    assert result.discounted_loss == pytest.approx(4.75, abs=1e-12)


def test_evaluate_welfare_accepts_simulation_result_object():
    sim = SimulationResult(
        data=pd.DataFrame({"y": [0.2, -0.1, 0.0]}),
        shocks=pd.DataFrame({"e": [0.0, 0.0, 0.0]}),
        n_periods=3,
        seed=7,
    )
    result = evaluate_welfare(
        sim,
        variables=["y"],
        quadratic_loss_weights={"y": 1.0},
        discount=0.95,
    )
    assert result.n_periods == 3
    assert result.variables == ["y"]


def test_compare_welfare_ranks_by_utility_and_loss():
    data = pd.DataFrame({"y": [1.0, 1.0, 1.0]})
    strict = evaluate_welfare(
        data,
        variables=["y"],
        quadratic_loss_weights={"y": 1.0},
        discount=1.0,
        name="strict",
    )
    loose = evaluate_welfare(
        data,
        variables=["y"],
        quadratic_loss_weights={"y": 0.2},
        discount=1.0,
        name="loose",
    )

    by_utility = compare_welfare(
        {"strict": strict, "loose": loose},
        baseline="strict",
        metric="discounted_utility",
    )
    assert by_utility.table.index[0] == "loose"
    assert by_utility.table.loc["loose", "improvement_vs_baseline"] > 0.0

    by_loss = compare_welfare(
        {"strict": strict, "loose": loose},
        baseline="strict",
        metric="discounted_loss",
    )
    assert by_loss.table.index[0] == "loose"
    assert by_loss.table.loc["loose", "improvement_vs_baseline"] > 0.0
    assert "Welfare Comparison" in by_loss.summary()


def test_compare_welfare_baseline_validation():
    sample = evaluate_welfare(
        pd.DataFrame({"y": [0.0, 0.0]}),
        variables=["y"],
        quadratic_loss_weights={"y": 1.0},
    )
    with pytest.raises(ValueError, match="baseline 'missing' not found"):
        compare_welfare({"sample": sample}, baseline="missing")
