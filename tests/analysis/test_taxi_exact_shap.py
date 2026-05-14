from __future__ import annotations

import pandas as pd

from dva.analysis.taxi_exact_shap import (
    DEFAULT_ORTOOLS_LOCAL_SEARCH_METAHEURISTIC,
    DEFAULT_ORTOOLS_TIME_LIMIT_S,
    TaxiExactShapConfig,
    _build_orienteering_solver_params,
    _ordered_summary_shap_frame,
    _predictive_shap_feature_order,
)


def test_predictive_shap_feature_order_uses_predictive_rank() -> None:
    summary_shap = pd.DataFrame(
        {
            "feature": ["low", "high", "mid"],
            "predictive_mean_abs_shap": [1.0, 3.0, 2.0],
            "predictive_rank": [3, 1, 2],
            "decision_mean_abs_shap": [10.0, 0.5, 2.0],
        }
    )

    assert _predictive_shap_feature_order(summary_shap) == ("high", "mid", "low")


def test_ordered_summary_shap_frame_preserves_predictive_order() -> None:
    summary_shap = pd.DataFrame(
        {
            "feature": ["low", "high", "mid"],
            "predictive_mean_abs_shap": [1.0, 3.0, 2.0],
            "decision_mean_abs_shap": [10.0, 0.5, 2.0],
        }
    )

    ordered = _ordered_summary_shap_frame(summary_shap, ("high", "mid", "low"))

    assert tuple(ordered["feature"]) == ("high", "mid", "low")
    assert tuple(ordered["decision_mean_abs_shap"]) == (0.5, 2.0, 10.0)


def test_ortools_solver_params_use_fast_greedy_descent_defaults() -> None:
    params = _build_orienteering_solver_params(
        TaxiExactShapConfig(orienteering_method="ortools")
    )

    assert params == {
        "time_limit_s": DEFAULT_ORTOOLS_TIME_LIMIT_S,
        "local_search_metaheuristic": DEFAULT_ORTOOLS_LOCAL_SEARCH_METAHEURISTIC,
    }
