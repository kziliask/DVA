from __future__ import annotations

import pandas as pd
import pytest

from dva.analysis.ems_exact_shap import EmsExactShapOutputs
from dva.analysis.run_ems_cvar_alpha_experiment import (
    DEFAULT_CVAR_ALPHA_GRID,
    _build_alpha_summary,
    _build_per_hour_comparison,
    _resolve_alpha_grid,
)


def test_default_alpha_grid_has_representative_risk_levels() -> None:
    assert DEFAULT_CVAR_ALPHA_GRID == (0.0, 0.5, 0.8, 0.9, 0.95)


def test_resolve_alpha_grid_validates_and_deduplicates_values() -> None:
    assert _resolve_alpha_grid([0.0, 0.5, 0.5, 0.95]) == (0.0, 0.5, 0.95)
    with pytest.raises(ValueError, match="alpha"):
        _resolve_alpha_grid([1.0])


def test_cvar_alpha_experiment_comparison_tables() -> None:
    normal_outputs = _fake_outputs(
        pd.DataFrame(
            {
                "timestamp_hour": ["2025-02-01 00:00:00", "2025-02-01 01:00:00"],
                "oracle_value": [10.0, 12.0],
                "decision_full_value": [8.0, 10.0],
                "decision_value_gain": [3.0, 4.0],
                "actual_regret": [2.0, 2.0],
                "full_selected_zip_codes": ['["10001"]', '["10002"]'],
                "decision_shap_weather": [2.0, 1.0],
                "decision_shap_lagged_demand": [1.0, 3.0],
            }
        ),
        runtime_seconds=5.0,
        compute_cvar=False,
    )
    cvar_outputs = _fake_outputs(
        pd.DataFrame(
            {
                "timestamp_hour": ["2025-02-01 00:00:00", "2025-02-01 01:00:00"],
                "oracle_value": [10.0, 12.0],
                "decision_full_value": [8.0, 10.0],
                "decision_value_gain": [3.0, 4.0],
                "actual_regret": [2.0, 2.0],
                "full_selected_zip_codes": ['["10001"]', '["10002"]'],
                "cvar_decision_full_value": [9.0, 9.0],
                "cvar_decision_value_gain": [4.0, 3.0],
                "cvar_actual_regret": [1.0, 3.0],
                "cvar_full_selected_zip_codes": ['["10003"]', '["10002"]'],
                "cvar_full_risk_objective_value": [6.0, 7.0],
                "cvar_baseline_risk_objective_value": [2.0, 3.0],
                "cvar_decision_shap_weather": [1.5, 1.0],
                "cvar_decision_shap_lagged_demand": [2.5, 2.0],
            }
        ),
        runtime_seconds=8.0,
        compute_cvar=True,
        cvar_alpha=0.9,
    )

    per_hour = _build_per_hour_comparison(normal_outputs, {0.9: cvar_outputs})
    summary = _build_alpha_summary(per_hour, normal_outputs, {0.9: cvar_outputs})

    assert len(per_hour) == 2
    assert per_hour["cvar_minus_normal_actual_regret"].tolist() == [-1.0, 1.0]
    assert per_hour["full_selection_changed"].tolist() == [True, False]
    assert per_hour["decision_vs_cvar_shap_l1"].tolist() == [2.0, 1.0]

    summary_row = summary.iloc[0]
    assert summary_row["alpha"] == pytest.approx(0.9)
    assert summary_row["mean_cvar_minus_normal_actual_regret"] == pytest.approx(0.0)
    assert summary_row["cvar_better_hour_share"] == pytest.approx(0.5)
    assert summary_row["full_selection_change_share"] == pytest.approx(0.5)
    assert summary_row["mean_cvar_full_risk_objective_value"] == pytest.approx(6.5)


def _fake_outputs(
    hourly_shap: pd.DataFrame,
    *,
    runtime_seconds: float,
    compute_cvar: bool,
    cvar_alpha: float | None = None,
) -> EmsExactShapOutputs:
    return EmsExactShapOutputs(
        hourly_shap=hourly_shap,
        predictive_zip_shap=pd.DataFrame(),
        coverage_solutions=pd.DataFrame(),
        summary_shap=pd.DataFrame(),
        prediction_metrics={
            "holdout": {
                "mae": 1.0,
                "mse": 2.0,
                "rmse": 2.0**0.5,
            }
        },
        evaluation_metrics={},
        run_metadata={
            "player_names": ["weather", "lagged_demand"],
            "runtime_seconds": runtime_seconds,
            "compute_cvar_decision_shap": compute_cvar,
            "cvar_alpha": cvar_alpha,
            "cvar_scenario_count": 25,
        },
    )
