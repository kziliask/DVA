from __future__ import annotations

import pandas as pd
import pytest

from dva.analysis.paired_bootstrap import (
    BootstrapConfig,
    bootstrap_metric_table,
    parse_method_column_specs,
    wide_metric_frame,
)


def test_bootstrap_metric_table_reports_paired_reference_deltas() -> None:
    metrics = pd.DataFrame(
        [
            {"unit_id": "a", "dataset": "CAISO", "metric": "AUC \\uparrow", "method": "Post-InfoDVA", "value": 0.8},
            {"unit_id": "b", "dataset": "CAISO", "metric": "AUC \\uparrow", "method": "Post-InfoDVA", "value": 0.6},
            {"unit_id": "c", "dataset": "CAISO", "metric": "AUC \\uparrow", "method": "Post-InfoDVA", "value": 0.7},
            {"unit_id": "a", "dataset": "CAISO", "metric": "AUC \\uparrow", "method": "Prediction SHAP", "value": 0.5},
            {"unit_id": "b", "dataset": "CAISO", "metric": "AUC \\uparrow", "method": "Prediction SHAP", "value": 0.4},
            {"unit_id": "c", "dataset": "CAISO", "metric": "AUC \\uparrow", "method": "Prediction SHAP", "value": 0.6},
        ]
    )

    result = bootstrap_metric_table(
        metrics,
        reference_method="Post-InfoDVA",
        config=BootstrapConfig(n_bootstrap=200, seed=17),
    )

    prediction_row = result.loc[result["method"].eq("Prediction SHAP")].iloc[0]
    assert prediction_row["n_pair_units"] == 3
    assert prediction_row["mean"] == pytest.approx(0.5)
    assert prediction_row["reference_minus_method"] == pytest.approx(0.2)
    assert prediction_row["reference_better_delta"] == pytest.approx(0.2)
    assert prediction_row["reference_better_delta_ci_low"] <= 0.2
    assert prediction_row["reference_better_delta_ci_high"] >= 0.2


def test_bootstrap_metric_table_flips_better_delta_for_down_metrics() -> None:
    metrics = pd.DataFrame(
        [
            {"unit_id": "a", "dataset": "EMS", "metric": "Infidelity \\downarrow", "method": "Post-InfoDVA", "value": 1.0},
            {"unit_id": "b", "dataset": "EMS", "metric": "Infidelity \\downarrow", "method": "Post-InfoDVA", "value": 2.0},
            {"unit_id": "a", "dataset": "EMS", "metric": "Infidelity \\downarrow", "method": "Prediction SHAP", "value": 4.0},
            {"unit_id": "b", "dataset": "EMS", "metric": "Infidelity \\downarrow", "method": "Prediction SHAP", "value": 6.0},
        ]
    )

    result = bootstrap_metric_table(
        metrics,
        reference_method="Post-InfoDVA",
        config=BootstrapConfig(n_bootstrap=100, seed=19),
    )

    prediction_row = result.loc[result["method"].eq("Prediction SHAP")].iloc[0]
    assert prediction_row["reference_minus_method"] == pytest.approx(-3.5)
    assert prediction_row["reference_better_delta"] == pytest.approx(3.5)


def test_wide_metric_frame_loads_method_column_specs(tmp_path) -> None:
    path = tmp_path / "daily.csv"
    pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "decision_decision_insertion_auc": [0.8, 0.6],
            "predictive_decision_insertion_auc": [0.5, 0.4],
        }
    ).to_csv(path, index=False)

    method_columns = parse_method_column_specs(
        [
            "Post-InfoDVA=decision_decision_insertion_auc",
            "Prediction SHAP=predictive_decision_insertion_auc",
        ]
    )
    result = wide_metric_frame(
        path,
        dataset="CAISO",
        metric="Decision insertion AUC \\uparrow",
        method_columns=method_columns,
        unit_column="date",
    )

    assert set(result["method"]) == {"Post-InfoDVA", "Prediction SHAP"}
    assert set(result["unit_id"]) == {"2026-01-01", "2026-01-02"}
    assert result["value"].mean() == pytest.approx(0.575)
