from __future__ import annotations

import pandas as pd

from dva.case_studies.caiso.outputs import write_canonical_caiso_dva_outputs
from dva.case_studies.ems.outputs import write_canonical_ems_dva_outputs


def test_caiso_canonical_dvi_keeps_two_way_interactions(tmp_path) -> None:
    pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "decision_shap_min_temp_c": [1.0],
            "decision_value_gain": [1.0],
        }
    ).to_csv(tmp_path / "daily_shap.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["min_temp_c"],
            "decision_mean_abs_shap": [1.0],
            "decision_rank": [1],
        }
    ).to_csv(tmp_path / "summary_shap.csv", index=False)
    pd.DataFrame(
        {
            "players": ["min_temp_c", "min_temp_c|throughput_penalty"],
            "subset_size": [1, 2],
            "decision_interaction_value": [3.0, 4.0],
        }
    ).to_csv(tmp_path / "daily_interaction_decision.csv", index=False)

    write_canonical_caiso_dva_outputs(tmp_path, value_mode="post")

    dvi = pd.read_csv(tmp_path / "dvi_interactions.csv")
    assert dvi["players"].tolist() == ["min_temp_c|throughput_penalty"]
    assert dvi["interaction_type"].tolist() == ["Cross-DVI"]


def test_ems_canonical_ante_dva_uses_ante_columns(tmp_path) -> None:
    pd.DataFrame(
        {
            "timestamp_hour": ["2025-01-01 00:00:00"],
            "ante_decision_shap_hour": [2.5],
            "ante_decision_value_gain": [2.5],
            "decision_shap_hour": [-1.0],
            "decision_value_gain": [-1.0],
        }
    ).to_csv(tmp_path / "hourly_shap.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["hour"],
            "decision_mean_abs_shap": [1.0],
            "decision_rank": [2],
            "ante_decision_mean_abs_shap": [2.5],
            "ante_decision_rank": [1],
        }
    ).to_csv(tmp_path / "summary_shap.csv", index=False)

    write_canonical_ems_dva_outputs(tmp_path, value_mode="ante")

    daily = pd.read_csv(tmp_path / "daily_dva.csv")
    summary = pd.read_csv(tmp_path / "summary_dva.csv")
    assert daily.loc[0, "dva_hour"] == 2.5
    assert daily.loc[0, "dva_value_gain"] == 2.5
    assert "dva_value_mode_warning" not in daily.columns
    assert summary.loc[0, "dva_mean_abs"] == 2.5
    assert summary.loc[0, "dva_rank"] == 1
