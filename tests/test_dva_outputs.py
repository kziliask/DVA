from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dva.analysis.caiso_shap import (
    DailyInteractionExplanation,
    _build_daily_interaction_decision_frame,
    _build_daily_interaction_ead_decision_frame,
)
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
            "decision_mean_signed_shap": [1.0],
            "decision_mean_abs_shap": [1.0],
            "decision_rank": [1],
        }
    ).to_csv(tmp_path / "summary_shap.csv", index=False)
    pd.DataFrame(
        {
            "players": ["min_temp_c", "min_temp_c|efficiency"],
            "subset_size": [1, 2],
            "decision_interaction_value": [3.0, 4.0],
        }
    ).to_csv(tmp_path / "daily_interaction_decision.csv", index=False)

    write_canonical_caiso_dva_outputs(tmp_path, value_mode="post")

    dvi = pd.read_csv(tmp_path / "dvi_interactions.csv")
    assert dvi["players"].tolist() == ["min_temp_c|efficiency"]
    assert dvi["interaction_type"].tolist() == ["Cross-DVI"]
    assert dvi["decision_interaction_value"].tolist() == [4.0]


def test_caiso_canonical_ante_dvi_uses_ead_interactions(tmp_path) -> None:
    pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "ead_decision_shap_min_temp_c": [2.0],
            "ead_decision_value_gain": [2.0],
            "decision_shap_min_temp_c": [-1.0],
            "decision_value_gain": [-1.0],
        }
    ).to_csv(tmp_path / "daily_shap.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["min_temp_c"],
            "decision_mean_signed_shap": [-1.0],
            "decision_mean_abs_shap": [1.0],
            "decision_rank": [2],
            "ead_decision_mean_signed_shap": [2.0],
            "ead_decision_mean_abs_shap": [2.0],
            "ead_decision_rank": [1],
        }
    ).to_csv(tmp_path / "summary_shap.csv", index=False)
    pd.DataFrame(
        {
            "players": ["min_temp_c|efficiency"],
            "subset_size": [2],
            "decision_interaction_value": [-10.0],
        }
    ).to_csv(tmp_path / "daily_interaction_decision.csv", index=False)
    pd.DataFrame(
        {
            "players": ["min_temp_c|efficiency"],
            "subset_size": [2],
            "ead_decision_interaction_value": [7.0],
        }
    ).to_csv(tmp_path / "daily_interaction_ead_decision.csv", index=False)

    write_canonical_caiso_dva_outputs(tmp_path, value_mode="ante")

    dvi = pd.read_csv(tmp_path / "dvi_interactions.csv")
    assert dvi["players"].tolist() == ["min_temp_c|efficiency"]
    assert dvi["ead_decision_interaction_value"].tolist() == [7.0]
    assert dvi["decision_interaction_value"].tolist() == [7.0]


def test_caiso_canonical_ante_dvi_requires_ead_interactions(tmp_path) -> None:
    pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "ead_decision_shap_min_temp_c": [2.0],
            "ead_decision_value_gain": [2.0],
        }
    ).to_csv(tmp_path / "daily_shap.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["min_temp_c"],
            "ead_decision_mean_signed_shap": [2.0],
            "ead_decision_mean_abs_shap": [2.0],
            "ead_decision_rank": [1],
        }
    ).to_csv(tmp_path / "summary_shap.csv", index=False)

    with pytest.raises(FileNotFoundError, match="Ante CAISO DVI requires"):
        write_canonical_caiso_dva_outputs(tmp_path, value_mode="ante")


def test_caiso_canonical_writes_joint_dva_outputs(tmp_path) -> None:
    pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "ead_decision_shap_min_temp_c": [1.0, 3.0],
            "ead_decision_shap_efficiency": [2.0, 4.0],
            "ead_decision_value_gain": [3.0, 7.0],
        }
    ).to_csv(tmp_path / "daily_shap.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["min_temp_c", "efficiency"],
            "ead_decision_mean_signed_shap": [2.0, 3.0],
            "ead_decision_mean_abs_shap": [2.0, 3.0],
            "ead_decision_rank": [2, 1],
        }
    ).to_csv(tmp_path / "summary_shap.csv", index=False)
    pd.DataFrame(
        {
            "players": ["min_temp_c|efficiency"],
            "subset_size": [2],
            "ead_decision_interaction_value": [5.0],
        }
    ).to_csv(tmp_path / "daily_interaction_ead_decision.csv", index=False)
    (tmp_path / "run_metadata.json").write_text(
        """
{
  "evaluation_label": "joint_dvi_xgb_001_conservative_ante",
  "parameter_player_spec": {"charge_efficiency_baseline": 0.8},
  "player_names": ["min_temp_c", "efficiency"],
  "storage_parameters": {"charge_efficiency": 0.95}
}
""".strip(),
        encoding="utf-8",
    )

    write_canonical_caiso_dva_outputs(tmp_path, value_mode="ante")

    joint = pd.read_csv(tmp_path / "joint_dva.csv")
    joint_summary = pd.read_csv(tmp_path / "joint_summary_dva.csv")
    assert joint["player"].tolist() == [
        "min_temp_c",
        "efficiency",
        "min_temp_c",
        "efficiency",
    ]
    assert joint["player_kind"].tolist() == ["info", "design", "info", "design"]
    assert joint.loc[joint["player"].eq("efficiency"), "baseline"].tolist() == [0.8, 0.8]
    assert joint.loc[joint["player"].eq("efficiency"), "actual"].tolist() == [0.95, 0.95]
    assert joint["model_id"].dropna().unique().tolist() == ["xgb_001"]
    assert joint_summary["player"].tolist() == ["efficiency", "min_temp_c"]
    assert joint_summary["dva_rank"].tolist() == [1, 2]


def test_caiso_ead_interaction_frame_is_separate_from_post_decision() -> None:
    explanation = DailyInteractionExplanation(
        date="2026-01-01",
        method="faith_shap",
        order=2,
        player_names=("min_temp_c", "efficiency"),
        decision_indices={frozenset({0, 1}): 1.0},
        ead_decision_indices={frozenset({0, 1}): 2.0},
        predictive_indices={frozenset({0, 1}): np.array([3.0])},
        decision_value_full=1.0,
        ead_decision_value_full=2.0,
        predictive_value_full=np.array([4.0]),
        predictive_value_empty=np.array([1.0]),
    )

    decision = _build_daily_interaction_decision_frame([explanation])
    ead_decision = _build_daily_interaction_ead_decision_frame([explanation])

    assert decision is not None
    assert ead_decision is not None
    assert decision.loc[0, "decision_interaction_value"] == 1.0
    assert ead_decision.loc[0, "ead_decision_interaction_value"] == 2.0


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
