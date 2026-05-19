from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dva.plots.make_caiso_joint_dvi_interaction_heatmaps import (
    BATTERY_PARAMETERS,
    FEATURES,
    INDIVIDUAL_VALUE_KEY,
    SCENARIOS,
    aggregate_individual_values,
    aggregate_interactions,
    color_limit,
    load_individual_value_frame,
    load_interaction_frame,
    scenario_matrix,
)


def _write_scenario_outputs(root: Path, *, baseline: str, value_mode: str) -> None:
    run_dir = root / "xgb_001" / f"{baseline}_{value_mode}"
    run_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "interaction_method": "faith_shap",
                "order": 2,
                "subset_size": 2,
                "players": "min_temp_c|efficiency",
                "decision_interaction_value": 1.0,
                "interaction_type": "Cross-DVI",
            },
            {
                "date": "2026-01-01",
                "interaction_method": "faith_shap",
                "order": 2,
                "subset_size": 2,
                "players": "energy_capacity|day_of_week",
                "decision_interaction_value": 2.0,
                "interaction_type": "Cross-DVI",
            },
            {
                "date": "2026-01-01",
                "interaction_method": "faith_shap",
                "order": 2,
                "subset_size": 2,
                "players": "min_temp_c|max_temp_c",
                "decision_interaction_value": 999.0,
                "interaction_type": "Info-Info",
            },
        ]
    ).to_csv(run_dir / "dvi_interactions.csv", index=False)

    daily_values: dict[str, object] = {"date": "2026-01-01"}
    for player, value in {
        **{feature: 0.0 for feature, _ in FEATURES},
        **{parameter: 0.0 for parameter, _ in BATTERY_PARAMETERS},
        "min_temp_c": 10.0,
        "day_of_week": 20.0,
        "efficiency": 30.0,
        "energy_capacity": 40.0,
    }.items():
        daily_values[f"dva_{player}"] = value
    pd.DataFrame([daily_values]).to_csv(run_dir / "daily_dva.csv", index=False)


def test_caiso_joint_dvi_heatmap_appends_individual_value_margins(
    tmp_path: Path,
) -> None:
    for _, baseline, value_mode, _ in SCENARIOS:
        _write_scenario_outputs(tmp_path, baseline=baseline, value_mode=value_mode)

    interactions = aggregate_interactions(load_interaction_frame(tmp_path), "mean")
    individuals = aggregate_individual_values(load_individual_value_frame(tmp_path), "mean")

    matrix = scenario_matrix(
        interactions,
        individuals,
        baseline="conservative",
        value_mode="ante",
    )

    assert matrix.shape == (len(FEATURES) + 1, len(BATTERY_PARAMETERS) + 1)
    assert matrix.loc["min_temp_c", "efficiency"] == pytest.approx(1.0)
    assert matrix.loc["day_of_week", "energy_capacity"] == pytest.approx(2.0)
    assert matrix.loc["min_temp_c", INDIVIDUAL_VALUE_KEY] == pytest.approx(10.0)
    assert matrix.loc["day_of_week", INDIVIDUAL_VALUE_KEY] == pytest.approx(20.0)
    assert matrix.loc[INDIVIDUAL_VALUE_KEY, "efficiency"] == pytest.approx(30.0)
    assert matrix.loc[INDIVIDUAL_VALUE_KEY, "energy_capacity"] == pytest.approx(40.0)
    assert np.isnan(matrix.loc[INDIVIDUAL_VALUE_KEY, INDIVIDUAL_VALUE_KEY])


def test_caiso_joint_dvi_color_limit_excludes_battery_individual_values() -> None:
    interactions = pd.DataFrame(
        {
            "signed_interaction_value": [-2.0, 3.0],
        }
    )
    individuals = pd.DataFrame(
        {
            "player": ["min_temp_c", "efficiency", "energy_capacity"],
            "signed_individual_value": [4.0, 1000.0, -2000.0],
        }
    )

    assert color_limit(interactions, individuals, None) == pytest.approx(4.0)
