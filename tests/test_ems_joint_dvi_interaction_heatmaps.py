from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dva.plots.make_ems_joint_dvi_interaction_heatmaps import (
    DEFAULT_PERCENT_VMAX,
    FEATURES,
    INDIVIDUAL_VALUE_KEY,
    SCENARIOS,
    aggregate_individual_values,
    aggregate_interactions,
    build_parser,
    color_limit,
    design_parameters_present,
    load_individual_value_frame,
    load_interaction_frame,
    scenario_matrix,
)


def _write_scenario_outputs(root: Path, *, value_mode: str) -> None:
    run_dir = root / "xgb_001" / value_mode
    run_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "timestamp_hour": "2025-08-01 01:00:00",
                "order": 2,
                "players": "hour|staging_areas",
                "decision_interaction_value": 1.0,
                "interaction_type": "Cross-DVI",
                "value_mode": value_mode,
                "model_id": "xgb_001",
            },
            {
                "timestamp_hour": "2025-08-01 01:00:00",
                "order": 2,
                "players": "staging_areas|temp_c",
                "decision_interaction_value": 2.0,
                "interaction_type": "Cross-DVI",
                "value_mode": value_mode,
                "model_id": "xgb_001",
            },
            {
                "timestamp_hour": "2025-08-01 01:00:00",
                "order": 2,
                "players": "hour|day_of_week",
                "decision_interaction_value": 999.0,
                "interaction_type": "Info-Info",
                "value_mode": value_mode,
                "model_id": "xgb_001",
            },
        ]
    ).to_csv(run_dir / "dvi_interactions.csv", index=False)

    values: dict[str, float] = {feature: 0.0 for feature, _ in FEATURES}
    values.update({"hour": 10.0, "temp_c": 20.0, "staging_areas": 30.0})
    pd.DataFrame(
        [
            {
                "timestamp_hour": "2025-08-01 01:00:00",
                "player": player,
                "dva_value": value,
                "value_mode": value_mode,
                "model_id": "xgb_001",
            }
            for player, value in values.items()
        ]
    ).to_csv(run_dir / "joint_dva.csv", index=False)


def test_ems_joint_dvi_heatmap_appends_individual_value_margins(
    tmp_path: Path,
) -> None:
    for _, value_mode, _ in SCENARIOS:
        _write_scenario_outputs(tmp_path, value_mode=value_mode)

    interactions = aggregate_interactions(load_interaction_frame(tmp_path), "mean")
    individuals = aggregate_individual_values(load_individual_value_frame(tmp_path), "mean")
    design_parameters = design_parameters_present(interactions, individuals)

    matrix = scenario_matrix(
        interactions,
        individuals,
        value_mode="ante",
        design_parameters=design_parameters,
    )

    assert design_parameters == (("staging_areas", "Staging areas"),)
    assert matrix.shape == (len(FEATURES) + 1, 2)
    assert matrix.loc["hour", "staging_areas"] == pytest.approx(1.0)
    assert matrix.loc["temp_c", "staging_areas"] == pytest.approx(2.0)
    assert matrix.loc["hour", INDIVIDUAL_VALUE_KEY] == pytest.approx(10.0)
    assert matrix.loc["temp_c", INDIVIDUAL_VALUE_KEY] == pytest.approx(20.0)
    assert matrix.loc[INDIVIDUAL_VALUE_KEY, "staging_areas"] == pytest.approx(30.0)
    assert np.isnan(matrix.loc[INDIVIDUAL_VALUE_KEY, INDIVIDUAL_VALUE_KEY])


def test_ems_joint_dvi_color_limit_excludes_design_individual_values() -> None:
    interactions = pd.DataFrame(
        {
            "signed_interaction_value": [-0.02, 0.03],
        }
    )
    individuals = pd.DataFrame(
        {
            "player": ["hour", "staging_areas"],
            "signed_individual_value": [0.04, 1000.0],
        }
    )

    assert color_limit(interactions, individuals, None) == pytest.approx(0.04)


def test_ems_joint_dvi_default_colorbar_spans_full_percent_range() -> None:
    assert build_parser().parse_args([]).vmax == pytest.approx(DEFAULT_PERCENT_VMAX)
