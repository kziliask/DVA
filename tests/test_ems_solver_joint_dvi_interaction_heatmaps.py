from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dva.plots.make_ems_joint_dvi_interaction_heatmaps import (
    FEATURES,
    INDIVIDUAL_VALUE_KEY,
)
from dva.plots.make_ems_solver_joint_dvi_interaction_heatmaps import (
    SOLVER_SCENARIOS,
    aggregate_individual_values,
    aggregate_interactions,
    load_individual_value_frame,
    load_interaction_frame,
    solver_scenario_matrix,
)


def _write_solver_scenario_outputs(
    root: Path,
    *,
    solver_comparison: str,
    value_mode: str,
) -> None:
    run_dir = root / "xgb_001" / f"{solver_comparison}_{value_mode}"
    run_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "timestamp_hour": "2025-08-01 01:00:00",
                "order": 2,
                "players": "hour|solver",
                "decision_interaction_value": 1.0,
                "interaction_type": "Cross-DVI",
                "value_mode": value_mode,
                "model_id": "xgb_001",
            },
            {
                "timestamp_hour": "2025-08-01 01:00:00",
                "order": 2,
                "players": "solver|temp_c",
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
    values.update({"hour": 10.0, "temp_c": 20.0, "solver": 30.0})
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


def test_ems_solver_joint_dvi_heatmap_uses_solver_comparison_dimension(
    tmp_path: Path,
) -> None:
    for _, solver_comparison, value_mode, _ in SOLVER_SCENARIOS:
        _write_solver_scenario_outputs(
            tmp_path,
            solver_comparison=solver_comparison,
            value_mode=value_mode,
        )

    interactions = aggregate_interactions(load_interaction_frame(tmp_path), "mean")
    individuals = aggregate_individual_values(load_individual_value_frame(tmp_path), "mean")

    matrix = solver_scenario_matrix(
        interactions,
        individuals,
        solver_comparison="exact_vs_greedy",
        value_mode="ante",
    )

    assert set(interactions["solver_comparison"]) == {
        "exact_vs_greedy",
        "exact_vs_naive",
    }
    assert matrix.shape == (len(FEATURES) + 1, 2)
    assert matrix.loc["hour", "solver"] == pytest.approx(1.0)
    assert matrix.loc["temp_c", "solver"] == pytest.approx(2.0)
    assert matrix.loc["hour", INDIVIDUAL_VALUE_KEY] == pytest.approx(10.0)
    assert matrix.loc["temp_c", INDIVIDUAL_VALUE_KEY] == pytest.approx(20.0)
    assert matrix.loc[INDIVIDUAL_VALUE_KEY, "solver"] == pytest.approx(30.0)
    assert np.isnan(matrix.loc[INDIVIDUAL_VALUE_KEY, INDIVIDUAL_VALUE_KEY])
