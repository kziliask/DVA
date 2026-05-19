from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dva.analysis.run_ems_design_utility_experiment import (
    DEFAULT_HOLDOUT_HOURS,
    DesignGameSpec,
    EmsDesignSpec,
    _add_utility_columns,
    _build_configuration_summary,
    _build_design_dva_frames,
    _build_design_games,
    _build_target_designs,
    _resolve_solvers,
    _summarize_hourly_design_dva,
    _unique_designs_for_games,
    build_parser,
    compute_design_utilities,
    design_for_mask,
    design_players_for_designs,
    normalized_log_time_penalty,
)


def test_time_penalty_normalizes_log_scale_and_validates_bounds() -> None:
    assert normalized_log_time_penalty(0.0050) == pytest.approx(0.0)
    assert normalized_log_time_penalty(0.4056) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="greater"):
        normalized_log_time_penalty(0.1, time_min_seconds=0.2, time_max_seconds=0.2)
    with pytest.raises(ValueError, match="non-negative"):
        normalized_log_time_penalty(-0.1)


def test_compute_design_utilities_reports_net_and_runtime_values() -> None:
    utilities = compute_design_utilities(
        coverage=0.75,
        solve_time_seconds=0.4056,
        lambda_value=0.01,
        runtime_epsilon=1e-6,
    )

    assert utilities["coverage_utility"] == pytest.approx(0.75)
    assert utilities["time_penalty"] == pytest.approx(1.0)
    assert utilities["net_utility"] == pytest.approx(0.74)
    assert utilities["runtime_norm_utility"] == pytest.approx(-1.0)
    assert utilities["runtime_log_utility"] == pytest.approx(-np.log(0.405601))


def test_design_mask_uses_categorical_solver_and_binary_parameters() -> None:
    reference = EmsDesignSpec("exact", radius_km=1.0, facility_budget=3)
    target = EmsDesignSpec("greedy_max_cover", radius_km=2.0, facility_budget=5)
    players = design_players_for_designs(reference, target)
    game = DesignGameSpec(
        game_id="example",
        reference=reference,
        target=target,
        players=players,
    )

    assert players == ("solver", "radius_km", "facility_budget")
    assert design_for_mask(0, game=game) == reference
    assert design_for_mask(1, game=game) == EmsDesignSpec(
        "greedy_max_cover",
        radius_km=1.0,
        facility_budget=3,
    )
    assert design_for_mask(2, game=game) == EmsDesignSpec(
        "exact",
        radius_km=2.0,
        facility_budget=3,
    )
    assert design_for_mask(7, game=game) == target


def test_design_grid_skips_identity_game_but_keeps_identity_configuration() -> None:
    reference = EmsDesignSpec("exact", radius_km=1.0, facility_budget=3)
    targets = _build_target_designs(
        target_solvers=("exact", "greedy_max_cover"),
        coverage_radii_km=(1.0,),
        facility_budgets=(3, 5),
    )

    games = _build_design_games(reference, targets)
    unique_designs = _unique_designs_for_games(reference, targets, games)

    assert reference in unique_designs
    assert all(game.target != reference for game in games)
    assert {game.target.design_id for game in games} == {
        "exact_tau1_p5",
        "greedy_tau1_p3",
        "greedy_tau1_p5",
    }


def test_solver_alias_resolution_preserves_supported_order() -> None:
    assert _resolve_solvers(["greedy", "lp", "naive", "exact", "greedy"]) == (
        "greedy_max_cover",
        "lp_relaxation",
        "naive_greedy",
        "exact",
    )


def test_hourly_design_dva_is_additive_for_solver_game() -> None:
    reference = EmsDesignSpec("exact", radius_km=1.0, facility_budget=3)
    target = EmsDesignSpec("greedy_max_cover", radius_km=1.0, facility_budget=3)
    game = DesignGameSpec(
        game_id="exact_to_greedy",
        reference=reference,
        target=target,
        players=("solver",),
    )
    evaluation = _add_utility_columns(
        _toy_evaluation_frame(reference, target),
        lambda_values=(0.01,),
        time_min_seconds=0.0050,
        time_max_seconds=0.4056,
        runtime_epsilon=1e-6,
    )

    coalition_values, hourly = _build_design_dva_frames(
        design_games=(game,),
        evaluation=evaluation,
        lambda_values=(0.01,),
        model_id="xgb_001",
    )

    coverage = hourly.loc[hourly["utility_kind"].eq("coverage")]
    assert len(coalition_values) == 4
    assert coverage["player"].tolist() == ["solver", "solver"]
    np.testing.assert_allclose(coverage["dva_value"], [-0.05, 0.05], atol=1e-12)
    assert hourly["shapley_additivity_abs_error"].max() == pytest.approx(0.0)


def test_summary_bootstraps_by_game_utility_and_player() -> None:
    hourly = pd.DataFrame(
        [
            {
                "game_id": "g",
                "utility_kind": "coverage",
                "lambda_value": np.nan,
                "player": "solver",
                "reference_design_id": "exact_tau1_p3",
                "target_design_id": "greedy_tau1_p3",
                "dva_value": 0.1,
                "shapley_additivity_abs_error": 0.0,
                "baseline": "exact",
                "target": "greedy_max_cover",
                "reference_solver": "exact",
                "target_solver": "greedy_max_cover",
                "reference_coverage_radius_km": 1.0,
                "target_coverage_radius_km": 1.0,
                "reference_facility_budget": 3,
                "target_facility_budget": 3,
            },
            {
                "game_id": "g",
                "utility_kind": "coverage",
                "lambda_value": np.nan,
                "player": "solver",
                "reference_design_id": "exact_tau1_p3",
                "target_design_id": "greedy_tau1_p3",
                "dva_value": 0.3,
                "shapley_additivity_abs_error": 0.0,
                "baseline": "exact",
                "target": "greedy_max_cover",
                "reference_solver": "exact",
                "target_solver": "greedy_max_cover",
                "reference_coverage_radius_km": 1.0,
                "target_coverage_radius_km": 1.0,
                "reference_facility_budget": 3,
                "target_facility_budget": 3,
            },
        ]
    )

    summary = _summarize_hourly_design_dva(
        hourly,
        bootstrap_draws=100,
        bootstrap_seed=0,
    )

    assert summary.loc[0, "mean_dva_value"] == pytest.approx(0.2)
    assert summary.loc[0, "mean_abs_dva_value"] == pytest.approx(0.2)
    assert summary.loc[0, "dva_rank"] == 1
    assert summary.loc[0, "n_hours"] == 2


def test_configuration_summary_deduplicates_designs_across_hours() -> None:
    reference = EmsDesignSpec("exact", radius_km=1.0, facility_budget=3)
    target = EmsDesignSpec("greedy_max_cover", radius_km=1.0, facility_budget=3)
    evaluation = _add_utility_columns(
        _toy_evaluation_frame(reference, target),
        lambda_values=(0.01,),
        time_min_seconds=0.0050,
        time_max_seconds=0.4056,
        runtime_epsilon=1e-6,
    )

    summary = _build_configuration_summary(
        evaluation,
        lambda_values=(0.01,),
        primary_lambda=0.01,
    )

    assert summary["design_id"].tolist() == [
        "exact_tau1_p3",
        "greedy_tau1_p3",
    ]
    assert summary.loc[0, "hour_count"] == 2
    assert summary.loc[0, "mean_net_utility"] == pytest.approx(
        summary.loc[0, "mean_net_utility_lambda_0p01"]
    )


def test_parser_defaults_to_existing_ems_design_holdout_size() -> None:
    args = build_parser().parse_args([])

    assert args.holdout_hours == DEFAULT_HOLDOUT_HOURS == 100
    assert args.model_id == "xgb_001"
    assert args.target_solver is None
    assert args.repetitions == 3


def _toy_evaluation_frame(
    reference: EmsDesignSpec,
    target: EmsDesignSpec,
) -> pd.DataFrame:
    rows = []
    for hour_index, timestamp in enumerate(
        ["2025-01-01 00:00:00", "2025-01-01 01:00:00"]
    ):
        for design, coverage, seconds in (
            (reference, 0.80 + 0.05 * hour_index, 0.4056),
            (target, 0.75 + 0.15 * hour_index, 0.0050),
        ):
            rows.append(
                {
                    "model_id": "xgb_001",
                    "design_id": design.design_id,
                    "coverage_solver": design.solver,
                    "coverage_solver_label": design.solver,
                    "coverage_radius_km": design.radius_km,
                    "facility_budget": design.facility_budget,
                    "hour_index": hour_index,
                    "timestamp_hour": timestamp,
                    "predicted_total_demand": 10.0,
                    "actual_total_demand": 10.0,
                    "realized_covered_demand": 10.0 * coverage,
                    "realized_coverage": coverage,
                    "median_wall_clock_solve_seconds": seconds,
                    "mean_wall_clock_solve_seconds": seconds,
                    "min_wall_clock_solve_seconds": seconds,
                    "max_wall_clock_solve_seconds": seconds,
                    "median_solver_reported_runtime_seconds": np.nan,
                    "selected_facility_zip_codes": "[]",
                    "covered_zip_codes": "[]",
                    "selected_facility_count": 0,
                    "covered_zone_count": 0,
                    "solver_status": "ok",
                    "optimal": True,
                    "mip_gap": np.nan,
                }
            )
    return pd.DataFrame(rows)
