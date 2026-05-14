from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dva.data_scripts.ems.build_ems_zip_hour_features import (
    build_zip_alias_map,
    canonicalize_zip_hourly,
)
from dva.data_scripts.ems.build_ems_zip_wide_model_table import filter_excluded_zip_codes
from dva.analysis.ems_exact_shap import (
    EmsExactShapConfig,
    EmsExactShapOutputs,
    GroupedBackgroundCoalitionPredictor,
    _build_coalition_demand_scenarios,
    _build_training_residual_matrix,
    _build_time_split,
    _load_ems_frames,
    _maximum_coverage_decision_changed,
    _sample_explanation_hours,
    _sample_residual_scenarios,
    _solve_cvar_decision_values,
    _solve_decision_values,
    _validate_config,
    build_coverage_matrix,
    build_ems_feature_groups,
    compute_exact_shapley_values,
    compute_kernel_shapley_values,
    compute_permutation_shapley_values,
    load_ems_exact_shap_outputs,
    run_ems_exact_shap,
    solve_cvar_coverage,
    solve_ems_coverage,
    solve_greedy_max_cover_coverage,
    solve_gurobi_lp_relaxation_coverage,
    solve_maximum_coverage,
    solve_naive_greedy_coverage,
    write_ems_exact_shap_outputs,
)
from dva.analysis.evaluation_metrics import compute_decision_activation_metrics
from dva.analysis.run_ems_exact_shap import (
    build_parser as build_ems_exact_shap_parser,
)


def _require_gurobi() -> None:
    gp = pytest.importorskip("gurobipy")
    try:
        model = gp.Model()
        model.Params.OutputFlag = 0
        model.dispose()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Gurobi unavailable: {exc}")


def _additive_coalition_game(feature_values: tuple[float, ...]) -> np.ndarray:
    coalition_values = np.zeros(1 << len(feature_values), dtype=float)
    for coalition_mask in range(len(coalition_values)):
        coalition_values[coalition_mask] = sum(
            feature_value
            for feature_idx, feature_value in enumerate(feature_values)
            if coalition_mask & (1 << feature_idx)
        )
    return coalition_values


def test_build_ems_feature_groups_drops_month_and_groups_zip_features() -> None:
    feature_columns = (
        "hour",
        "month",
        "day_of_week",
        "temp_c",
        "precip_mm",
        "citywide_ems_incidents_lag_1",
        "ems_incidents_lag_1_zip_10001",
        "ems_incidents_lag_1_zip_10002",
        "neighbor_ems_incidents_lag_1_mean_zip_10001",
        "neighbor_ems_incidents_lag_1_mean_zip_10002",
        "zone_hour_baseline_zip_10001",
        "zone_hour_baseline_zip_10002",
    )

    groups = build_ems_feature_groups(feature_columns)

    assert tuple(group.name for group in groups) == (
        "hour",
        "day_of_week",
        "temp_c",
        "precip_mm",
        "citywide_ems_incidents_lag_1",
        "ems_incidents_lag_1",
        "neighbor_ems_incidents_lag_1_mean",
        "zone_hour_baseline",
    )
    grouped_columns = {group.name: group.columns for group in groups}
    assert "month" not in grouped_columns
    assert grouped_columns["ems_incidents_lag_1"] == (
        "ems_incidents_lag_1_zip_10001",
        "ems_incidents_lag_1_zip_10002",
    )
    assert grouped_columns["zone_hour_baseline"] == (
        "zone_hour_baseline_zip_10001",
        "zone_hour_baseline_zip_10002",
    )


def test_zip_aliases_roll_up_to_canonical_polygon_outputs() -> None:
    zip_geo = pd.DataFrame(
        {
            "zip_code": ["10001", "10005"],
            "alias_zip_codes": ["10001|10118|10119", "10005|10271"],
        }
    )
    zip_hourly = pd.DataFrame(
        {
            "timestamp_hour": pd.to_datetime(
                [
                    "2025-01-01 00:00:00",
                    "2025-01-01 00:00:00",
                    "2025-01-01 00:00:00",
                    "2025-01-01 01:00:00",
                ]
            ),
            "zip_code": ["10001", "10118", "10271", "10119"],
            "ems_incident_count": [2, 3, 5, 7],
        }
    )

    alias_map = build_zip_alias_map(zip_geo)
    canonicalized, alias_rollup = canonicalize_zip_hourly(zip_hourly, zip_geo)

    assert alias_map["10118"] == "10001"
    assert alias_map["10271"] == "10005"
    assert set(canonicalized["zip_code"]) == {"10001", "10005"}
    first_hour = canonicalized.loc[
        canonicalized["timestamp_hour"].eq(pd.Timestamp("2025-01-01 00:00:00"))
    ].set_index("zip_code")
    assert first_hour.loc["10001", "ems_incident_count"] == 5
    assert first_hour.loc["10005", "ems_incident_count"] == 5
    assert alias_rollup["rolled_up_ems_incidents"].sum() == 15


def test_wide_table_builder_filters_excluded_zip_codes() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_hour": pd.to_datetime(
                ["2025-01-01 00:00:00", "2025-01-01 00:00:00"]
            ),
            "zip_code": ["10001", "10468"],
            "ems_incident_count": [3, 9],
        }
    )

    filtered = filter_excluded_zip_codes(frame)

    assert filtered["zip_code"].tolist() == ["10001"]


def test_ems_exact_shap_rejects_stale_inputs_with_excluded_zip(tmp_path: Path) -> None:
    timestamp = pd.Timestamp("2025-01-01 00:00:00")
    x_path = tmp_path / "x.csv"
    y_path = tmp_path / "y.csv"
    metadata_path = tmp_path / "metadata.json"
    pd.DataFrame(
        {
            "timestamp_hour": [timestamp],
            "ems_incidents_lag_1_zip_10468": [1.0],
        }
    ).to_csv(x_path, index=False)
    pd.DataFrame(
        {
            "timestamp_hour": [timestamp],
            "target_ems_incident_count_zip_10468": [1.0],
        }
    ).to_csv(y_path, index=False)
    metadata_path.write_text(
        json.dumps({"zip_codes": ["10468"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="excluded ZIP codes"):
        _load_ems_frames(
            EmsExactShapConfig(
                x_path=x_path,
                y_path=y_path,
                metadata_path=metadata_path,
            )
        )


def test_ems_exact_shap_validates_cvar_config() -> None:
    with pytest.raises(ValueError, match="cvar_alpha"):
        _validate_config(EmsExactShapConfig(cvar_alpha=1.0))
    with pytest.raises(ValueError, match="cvar_scenario_count"):
        _validate_config(EmsExactShapConfig(cvar_scenario_count=0))


def test_ems_exact_shap_validates_decision_approximation_sample_counts() -> None:
    with pytest.raises(ValueError, match="decision_permutation_shap_samples"):
        _validate_config(EmsExactShapConfig(decision_permutation_shap_samples=(0,)))
    with pytest.raises(ValueError, match="decision_kernel_shap_samples"):
        _validate_config(EmsExactShapConfig(decision_kernel_shap_samples=(-1,)))
    with pytest.raises(ValueError, match="decision_permutation_shap_seed"):
        _validate_config(EmsExactShapConfig(decision_permutation_shap_seed=True))
    with pytest.raises(ValueError, match="decision_kernel_shap_seed"):
        _validate_config(EmsExactShapConfig(decision_kernel_shap_seed=True))


def test_ems_exact_runner_parses_decision_approximation_sample_counts() -> None:
    args = build_ems_exact_shap_parser().parse_args(
        [
            "--decision-permutation-shap-samples",
            "32",
            "128",
            "--decision-kernel-shap-samples",
            "64",
            "--decision-permutation-shap-seed",
            "17",
            "--decision-kernel-shap-seed",
            "29",
        ]
    )

    assert args.decision_permutation_shap_samples == [32, 128]
    assert args.decision_kernel_shap_samples == [64]
    assert args.decision_permutation_shap_seed == 17
    assert args.decision_kernel_shap_seed == 29


def test_build_coverage_matrix_thresholds_toy_distances() -> None:
    distance_matrix = pd.DataFrame(
        [
            [0.0, 0.5, 2.0],
            [0.5, 0.0, 1.2],
            [2.0, 1.2, 0.0],
        ],
        index=["10001", "10002", "10003"],
        columns=["10001", "10002", "10003"],
    )

    coverage = build_coverage_matrix(
        distance_matrix,
        ("10001", "10002", "10003"),
        coverage_radius_km=1.0,
    )

    np.testing.assert_array_equal(
        coverage,
        np.array(
            [
                [True, True, False],
                [True, True, False],
                [False, False, True],
            ],
        ),
    )


def test_solve_maximum_coverage_selects_best_facility_on_toy_instance() -> None:
    _require_gurobi()

    coverage_matrix = np.array(
        [
            [True, False, False],
            [True, False, False],
            [False, True, False],
        ],
        dtype=bool,
    )

    result = solve_maximum_coverage(
        np.array([5.0, 4.0, 10.0]),
        coverage_matrix,
        ("10001", "10002", "10003"),
        facility_budget=1,
    )

    assert result.optimal
    assert result.objective_value == pytest.approx(10.0 / 19.0)
    assert result.covered_demand == pytest.approx(10.0)
    assert result.total_demand == pytest.approx(19.0)
    assert result.selected_facility_zip_codes == ("10002",)
    assert result.covered_zip_codes == ("10003",)


def test_solve_maximum_coverage_uses_stable_tie_break() -> None:
    _require_gurobi()

    coverage_matrix = np.array([[True, True]], dtype=bool)

    result = solve_maximum_coverage(
        np.array([5.0]),
        coverage_matrix,
        ("10001", "10002"),
        facility_budget=1,
        objective_tolerance=0.0,
    )

    assert result.optimal
    assert result.objective_value == pytest.approx(1.0)
    assert result.covered_demand == pytest.approx(5.0)
    assert result.total_demand == pytest.approx(5.0)
    assert result.selected_facility_zip_codes == ("10001",)
    assert result.covered_zip_codes == ("10001",)


def test_cvar_residual_scenario_helpers_preserve_shapes_and_seed() -> None:
    class DummyModel:
        def predict(self, feature_matrix: np.ndarray) -> np.ndarray:
            return np.column_stack(
                (
                    feature_matrix[:, 0] + 1.0,
                    feature_matrix[:, 1] + 2.0,
                )
            )

    train_frame = pd.DataFrame({"feature_a": [0.0, 1.0], "feature_b": [2.0, 3.0]})
    train_y = pd.DataFrame({"target_a": [2.0, 2.5], "target_b": [5.0, 6.5]})

    residual_matrix = _build_training_residual_matrix(
        model=DummyModel(),
        train_frame=train_frame,
        train_y=train_y,
        feature_columns=("feature_a", "feature_b"),
        target_columns=("target_a", "target_b"),
    )
    residual_scenarios = _sample_residual_scenarios(
        residual_matrix,
        scenario_count=5,
        random_state=13,
    )
    repeated_scenarios = _sample_residual_scenarios(
        residual_matrix,
        scenario_count=5,
        random_state=13,
    )
    coalition_predictions = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    coalition_scenarios = _build_coalition_demand_scenarios(
        coalition_predictions,
        residual_scenarios,
    )

    np.testing.assert_allclose(
        residual_matrix,
        np.array([[1.0, 1.0], [0.5, 1.5]], dtype=float),
    )
    np.testing.assert_allclose(residual_scenarios, repeated_scenarios)
    assert residual_scenarios.shape == (5, 2)
    assert coalition_scenarios.shape == (2, 5, 2)
    assert np.all(coalition_scenarios >= 0.0)


def test_solve_cvar_coverage_uses_tail_percentage_reward() -> None:
    _require_gurobi()

    coverage_matrix = np.eye(2, dtype=bool)
    demand_scenarios = np.array(
        [
            [60.0, 40.0],
            [0.0, 10.0],
        ],
        dtype=float,
    )

    deterministic_result = solve_maximum_coverage(
        demand_scenarios.mean(axis=0),
        coverage_matrix,
        ("10001", "10002"),
        facility_budget=1,
        objective_tolerance=0.0,
    )
    cvar_result = solve_cvar_coverage(
        demand_scenarios,
        coverage_matrix,
        ("10001", "10002"),
        facility_budget=1,
        alpha=0.0,
        objective_tolerance=0.0,
    )

    assert deterministic_result.selected_facility_zip_codes == ("10001",)
    assert cvar_result.selected_facility_zip_codes == ("10002",)
    assert deterministic_result.covered_demand > cvar_result.covered_demand
    assert cvar_result.objective_value == pytest.approx(25.0 / 55.0)
    assert cvar_result.risk_objective_value == pytest.approx(0.7)


def test_solve_naive_greedy_coverage_picks_highest_demand_zones() -> None:
    coverage_matrix = np.array(
        [
            [True, True, False],
            [False, True, False],
            [False, True, True],
        ],
        dtype=bool,
    )

    result = solve_naive_greedy_coverage(
        np.array([10.0, 1.0, 1.0]),
        coverage_matrix,
        ("10001", "10002", "10003"),
        facility_budget=1,
    )

    assert result.solver_name == "naive_greedy"
    assert not result.optimal
    assert result.objective_value == pytest.approx(10.0 / 12.0)
    assert result.covered_demand == pytest.approx(10.0)
    assert result.selected_facility_zip_codes == ("10001",)
    assert result.covered_zip_codes == ("10001",)


def test_solve_greedy_max_cover_coverage_picks_largest_marginal_cover() -> None:
    coverage_matrix = np.array(
        [
            [True, True, False],
            [False, True, False],
            [False, True, True],
        ],
        dtype=bool,
    )

    result = solve_greedy_max_cover_coverage(
        np.array([10.0, 1.0, 1.0]),
        coverage_matrix,
        ("10001", "10002", "10003"),
        facility_budget=1,
    )

    assert result.solver_name == "greedy_max_cover"
    assert not result.optimal
    assert result.objective_value == pytest.approx(1.0)
    assert result.covered_demand == pytest.approx(12.0)
    assert result.selected_facility_zip_codes == ("10002",)
    assert result.covered_zip_codes == ("10001", "10002", "10003")


def test_solve_gurobi_lp_relaxation_coverage_rounds_relaxed_solution() -> None:
    _require_gurobi()

    coverage_matrix = np.eye(3, dtype=bool)

    result = solve_gurobi_lp_relaxation_coverage(
        np.array([1.0, 5.0, 3.0]),
        coverage_matrix,
        ("10001", "10002", "10003"),
        facility_budget=2,
        objective_tolerance=0.0,
    )
    alias_result = solve_ems_coverage(
        np.array([1.0, 5.0, 3.0]),
        coverage_matrix,
        ("10001", "10002", "10003"),
        facility_budget=2,
        solver_name="linear-relaxation",
        objective_tolerance=0.0,
    )

    assert result.solver_name == "gurobi_lp_relaxation"
    assert alias_result.solver_name == "gurobi_lp_relaxation"
    assert not result.optimal
    assert result.selected_facility_zip_codes == ("10002", "10003")
    assert result.covered_zip_codes == ("10002", "10003")
    assert result.covered_demand == pytest.approx(8.0)
    assert result.objective_value == pytest.approx(8.0 / 9.0)


def test_decision_shap_matches_decision_value_game_on_toy_boundary() -> None:
    _require_gurobi()

    coalition_demand_matrix = np.array(
        [
            [2.0, 1.0],
            [1.0, 2.0],
            [4.0, 1.0],
            [3.0, 4.0],
        ],
        dtype=float,
    )
    decision_values, _, _, decision_solutions = _solve_decision_values(
        coalition_demand_matrix=coalition_demand_matrix,
        true_demand=np.array([0.0, 5.0], dtype=float),
        coverage_matrix=np.eye(2, dtype=bool),
        zip_codes=("10001", "10002"),
        config=EmsExactShapConfig(
            facility_budget=1,
            progress_every_coalitions=0,
            objective_tolerance=0.0,
        ),
    )
    decision_game_values = decision_values - decision_values[0]
    decision_shap = compute_exact_shapley_values(decision_game_values, feature_count=2)
    activation = compute_decision_activation_metrics(
        decision_game_values,
        decision_solutions,
        feature_count=2,
        decision_changed=_maximum_coverage_decision_changed,
    )

    np.testing.assert_allclose(decision_values, np.array([0.0, 1.0, 0.0, 1.0]))
    np.testing.assert_allclose(decision_shap, np.array([1.0, 0.0]))
    np.testing.assert_allclose(activation.activation_rate, np.array([1.0, 0.0]))
    np.testing.assert_allclose(activation.activated_value_sum, np.array([1.0, 0.0]))
    np.testing.assert_allclose(activation.activated_value, np.array([1.0, 0.0]))
    assert decision_shap.sum() == pytest.approx(decision_game_values[-1])


def test_permutation_shapley_values_are_additive_on_additive_game() -> None:
    coalition_values = _additive_coalition_game((1.5, -2.0, 3.25))

    shap_values = compute_permutation_shapley_values(
        coalition_values,
        feature_count=3,
        sample_count=7,
        random_state=123,
    )

    np.testing.assert_allclose(shap_values, np.array([1.5, -2.0, 3.25]))
    assert shap_values.sum() == pytest.approx(coalition_values[-1] - coalition_values[0])


def test_kernel_shapley_values_are_additive_on_additive_game() -> None:
    coalition_values = _additive_coalition_game((2.5, -0.75))

    shap_values = compute_kernel_shapley_values(
        coalition_values,
        feature_count=2,
        sample_count=8,
        random_state=11,
    )

    np.testing.assert_allclose(shap_values, np.array([2.5, -0.75]))
    assert shap_values.sum() == pytest.approx(coalition_values[-1] - coalition_values[0])


def test_kernel_shapley_values_use_all_interior_coalitions_without_replacement() -> None:
    coalition_values = np.array([0.0, 1.0, -0.5, 0.7, 2.0, 2.8, 1.5, 3.0])
    exact_shap = compute_exact_shapley_values(coalition_values, feature_count=3)

    first_seed = compute_kernel_shapley_values(
        coalition_values,
        feature_count=3,
        sample_count=6,
        random_state=0,
    )
    second_seed = compute_kernel_shapley_values(
        coalition_values,
        feature_count=3,
        sample_count=6,
        random_state=1,
    )
    replacement_budget = compute_kernel_shapley_values(
        coalition_values,
        feature_count=3,
        sample_count=7,
        random_state=0,
    )

    np.testing.assert_allclose(first_seed, exact_shap)
    np.testing.assert_allclose(second_seed, exact_shap)
    assert replacement_budget.sum() == pytest.approx(coalition_values[-1])


def test_decision_shap_approximations_are_reproducible_for_seed() -> None:
    coalition_values = np.array([0.0, 1.0, -0.5, 0.7, 2.0, 2.8, 1.5, 3.0])

    first_permutation = compute_permutation_shapley_values(
        coalition_values,
        feature_count=3,
        sample_count=12,
        random_state=[4, 5, 6],
    )
    second_permutation = compute_permutation_shapley_values(
        coalition_values,
        feature_count=3,
        sample_count=12,
        random_state=[4, 5, 6],
    )
    first_kernel = compute_kernel_shapley_values(
        coalition_values,
        feature_count=3,
        sample_count=16,
        random_state=[7, 8, 9],
    )
    second_kernel = compute_kernel_shapley_values(
        coalition_values,
        feature_count=3,
        sample_count=16,
        random_state=[7, 8, 9],
    )

    np.testing.assert_allclose(first_permutation, second_permutation)
    np.testing.assert_allclose(first_kernel, second_kernel)
    assert first_permutation.sum() == pytest.approx(coalition_values[-1])
    assert first_kernel.sum() == pytest.approx(coalition_values[-1])


def test_decision_shap_approximations_can_vary_by_seed() -> None:
    coalition_values = np.array([0.0, 1.0, -0.5, 0.7, 2.0, 2.8, 1.5, 3.0])

    permutation_seed_1 = compute_permutation_shapley_values(
        coalition_values,
        feature_count=3,
        sample_count=2,
        random_state=0,
    )
    permutation_seed_2 = compute_permutation_shapley_values(
        coalition_values,
        feature_count=3,
        sample_count=2,
        random_state=1,
    )
    kernel_seed_1 = compute_kernel_shapley_values(
        coalition_values,
        feature_count=3,
        sample_count=2,
        random_state=0,
    )
    kernel_seed_2 = compute_kernel_shapley_values(
        coalition_values,
        feature_count=3,
        sample_count=2,
        random_state=1,
    )

    assert not np.allclose(permutation_seed_1, permutation_seed_2)
    assert not np.allclose(kernel_seed_1, kernel_seed_2)
    assert permutation_seed_1.sum() == pytest.approx(coalition_values[-1])
    assert permutation_seed_2.sum() == pytest.approx(coalition_values[-1])
    assert kernel_seed_1.sum() == pytest.approx(coalition_values[-1])
    assert kernel_seed_2.sum() == pytest.approx(coalition_values[-1])


def test_zero_residual_cvar_decision_shap_matches_deterministic_path() -> None:
    _require_gurobi()

    coalition_demand_matrix = np.array(
        [
            [2.0, 1.0],
            [1.0, 2.0],
            [4.0, 1.0],
            [3.0, 4.0],
        ],
        dtype=float,
    )
    true_demand = np.array([0.0, 5.0], dtype=float)
    config = EmsExactShapConfig(
        facility_budget=1,
        progress_every_coalitions=0,
        objective_tolerance=0.0,
        cvar_alpha=0.5,
        cvar_scenario_count=3,
    )
    decision_values, _, _, decision_solutions = _solve_decision_values(
        coalition_demand_matrix=coalition_demand_matrix,
        true_demand=true_demand,
        coverage_matrix=np.eye(2, dtype=bool),
        zip_codes=("10001", "10002"),
        config=config,
    )
    coalition_demand_scenarios = np.repeat(
        coalition_demand_matrix[:, np.newaxis, :],
        config.cvar_scenario_count,
        axis=1,
    )
    cvar_decision_values, _, _, cvar_decision_solutions = _solve_cvar_decision_values(
        coalition_demand_scenarios=coalition_demand_scenarios,
        true_demand=true_demand,
        coverage_matrix=np.eye(2, dtype=bool),
        zip_codes=("10001", "10002"),
        config=config,
    )
    decision_shap = compute_exact_shapley_values(
        decision_values - decision_values[0],
        feature_count=2,
    )
    cvar_decision_shap = compute_exact_shapley_values(
        cvar_decision_values - cvar_decision_values[0],
        feature_count=2,
    )
    decision_activation = compute_decision_activation_metrics(
        decision_values - decision_values[0],
        decision_solutions,
        feature_count=2,
        decision_changed=_maximum_coverage_decision_changed,
    )
    cvar_decision_activation = compute_decision_activation_metrics(
        cvar_decision_values - cvar_decision_values[0],
        cvar_decision_solutions,
        feature_count=2,
        decision_changed=_maximum_coverage_decision_changed,
    )

    np.testing.assert_allclose(cvar_decision_values, decision_values)
    np.testing.assert_allclose(cvar_decision_shap, decision_shap)
    np.testing.assert_allclose(
        cvar_decision_activation.activation_rate,
        decision_activation.activation_rate,
    )
    np.testing.assert_allclose(
        cvar_decision_activation.activated_value_sum,
        decision_activation.activated_value_sum,
    )
    np.testing.assert_allclose(
        cvar_decision_activation.activated_value,
        decision_activation.activated_value,
    )


def test_grouped_coalition_predictor_preserves_multioutput_shape_and_efficiency() -> None:
    class MultiOutputModel:
        def predict(self, X: np.ndarray) -> np.ndarray:
            X_array = np.asarray(X, dtype=float)
            base = X_array @ np.array([0.5, 1.0, 1.5], dtype=float)
            return np.column_stack((base, base + 1.0, 2.0 * base))

    feature_names = (
        "hour",
        "ems_incidents_lag_1_zip_10001",
        "ems_incidents_lag_1_zip_10002",
    )
    feature_groups = build_ems_feature_groups(feature_names)
    x_frame = pd.DataFrame(
        {
            "hour": [0.0, 1.0, 2.0, 3.0],
            "ems_incidents_lag_1_zip_10001": [1.0, 2.0, 3.0, 4.0],
            "ems_incidents_lag_1_zip_10002": [0.0, 1.0, 0.0, 1.0],
        }
    )
    model = MultiOutputModel()
    predictor = GroupedBackgroundCoalitionPredictor(
        model,
        feature_names,
        feature_groups,
        x_frame.iloc[:2],
        coalition_batch_size=2,
    )

    coalition_predictions = predictor.predict_all_coalitions(
        x_frame.iloc[3],
        progress_every_coalitions=0,
    )
    shap_values = compute_exact_shapley_values(
        coalition_predictions,
        feature_count=len(feature_groups),
    )

    assert coalition_predictions.shape == (4, 3)
    np.testing.assert_allclose(
        shap_values.sum(axis=0),
        coalition_predictions[-1] - coalition_predictions[0],
    )


def test_ems_time_split_samples_explanations_from_final_month() -> None:
    timestamps = pd.date_range("2025-01-30 00:00:00", periods=120, freq="h")
    x_frame = pd.DataFrame(
        {
            "timestamp_hour": timestamps,
            "feature": np.arange(len(timestamps), dtype=float),
        }
    )
    y_frame = pd.DataFrame(
        {
            "timestamp_hour": timestamps,
            "target": np.arange(len(timestamps), dtype=float),
        }
    )

    split = _build_time_split(x_frame, y_frame, test_months=1)
    explain_x, explain_y, explained_rows = _sample_explanation_hours(
        split.holdout_x,
        split.holdout_y,
        split.holdout_source_rows,
        holdout_hours=24,
        max_hours=None,
        random_state=0,
    )

    assert split.train_x["timestamp_hour"].max() == pd.Timestamp("2025-01-31 23:00:00")
    assert split.holdout_x["timestamp_hour"].min() == pd.Timestamp("2025-02-01 00:00:00")
    assert len(explain_x) == 24
    assert len(explain_y) == 24
    assert len({row["timestamp_hour"] for row in explained_rows}) == 24
    assert len({pd.Timestamp(row["timestamp_hour"]).date() for row in explained_rows}) > 1
    assert [row["source_row_position"] for row in explained_rows] != list(range(48, 72))
    assert explain_x["timestamp_hour"].tolist() == [
        pd.Timestamp(row["timestamp_hour"]) for row in explained_rows
    ]


def test_run_ems_exact_shap_smoke_writes_expected_outputs(tmp_path: Path) -> None:
    _require_gurobi()
    paths = _write_toy_ems_case(tmp_path)
    config = EmsExactShapConfig(
        x_path=paths["x"],
        y_path=paths["y"],
        metadata_path=paths["metadata"],
        zone_order_path=paths["zone_order"],
        distance_matrix_path=paths["distance"],
        outdir=tmp_path / "out",
        holdout_hours=1,
        test_months=1,
        background_rows=2,
        coalition_batch_size=32,
        progress_every_coalitions=0,
        xgb_n_estimators=2,
        xgb_max_depth=1,
        xgb_learning_rate=0.3,
        n_jobs=1,
        coverage_radius_km=1.0,
        facility_budget=1,
        save_coalition_values=True,
        cvar_alpha=0.5,
        cvar_scenario_count=3,
        decision_permutation_shap_samples=(3, 3),
        decision_kernel_shap_samples=(4, 4),
        decision_permutation_shap_seed=17,
        decision_kernel_shap_seed=29,
    )

    outputs = run_ems_exact_shap(config)
    write_ems_exact_shap_outputs(outputs, config.outdir)
    reloaded_outputs = load_ems_exact_shap_outputs(config.outdir)

    expected_players = {
        "hour",
        "day_of_week",
        "temp_c",
        "precip_mm",
        "citywide_ems_incidents_lag_1",
        "ems_incidents_lag_1",
        "neighbor_ems_incidents_lag_1_mean",
        "zone_hour_baseline",
    }
    hourly_row = outputs.hourly_shap.iloc[0]
    reloaded_hourly_row = reloaded_outputs.hourly_shap.iloc[0]
    predictive_shap_sum = sum(
        float(hourly_row[f"predictive_shap_{player_name}"])
        for player_name in expected_players
    )
    decision_shap_sum = sum(
        float(hourly_row[f"decision_shap_{player_name}"])
        for player_name in expected_players
    )
    decision_permutation_shap_sum = sum(
        float(hourly_row[f"decision_permutation_shap_3_{player_name}"])
        for player_name in expected_players
    )
    decision_kernel_shap_sum = sum(
        float(hourly_row[f"decision_kernel_shap_4_{player_name}"])
        for player_name in expected_players
    )

    assert predictive_shap_sum == pytest.approx(hourly_row["predictive_total_gain"])
    assert decision_shap_sum == pytest.approx(hourly_row["decision_value_gain"])
    assert decision_permutation_shap_sum == pytest.approx(
        hourly_row["decision_value_gain"]
    )
    assert decision_kernel_shap_sum == pytest.approx(hourly_row["decision_value_gain"])
    assert 0.0 <= hourly_row["decision_full_value"] <= 1.0
    assert 0.0 <= hourly_row["oracle_value"] <= 1.0
    assert "actual_total_demand" in hourly_row.index
    assert "decision_baseline_covered_demand" in hourly_row.index
    assert "decision_full_covered_demand" in hourly_row.index
    assert "oracle_covered_demand" in hourly_row.index
    assert "predictive_decision_deletion_auc" in hourly_row.index
    assert "decision_decision_deletion_auc" in hourly_row.index
    assert "cvar_decision_value_gain" in hourly_row.index
    assert "cvar_decision_baseline_covered_demand" in hourly_row.index
    assert "cvar_decision_full_covered_demand" in hourly_row.index
    assert "cvar_decision_shap_hour" in hourly_row.index
    assert "decision_permutation_shap_3_hour" in reloaded_hourly_row.index
    assert "decision_kernel_shap_4_hour" in reloaded_hourly_row.index
    assert "decision_activation_rate_hour" in hourly_row.index
    assert "decision_activated_value_sum_hour" in hourly_row.index
    assert "decision_activated_value_hour" in hourly_row.index
    assert "cvar_decision_activation_rate_hour" in hourly_row.index
    assert "cvar_decision_activated_value_sum_hour" in hourly_row.index
    assert "cvar_decision_activated_value_hour" in hourly_row.index
    assert 0.0 <= hourly_row["decision_activation_rate_hour"] <= 1.0
    assert 0.0 <= hourly_row["cvar_decision_activation_rate_hour"] <= 1.0
    assert "predictive_decision_deletion_auc" in outputs.evaluation_metrics
    assert "decision_decision_deletion_auc" in outputs.evaluation_metrics
    assert len(outputs.hourly_shap) == 1
    assert outputs.run_metadata["test_months"] == 1
    assert outputs.run_metadata["compute_cvar_decision_shap"] is True
    assert outputs.run_metadata["cvar_alpha"] == pytest.approx(0.5)
    assert outputs.run_metadata["cvar_scenario_count"] == 3
    assert outputs.run_metadata["cvar_predictive_model_changed"] is False
    assert outputs.run_metadata["decision_permutation_shap_samples"] == [3]
    assert outputs.run_metadata["decision_kernel_shap_samples"] == [4]
    assert outputs.run_metadata["decision_permutation_shap_seed"] == 17
    assert outputs.run_metadata["decision_kernel_shap_seed"] == 29
    assert outputs.run_metadata["holdout_rows"] == 24
    assert len(outputs.run_metadata["explained_rows"]) == 1
    assert outputs.run_metadata["explained_rows"][0]["timestamp_hour"] == hourly_row[
        "timestamp_hour"
    ]
    assert set(outputs.summary_shap["feature"]) == expected_players
    assert "decision_activation_rate" in outputs.summary_shap.columns
    assert "decision_activated_value" in outputs.summary_shap.columns
    assert outputs.summary_shap["decision_activation_rate"].between(0.0, 1.0).all()
    assert outputs.cvar_summary_shap is not None
    assert set(outputs.cvar_summary_shap["feature"]) == expected_players
    assert "cvar_decision_activation_rate" in outputs.cvar_summary_shap.columns
    assert "cvar_decision_activated_value" in outputs.cvar_summary_shap.columns
    assert outputs.cvar_summary_shap["cvar_decision_activation_rate"].between(0.0, 1.0).all()
    assert reloaded_outputs.cvar_summary_shap is not None
    assert set(reloaded_outputs.cvar_summary_shap["feature"]) == expected_players
    assert len(outputs.predictive_zip_shap) == len(expected_players) * 3
    assert set(outputs.coverage_solutions["solution_type"]) == {
        "baseline_model",
        "cvar_baseline_model",
        "cvar_full_model",
        "full_model",
        "oracle_truth",
    }
    assert "risk_objective_value" in outputs.coverage_solutions.columns
    assert "predicted_covered_demand" in outputs.coverage_solutions.columns
    assert "predicted_total_demand" in outputs.coverage_solutions.columns
    assert "realized_covered_demand" in outputs.coverage_solutions.columns
    assert "actual_total_demand" in outputs.coverage_solutions.columns
    assert outputs.coalition_values is not None
    assert len(outputs.coalition_values) == 1 << len(expected_players)
    assert "cvar_decision_value" in outputs.coalition_values.columns
    assert "decision_selected_facility_indices" in outputs.coalition_values.columns
    assert "cvar_decision_selected_facility_indices" in outputs.coalition_values.columns
    assert (config.outdir / "hourly_shap.csv").exists()
    assert (config.outdir / "predictive_zip_shap.csv").exists()
    assert (config.outdir / "coverage_solutions.csv").exists()
    assert (config.outdir / "cvar_summary_shap.csv").exists()
    assert (config.outdir / "coalition_values.csv").exists()


def test_write_ems_exact_shap_outputs_removes_stale_optional_files(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "out"
    outdir.mkdir()
    (outdir / "cvar_summary_shap.csv").write_text("stale cvar\n", encoding="utf-8")
    (outdir / "coalition_values.csv").write_text("stale coalitions\n", encoding="utf-8")

    outputs = EmsExactShapOutputs(
        hourly_shap=pd.DataFrame([{"timestamp_hour": "2025-01-01 00:00:00"}]),
        predictive_zip_shap=pd.DataFrame(),
        coverage_solutions=pd.DataFrame(),
        summary_shap=pd.DataFrame(),
        prediction_metrics={},
        evaluation_metrics={},
        run_metadata={},
        coalition_values=None,
        cvar_summary_shap=None,
    )

    write_ems_exact_shap_outputs(outputs, outdir)

    assert not (outdir / "cvar_summary_shap.csv").exists()
    assert not (outdir / "coalition_values.csv").exists()
    assert (outdir / "hourly_shap.csv").exists()


def _write_toy_ems_case(tmp_path: Path) -> dict[str, Path]:
    zip_codes = ("10001", "10002", "10003")
    timestamps = pd.date_range("2025-01-30 00:00:00", periods=72, freq="h")
    feature_columns = [
        "hour",
        "month",
        "day_of_week",
        "temp_c",
        "precip_mm",
        "citywide_ems_incidents_lag_1",
    ]
    for family in (
        "ems_incidents_lag_1",
        "neighbor_ems_incidents_lag_1_mean",
        "zone_hour_baseline",
    ):
        feature_columns.extend(f"{family}_zip_{zip_code}" for zip_code in zip_codes)
    target_columns = [
        f"target_ems_incident_count_zip_{zip_code}" for zip_code in zip_codes
    ]

    x_rows: list[dict[str, float | int | pd.Timestamp]] = []
    y_rows: list[dict[str, float | int | pd.Timestamp]] = []
    for hour_idx, timestamp in enumerate(timestamps):
        x_row: dict[str, float | int | pd.Timestamp] = {
            "timestamp_hour": timestamp,
            "hour": int(timestamp.hour),
            "month": int(timestamp.month),
            "day_of_week": int(timestamp.dayofweek),
            "temp_c": 5.0 + hour_idx,
            "precip_mm": float(hour_idx % 2),
            "citywide_ems_incidents_lag_1": 10.0 + hour_idx,
        }
        y_row: dict[str, float | int | pd.Timestamp] = {"timestamp_hour": timestamp}
        for zip_idx, zip_code in enumerate(zip_codes):
            x_row[f"ems_incidents_lag_1_zip_{zip_code}"] = float(hour_idx + zip_idx)
            x_row[f"neighbor_ems_incidents_lag_1_mean_zip_{zip_code}"] = float(
                hour_idx + 0.5 * zip_idx
            )
            x_row[f"zone_hour_baseline_zip_{zip_code}"] = float(1 + zip_idx)
            y_row[f"target_ems_incident_count_zip_{zip_code}"] = float(
                (hour_idx + zip_idx) % 4 + zip_idx
            )
        x_rows.append(x_row)
        y_rows.append(y_row)

    x_path = tmp_path / "ems_X.csv"
    y_path = tmp_path / "ems_y.csv"
    metadata_path = tmp_path / "metadata.json"
    zone_order_path = tmp_path / "zone_order.csv"
    distance_path = tmp_path / "distances.csv"

    pd.DataFrame(x_rows).to_csv(x_path, index=False)
    pd.DataFrame(y_rows).to_csv(y_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "target_columns": target_columns,
                "zip_codes": list(zip_codes),
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "zip_code": list(zip_codes),
            "target_column": target_columns,
            "output_index": [0, 1, 2],
        }
    ).to_csv(zone_order_path, index=False)
    pd.DataFrame(
        {
            "zip_code": list(zip_codes),
            "10001": [0.0, 0.5, 2.0],
            "10002": [0.5, 0.0, 0.8],
            "10003": [2.0, 0.8, 0.0],
        }
    ).to_csv(distance_path, index=False)
    return {
        "x": x_path,
        "y": y_path,
        "metadata": metadata_path,
        "zone_order": zone_order_path,
        "distance": distance_path,
    }
