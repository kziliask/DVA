from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from collections import defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.exceptions import ConvergenceWarning

import dva.analysis.caiso_shap as caiso_shap_module
from dva.analysis.caiso_shap import (
    CaisoShapCaseStudyConfig,
    CaisoShapCaseStudyOutputs,
    ExtendedPlayerCoalitionEvaluator,
    ParameterPlayerSpec,
    build_default_storage_parameters,
    compute_exact_faith_shap_values,
    compute_mobius_transform,
    compute_exact_shapley_taylor_values,
    compute_exact_shapley_values,
    run_caiso_shap_case_study,
    write_caiso_shap_case_study_outputs,
)
from dva.model.storage_dispatch import StorageDispatchEvaluation, StorageDispatchParameters
from dva.model.train import DEFAULT_FEATURE_COLUMNS, ModelTrainingArtifacts, TrainExplainSplit


ATOL = 1e-10
RTOL = 1e-10
FEATURE_COUNT = 8
TARGET_COUNT = 24


def _subset_mask(players: tuple[int, ...] | list[int] | frozenset[int]) -> int:
    mask = 0
    for player in players:
        mask |= 1 << player
    return mask


def _swap_bits(mask: int, i: int, j: int) -> int:
    bit_i = (mask >> i) & 1
    bit_j = (mask >> j) & 1
    if bit_i == bit_j:
        return mask
    return mask ^ ((1 << i) | (1 << j))


def _scalar_coalition_values(
    player_count: int,
    value_fn: Callable[[int], float],
) -> np.ndarray:
    values = np.zeros(1 << player_count, dtype=float)
    for mask in range(1 << player_count):
        values[mask] = float(value_fn(mask))
    return values


def _sum_st_indices(
    indices: Mapping[frozenset[int], np.ndarray | float],
) -> np.ndarray:
    total: np.ndarray | None = None
    for value in indices.values():
        array = np.asarray(value, dtype=float)
        total = array.copy() if total is None else total + array
    if total is None:
        raise AssertionError("Expected at least one interaction index.")
    return total


def _require_gurobi() -> None:
    gp = pytest.importorskip("gurobipy")
    try:
        model = gp.Model()
        model.Params.OutputFlag = 0
        model.dispose()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Gurobi unavailable: {exc}")


class _StubFeatureEvaluator:
    def __init__(self, output_count: int = TARGET_COUNT) -> None:
        self.output_count = output_count

    def evaluate_all_coalitions(
        self,
        observation: pd.Series | np.ndarray | list[float],
    ) -> np.ndarray:
        return np.zeros((1 << FEATURE_COUNT, self.output_count), dtype=float)


class _MockModel:
    def __init__(self, feature_count: int, target_count: int) -> None:
        self.feature_weights = np.arange(1, feature_count + 1, dtype=float)
        self.hour_offsets = np.arange(1, target_count + 1, dtype=float)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim == 1:
            X_array = X_array[np.newaxis, :]
        base = X_array @ self.feature_weights
        return base[:, np.newaxis] + self.hour_offsets[np.newaxis, :]


class _MockDispatchResult:
    def __init__(self, obj: float) -> None:
        self.objective_value = obj
        self.charge = np.zeros(TARGET_COUNT)
        self.discharge = np.zeros(TARGET_COUNT)
        self.mode = np.zeros(TARGET_COUNT, dtype=int)
        self.state_of_charge = np.zeros(TARGET_COUNT + 1)


def _mock_evaluate(
    true_prices: Sequence[float],
    result: _MockDispatchResult,
    params: StorageDispatchParameters,
) -> StorageDispatchEvaluation:
    objective_value = float(
        np.asarray(true_prices, dtype=float).sum() * params.charge_efficiency
    )
    return StorageDispatchEvaluation(
        objective_value=objective_value,
        revenue_value=objective_value,
        throughput_penalty_value=0.0,
    )


def _make_mock_split_and_training_artifacts() -> tuple[TrainExplainSplit, ModelTrainingArtifacts]:
    feature_columns = tuple(f"f{i}" for i in range(FEATURE_COUNT))
    target_columns = tuple(f"hour_{hour:02d}" for hour in range(1, TARGET_COUNT + 1))
    model = _MockModel(FEATURE_COUNT, TARGET_COUNT)

    X_train = pd.DataFrame(
        [
            [0.0, 1.0, 2.0, 3.0, 1.0, 0.5, 2.5, 1.0],
            [1.0, 0.5, 1.5, 2.0, 2.0, 1.5, 1.0, 2.0],
            [2.0, 1.5, 0.0, 1.0, 3.0, 2.5, 0.5, 3.0],
            [3.0, 2.0, 1.0, 0.0, 4.0, 3.5, 1.5, 4.0],
        ],
        columns=feature_columns,
        dtype=float,
    )
    X_explain = pd.DataFrame(
        [
            [0.5, 1.5, 2.5, 3.5, 1.5, 0.75, 2.75, 2.0],
            [1.5, 0.25, 1.75, 2.25, 2.5, 1.25, 1.25, 5.0],
        ],
        columns=feature_columns,
        dtype=float,
    )
    y_train = pd.DataFrame(model.predict(X_train), columns=target_columns, dtype=float)
    y_explain = pd.DataFrame(model.predict(X_explain) + 0.5, columns=target_columns, dtype=float)

    train_dates = pd.Series(
        ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        name="date",
    )
    explain_dates = pd.Series(["2026-01-05", "2026-01-06"], name="date")
    train_frame = pd.concat([train_dates, X_train, y_train], axis=1)
    explain_frame = pd.concat([explain_dates, X_explain, y_explain], axis=1)
    split = TrainExplainSplit(
        dataset_path=Path("mock_dataset.csv"),
        date_column="date",
        feature_columns=feature_columns,
        target_columns=target_columns,
        train_frame=train_frame,
        explain_frame=explain_frame,
        train_dates=train_dates,
        explain_dates=explain_dates,
        X_train=X_train,
        y_train=y_train,
        X_explain=X_explain,
        y_explain=y_explain,
    )
    training_artifacts = ModelTrainingArtifacts(
        model=model,
        model_name="rf",
        model_description="mock_model",
        feature_columns=feature_columns,
        target_columns=target_columns,
        X_train=X_train,
        y_train=y_train,
    )
    return split, training_artifacts


def _run_mock_case_study(
    monkeypatch: pytest.MonkeyPatch,
    *,
    interaction_order: int,
    parameter_player_spec: ParameterPlayerSpec | None,
    interaction_method: str = "shapley_taylor",
    actual_parameters: StorageDispatchParameters | None = None,
    compute_ead_decision_shap: bool = False,
) -> tuple[CaisoShapCaseStudyOutputs, TrainExplainSplit, dict[str, int]]:
    split, training_artifacts = _make_mock_split_and_training_artifacts()
    actual_parameters = (
        actual_parameters
        if actual_parameters is not None
        else StorageDispatchParameters(
            energy_capacity=2.0,
            power_limit=1.0,
            charge_efficiency=0.9,
            discharge_efficiency=0.85,
            throughput_penalty=0.5,
            initial_state_of_charge=1.0,
            terminal_state_of_charge=1.0,
        )
    )
    non_oracle_solve_counts: dict[str, int] = defaultdict(int)

    def _mock_solve(
        predicted: Sequence[float],
        params: StorageDispatchParameters,
        **kwargs,
    ) -> _MockDispatchResult:
        name = str(kwargs.get("name", ""))
        if name.startswith("storage_dispatch_") and not name.startswith("storage_dispatch_oracle_"):
            parts = name.split("_")
            non_oracle_solve_counts[parts[2]] += 1
        return _MockDispatchResult(
            obj=float(np.asarray(predicted, dtype=float).sum() * params.charge_efficiency - params.throughput_penalty)
        )

    monkeypatch.setattr(
        caiso_shap_module,
        "load_default_train_explain_split",
        lambda **kwargs: split,
    )
    monkeypatch.setattr(
        caiso_shap_module,
        "train_model",
        lambda *args, **kwargs: training_artifacts,
    )
    monkeypatch.setattr(
        caiso_shap_module,
        "solve_storage_dispatch_lexicographic",
        _mock_solve,
    )
    monkeypatch.setattr(
        caiso_shap_module,
        "evaluate_storage_dispatch_result",
        _mock_evaluate,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"coalition_values\[0\] is not zero",
        )
        outputs = run_caiso_shap_case_study(
            CaisoShapCaseStudyConfig(
                model_name="rf",
                max_days=2,
                storage_parameters=actual_parameters,
                interaction_order=interaction_order,
                interaction_method=interaction_method,
                parameter_player_spec=parameter_player_spec,
                compute_ead_decision_shap=compute_ead_decision_shap,
            )
        )
    return outputs, split, dict(non_oracle_solve_counts)


def test_mobius_transform_for_two_player_game() -> None:
    values = np.array([0.0, 1.0, 2.0, 5.0])

    mobius = compute_mobius_transform(values, player_count=2)

    assert_allclose(mobius, np.array([0.0, 1.0, 2.0, 2.0]), atol=ATOL, rtol=RTOL)


def test_faith_shap_order_1_matches_shapley() -> None:
    rng = np.random.default_rng(20)
    values = rng.normal(size=(1 << 4, 3))

    shap = compute_exact_shapley_values(values, feature_count=4)
    faith = compute_exact_faith_shap_values(values, player_count=4, order=1)

    assert set(faith) == {frozenset({idx}) for idx in range(4)}
    for feature_idx in range(4):
        assert_allclose(
            faith[frozenset({feature_idx})],
            shap[feature_idx],
            atol=1e-12,
            rtol=1e-12,
        )


def test_faith_shap_full_order_matches_mobius_transform() -> None:
    rng = np.random.default_rng(21)
    player_count = 4
    values = rng.normal(size=1 << player_count)

    faith = compute_exact_faith_shap_values(
        values,
        player_count=player_count,
        order=player_count,
    )
    mobius = compute_mobius_transform(values, player_count=player_count)

    for subset, value in faith.items():
        mask = _subset_mask(subset)
        assert_allclose(value, mobius[mask], atol=1e-12, rtol=1e-12)


def test_faith_shap_efficiency_for_scalar_and_vector_games() -> None:
    rng = np.random.default_rng(22)
    for values in (
        rng.normal(size=1 << 5),
        rng.normal(size=(1 << 5, 4)),
    ):
        faith = compute_exact_faith_shap_values(
            values,
            player_count=5,
            order=2,
        )
        assert_allclose(
            _sum_st_indices(faith),
            values[-1] - values[0],
            atol=1e-10,
            rtol=1e-10,
        )


def test_faith_shap_pure_pairwise_game_recovers_true_interaction() -> None:
    values = _scalar_coalition_values(
        3,
        lambda mask: 1.0 if (mask & 0b011) == 0b011 else 0.0,
    )

    faith = compute_exact_faith_shap_values(values, player_count=3, order=2)

    for subset, value in faith.items():
        expected = 1.0 if subset == frozenset({0, 1}) else 0.0
        assert_allclose(value, expected, atol=ATOL, rtol=RTOL)


def test_shapley_taylor_order_1_matches_shapley() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(1 << 5, 3))
    values[0] = 0.0

    shap = compute_exact_shapley_values(values, feature_count=5)
    st = compute_exact_shapley_taylor_values(values, player_count=5, order=1)

    assert set(st) == {frozenset({idx}) for idx in range(5)}
    for feature_idx in range(5):
        assert_allclose(
            st[frozenset({feature_idx})],
            shap[feature_idx],
            atol=1e-12,
            rtol=1e-12,
        )


def test_shapley_taylor_efficiency_axiom() -> None:
    rng = np.random.default_rng(1)
    for player_count in (3, 5, 7):
        for order in (1, 2, 3):
            if order > player_count:
                continue
            values = rng.normal(size=1 << player_count)
            values[0] = 0.0
            st = compute_exact_shapley_taylor_values(
                values,
                player_count=player_count,
                order=order,
            )
            assert_allclose(
                _sum_st_indices(st),
                values[-1],
                atol=1e-9,
                rtol=0.0,
            )


def test_shapley_taylor_additive_game_has_zero_interactions() -> None:
    rng = np.random.default_rng(2)
    coefficients = rng.normal(size=6)
    values = _scalar_coalition_values(
        6,
        lambda mask: sum(coefficients[idx] for idx in range(6) if mask & (1 << idx)),
    )

    st = compute_exact_shapley_taylor_values(values, player_count=6, order=2)

    for feature_idx in range(6):
        assert_allclose(
            st[frozenset({feature_idx})],
            coefficients[feature_idx],
            atol=1e-12,
            rtol=1e-12,
        )
    for left in range(6):
        for right in range(left + 1, 6):
            assert_allclose(
                st[frozenset({left, right})],
                0.0,
                atol=1e-12,
                rtol=1e-12,
            )
    assert_allclose(coefficients.sum(), values[-1], atol=1e-12, rtol=1e-12)


def test_shapley_taylor_pairwise_interaction_game() -> None:
    values = _scalar_coalition_values(
        4,
        lambda mask: 1.0 if (mask & 0b0011) == 0b0011 else 0.0,
    )
    st = compute_exact_shapley_taylor_values(values, player_count=4, order=2)

    assert_allclose(st[frozenset({0, 1})], 1.0, atol=ATOL, rtol=RTOL)
    for pair in (
        frozenset({0, 2}),
        frozenset({0, 3}),
        frozenset({1, 2}),
        frozenset({1, 3}),
        frozenset({2, 3}),
    ):
        assert_allclose(st[pair], 0.0, atol=ATOL, rtol=RTOL)
    for singleton in range(4):
        assert_allclose(st[frozenset({singleton})], 0.0, atol=ATOL, rtol=RTOL)
    assert_allclose(_sum_st_indices(st), 1.0, atol=ATOL, rtol=RTOL)


def test_shapley_taylor_symmetry_axiom() -> None:
    rng = np.random.default_rng(3)
    player_count = 5
    player_i = 1
    player_j = 3
    values_raw = rng.normal(size=1 << player_count)
    values_raw[0] = 0.0
    values_symmetric = np.zeros_like(values_raw)
    for mask in range(1 << player_count):
        swapped_mask = _swap_bits(mask, player_i, player_j)
        values_symmetric[mask] = 0.5 * (values_raw[mask] + values_raw[swapped_mask])

    st = compute_exact_shapley_taylor_values(
        values_symmetric,
        player_count=player_count,
        order=2,
    )

    assert_allclose(
        st[frozenset({player_i})],
        st[frozenset({player_j})],
        atol=ATOL,
        rtol=RTOL,
    )
    for other_player in (0, 2, 4):
        assert_allclose(
            st[frozenset({player_i, other_player})],
            st[frozenset({player_j, other_player})],
            atol=ATOL,
            rtol=RTOL,
        )


def test_shapley_taylor_dummy_player() -> None:
    rng = np.random.default_rng(4)
    base_values = rng.normal(size=1 << 4)
    base_values[0] = 0.0
    keep_players = (0, 1, 2, 4)

    def _reduced_mask(mask: int) -> int:
        reduced = 0
        for reduced_idx, player_idx in enumerate(keep_players):
            if mask & (1 << player_idx):
                reduced |= 1 << reduced_idx
        return reduced

    values = np.zeros(1 << 5, dtype=float)
    for mask in range(1 << 5):
        values[mask] = base_values[_reduced_mask(mask)]

    st_full = compute_exact_shapley_taylor_values(values, player_count=5, order=2)
    st_restricted = compute_exact_shapley_taylor_values(base_values, player_count=4, order=2)

    assert_allclose(st_full[frozenset({3})], 0.0, atol=ATOL, rtol=RTOL)
    for other_player in (0, 1, 2, 4):
        assert_allclose(
            st_full[frozenset({3, other_player})],
            0.0,
            atol=ATOL,
            rtol=RTOL,
        )

    mapping = {0: 0, 1: 1, 2: 2, 4: 3}
    for subset, value in st_full.items():
        if 3 in subset:
            continue
        restricted_subset = frozenset(mapping[player] for player in subset)
        assert_allclose(
            value,
            st_restricted[restricted_subset],
            atol=ATOL,
            rtol=RTOL,
        )


def test_shapley_taylor_input_validation() -> None:
    with pytest.raises(ValueError, match="one entry per coalition"):
        compute_exact_shapley_taylor_values(np.zeros(3), player_count=2, order=1)
    with pytest.raises(ValueError, match="order"):
        compute_exact_shapley_taylor_values(np.zeros(4), player_count=2, order=0)
    with pytest.raises(ValueError, match="order"):
        compute_exact_shapley_taylor_values(np.zeros(4), player_count=2, order=3)
    with pytest.raises(ValueError, match="<= 20"):
        compute_exact_shapley_taylor_values(np.zeros(1), player_count=21, order=1)
    with pytest.raises(ValueError, match="at least 1"):
        compute_exact_shapley_taylor_values(np.zeros(1), player_count=0, order=1)


def test_shapley_taylor_quadratic_game_closed_form() -> None:
    values = _scalar_coalition_values(
        4,
        lambda mask: float(mask.bit_count() ** 2),
    )
    st = compute_exact_shapley_taylor_values(values, player_count=4, order=2)

    for feature_idx in range(4):
        assert_allclose(st[frozenset({feature_idx})], 1.0, atol=ATOL, rtol=RTOL)
    for left in range(4):
        for right in range(left + 1, 4):
            assert_allclose(st[frozenset({left, right})], 2.0, atol=ATOL, rtol=RTOL)
    assert_allclose(_sum_st_indices(st), 16.0, atol=ATOL, rtol=RTOL)


def test_parameter_player_spec_player_count_and_names() -> None:
    spec = ParameterPlayerSpec()
    assert spec.player_count == 0
    assert spec.player_names() == ()

    throughput_only = ParameterPlayerSpec(throughput_penalty_is_player=True)
    assert throughput_only.player_count == 1
    assert throughput_only.player_names() == ("throughput_penalty",)

    efficiency_capacity = ParameterPlayerSpec(
        efficiency_is_player=True,
        energy_capacity_is_player=True,
    )
    assert efficiency_capacity.player_count == 2
    assert efficiency_capacity.player_names() == ("efficiency", "energy_capacity")

    all_players = ParameterPlayerSpec(
        throughput_penalty_is_player=True,
        efficiency_is_player=True,
        energy_capacity_is_player=True,
    )
    assert all_players.player_count == 3
    assert all_players.player_names() == (
        "throughput_penalty",
        "efficiency",
        "energy_capacity",
    )


def test_extended_evaluator_parameters_for_param_mask() -> None:
    actual = StorageDispatchParameters(
        energy_capacity=2.0,
        power_limit=1.25,
        charge_efficiency=0.9,
        discharge_efficiency=0.85,
        throughput_penalty=0.5,
        initial_state_of_charge=0.75,
        terminal_state_of_charge=0.5,
    )
    spec = ParameterPlayerSpec(
        throughput_penalty_is_player=True,
        efficiency_is_player=True,
        energy_capacity_is_player=True,
    )
    evaluator = ExtendedPlayerCoalitionEvaluator(
        feature_evaluator=_StubFeatureEvaluator(),
        feature_names=tuple(f"f{i}" for i in range(FEATURE_COUNT)),
        actual_parameters=actual,
        parameter_player_spec=spec,
    )

    assert evaluator.parameters_for_param_mask(0b000) == replace(
        actual,
        throughput_penalty=0.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        energy_capacity=1e6,
    )
    assert evaluator.parameters_for_param_mask(0b111) == actual
    assert evaluator.parameters_for_param_mask(0b010) == replace(
        actual,
        throughput_penalty=0.0,
        energy_capacity=1e6,
    )

    for throughput_on in (False, True):
        for efficiency_on in (False, True):
            for capacity_on in (False, True):
                spec_variant = ParameterPlayerSpec(
                    throughput_penalty_is_player=throughput_on,
                    efficiency_is_player=efficiency_on,
                    energy_capacity_is_player=capacity_on,
                )
                evaluator_variant = ExtendedPlayerCoalitionEvaluator(
                    feature_evaluator=_StubFeatureEvaluator(),
                    feature_names=("f0", "f1"),
                    actual_parameters=actual,
                    parameter_player_spec=spec_variant,
                )
                for param_mask in range(1 << spec_variant.player_count):
                    expected = actual
                    bit = 0
                    if throughput_on:
                        if not (param_mask & (1 << bit)):
                            expected = replace(expected, throughput_penalty=0.0)
                        bit += 1
                    if efficiency_on:
                        if not (param_mask & (1 << bit)):
                            expected = replace(
                                expected,
                                charge_efficiency=1.0,
                                discharge_efficiency=1.0,
                            )
                        bit += 1
                    if capacity_on and not (param_mask & (1 << bit)):
                        expected = replace(expected, energy_capacity=1e6)
                    assert evaluator_variant.parameters_for_param_mask(param_mask) == expected


def test_extended_evaluator_bit_layout() -> None:
    feature_count = 3
    spec = ParameterPlayerSpec(
        throughput_penalty_is_player=True,
        efficiency_is_player=True,
    )
    evaluator = ExtendedPlayerCoalitionEvaluator(
        feature_evaluator=_StubFeatureEvaluator(),
        feature_names=tuple(f"f{i}" for i in range(feature_count)),
        actual_parameters=build_default_storage_parameters(),
        parameter_player_spec=spec,
    )
    mask = 0b10110
    feat_mask = mask & ((1 << evaluator.feature_count) - 1)
    param_mask = mask >> evaluator.feature_count

    assert feat_mask == 0b110
    assert param_mask == 0b10


def test_features_only_backward_compat_regression(tmp_path: Path) -> None:
    _require_gurobi()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        outputs = run_caiso_shap_case_study(
            CaisoShapCaseStudyConfig(model_name="mlp", max_days=2)
        )
    write_caiso_shap_case_study_outputs(outputs, tmp_path)

    daily_shap = pd.read_csv(tmp_path / "daily_shap.csv")
    summary_shap = pd.read_csv(tmp_path / "summary_shap.csv")
    with (tmp_path / "run_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    assert metadata["coalition_expectation_method"] == "empirical_background_marginalization"
    assert metadata["parameter_player_spec"] is None
    assert metadata["interaction_order"] is None
    assert metadata["player_names"] == list(DEFAULT_FEATURE_COLUMNS)
    assert set(summary_shap["feature"]) == set(DEFAULT_FEATURE_COLUMNS)
    assert len(daily_shap) == 2
    for _, row in daily_shap.iterrows():
        decision_shap_sum = sum(
            row[f"decision_shap_{feature_name}"]
            for feature_name in DEFAULT_FEATURE_COLUMNS
        )
        predictive_shap_sum = sum(
            row[f"predictive_shap_{feature_name}"]
            for feature_name in DEFAULT_FEATURE_COLUMNS
        )
        assert decision_shap_sum == pytest.approx(row["decision_value_gain"])
        assert predictive_shap_sum == pytest.approx(row["predictive_total_gain"])


def test_features_only_with_interaction_order_2(tmp_path: Path) -> None:
    _require_gurobi()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings(
            "ignore",
            message=r"coalition_values\[0\] is not zero",
        )
        outputs_order_2 = run_caiso_shap_case_study(
            CaisoShapCaseStudyConfig(
                model_name="mlp",
                max_days=2,
                interaction_order=2,
                interaction_method="shapley_taylor",
            )
        )
        outputs_order_1 = run_caiso_shap_case_study(
            CaisoShapCaseStudyConfig(
                model_name="mlp",
                max_days=2,
                interaction_order=1,
                interaction_method="shapley_taylor",
            )
        )

    write_caiso_shap_case_study_outputs(outputs_order_2, tmp_path)
    decision_st = pd.read_csv(tmp_path / "daily_shapley_taylor_decision.csv")
    decision_interaction = pd.read_csv(tmp_path / "daily_interaction_decision.csv")

    assert list(decision_st.columns) == [
        "date",
        "order",
        "subset_size",
        "players",
        "decision_shapley_taylor",
    ]
    assert set(decision_interaction["interaction_method"]) == {"shapley_taylor"}
    assert_allclose(
        decision_interaction["decision_interaction_value"].to_numpy(dtype=float),
        decision_st["decision_shapley_taylor"].to_numpy(dtype=float),
        atol=0.0,
        rtol=0.0,
    )

    assert len(decision_st) == 2 * (FEATURE_COUNT + math.comb(FEATURE_COUNT, 2))
    st_totals = decision_st.groupby(["date", "order"], sort=False)["decision_shapley_taylor"].sum()
    daily_gains = outputs_order_2.daily_shap.set_index("date")["decision_value_gain"]
    for (date, order), total in st_totals.items():
        assert order == 2
        assert_allclose(total, daily_gains[date], atol=1e-8, rtol=0.0)

    decision_st_order_1 = outputs_order_1.daily_shapley_taylor_decision
    assert decision_st_order_1 is not None
    for _, daily_row in outputs_order_1.daily_shap.iterrows():
        singleton_rows = decision_st_order_1[
            (decision_st_order_1["date"] == daily_row["date"])
            & (decision_st_order_1["subset_size"] == 1)
        ]
        singleton_map = dict(
            zip(
                singleton_rows["players"],
                singleton_rows["decision_shapley_taylor"],
                strict=True,
            )
        )
        for feature_name in outputs_order_1.run_metadata["feature_columns"]:
            assert_allclose(
                singleton_map[feature_name],
                daily_row[f"decision_shap_{feature_name}"],
                atol=1e-12,
                rtol=1e-12,
            )


def test_ead_decision_shap_outputs_are_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs, _, _ = _run_mock_case_study(
        monkeypatch,
        interaction_order=1,
        parameter_player_spec=None,
        compute_ead_decision_shap=True,
    )

    daily = outputs.daily_shap
    assert outputs.run_metadata["compute_ead_decision_shap"] is True
    assert "ead_decision" in outputs.evaluation_metrics
    assert "ead_decision_mean_abs_shap" in outputs.summary_shap.columns
    assert "daily_predictive_ead_abs_rank_spearman" in outputs.comparison_metrics
    for _, row in daily.iterrows():
        ead_shap_sum = sum(
            row[f"ead_decision_shap_{feature_name}"]
            for feature_name in outputs.run_metadata["feature_columns"]
        )
        assert_allclose(
            ead_shap_sum,
            row["ead_decision_value_gain"],
            atol=1e-10,
            rtol=0.0,
        )
        assert_allclose(
            row["ead_decision_value_gain"],
            row["ead_decision_full_value"] - row["ead_decision_baseline_value"],
            atol=1e-10,
            rtol=0.0,
        )
        assert_allclose(
            row["ead_decision_characteristic_baseline_value"],
            0.0,
            atol=0.0,
            rtol=0.0,
        )
        assert_allclose(
            row["ead_decision_characteristic_full_value"],
            row["ead_decision_value_gain"],
            atol=0.0,
            rtol=0.0,
        )


def test_extended_players_efficiency(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = ParameterPlayerSpec(
        throughput_penalty_is_player=True,
        efficiency_is_player=True,
        energy_capacity_is_player=True,
    )
    outputs, split, solve_counts = _run_mock_case_study(
        monkeypatch,
        interaction_order=2,
        parameter_player_spec=spec,
    )

    assert outputs.daily_shapley_taylor_decision is not None
    assert len(outputs.daily_shapley_taylor_decision) == len(split.explain_dates) * (11 + 55)
    totals = outputs.daily_shapley_taylor_decision.groupby("date", sort=False)["decision_shapley_taylor"].sum()
    decision_gains = outputs.daily_shap.set_index("date")["decision_value_gain"]
    for date, total in totals.items():
        assert_allclose(total, decision_gains[date], atol=1e-8, rtol=0.0)
    assert outputs.run_metadata["coalitions_per_day"] == 2 ** 11
    assert solve_counts == {date: 2 ** 11 for date in split.explain_dates.tolist()}


def test_parameter_baseline_equals_actual_is_dummy(monkeypatch: pytest.MonkeyPatch) -> None:
    actual_parameters = StorageDispatchParameters(
        energy_capacity=2.0,
        power_limit=1.0,
        charge_efficiency=0.9,
        discharge_efficiency=0.85,
        throughput_penalty=0.5,
        initial_state_of_charge=1.0,
        terminal_state_of_charge=1.0,
    )
    spec = ParameterPlayerSpec(
        throughput_penalty_is_player=True,
        throughput_penalty_baseline=actual_parameters.throughput_penalty,
        efficiency_is_player=True,
        energy_capacity_is_player=True,
    )
    outputs, _, _ = _run_mock_case_study(
        monkeypatch,
        interaction_order=2,
        parameter_player_spec=spec,
        actual_parameters=actual_parameters,
    )

    decision_st = outputs.daily_shapley_taylor_decision
    assert decision_st is not None
    throughput_rows = decision_st[decision_st["players"].str.contains("throughput_penalty")]
    assert np.allclose(
        throughput_rows["decision_shapley_taylor"].to_numpy(dtype=float),
        0.0,
        atol=1e-8,
        rtol=0.0,
    )


def test_prediction_shapley_taylor_dummy_for_params(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = ParameterPlayerSpec(
        throughput_penalty_is_player=True,
        efficiency_is_player=True,
        energy_capacity_is_player=True,
    )
    outputs, _, _ = _run_mock_case_study(
        monkeypatch,
        interaction_order=2,
        parameter_player_spec=spec,
    )

    predictive_st = outputs.daily_shapley_taylor_predictive
    assert predictive_st is not None
    param_names = set(spec.player_names())
    param_rows = predictive_st[
        predictive_st["players"].map(
            lambda player_subset: any(
                name in param_names for name in str(player_subset).split("|")
            )
        )
    ]
    assert np.allclose(
        param_rows["predictive_shapley_taylor"].to_numpy(dtype=float),
        0.0,
        atol=1e-12,
        rtol=0.0,
    )


def test_faith_shap_case_study_outputs_use_generic_interaction_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = ParameterPlayerSpec(
        throughput_penalty_is_player=True,
        efficiency_is_player=True,
        energy_capacity_is_player=True,
    )
    outputs, split, solve_counts = _run_mock_case_study(
        monkeypatch,
        interaction_order=2,
        interaction_method="faith_shap",
        parameter_player_spec=spec,
    )

    assert outputs.daily_interaction_decision is not None
    assert outputs.daily_interaction_predictive is not None
    assert outputs.daily_shapley_taylor_decision is None
    assert outputs.daily_shapley_taylor_predictive is None
    assert set(outputs.daily_interaction_decision["interaction_method"]) == {"faith_shap"}
    assert set(outputs.daily_interaction_predictive["interaction_method"]) == {"faith_shap"}
    assert outputs.run_metadata["interaction_method"] == "faith_shap"
    assert outputs.run_metadata["interaction_order"] == 2
    assert outputs.run_metadata["shapley_taylor_efficiency_gap_decision"] is None
    assert outputs.run_metadata["interaction_efficiency_gap_decision"] < 1e-8
    assert outputs.run_metadata["interaction_efficiency_gap_predictive"] < 1e-8
    assert solve_counts == {date: 2 ** 11 for date in split.explain_dates.tolist()}

    write_caiso_shap_case_study_outputs(outputs, tmp_path)

    assert (tmp_path / "daily_interaction_decision.csv").exists()
    assert (tmp_path / "daily_interaction_predictive.csv").exists()
    assert (tmp_path / "daily_faith_shap_decision.csv").exists()
    assert (tmp_path / "daily_faith_shap_predictive.csv").exists()
    assert not (tmp_path / "daily_shapley_taylor_decision.csv").exists()
    assert not (tmp_path / "daily_shapley_taylor_predictive.csv").exists()


def test_shapley_taylor_large_n() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(size=1 << 15)
    values[0] = 0.0

    started_at = time.perf_counter()
    st = compute_exact_shapley_taylor_values(values, player_count=15, order=2)
    elapsed = time.perf_counter() - started_at

    assert elapsed < 10.0
    assert_allclose(_sum_st_indices(st), values[-1], atol=1e-8, rtol=0.0)
