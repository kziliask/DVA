from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from dva.analysis.caiso_shap import (
    CaisoShapCaseStudyConfig,
    StorageDispatchParameters,
    _build_coalition_evaluator,
    _solve_all_coalitions_for_day,
    compute_exact_shapley_values,
    select_recent_background_frame,
)
from dva.analysis.ems_exact_shap import (
    EMS_TIMESTAMP_COLUMN,
    EmsExactShapConfig,
    _build_solver_params,
    _build_time_split,
    _fit_xgb_regressor,
    _frame_to_feature_matrix,
    _load_distance_matrix,
    _load_ems_frames,
    _load_zone_order,
    _maybe_sample_training_rows,
    _realized_coverage_value,
    _resolve_feature_columns,
    _sample_explanation_hours,
    _target_columns,
    build_coverage_matrix,
    build_ems_feature_groups,
    solve_ems_coverage,
)
from dva.analysis.evaluation_metrics import (
    build_attribution_ranking,
    compute_decision_deletion_auc,
    compute_exact_decision_infidelity,
    compute_decision_insertion_auc,
    compute_rank_kendall_tau_from_rankings,
    compute_rank_spearman_from_rankings,
)
from dva.analysis.paired_bootstrap import BootstrapConfig, bootstrap_metric_table
from dva.analysis.run_caiso_decision_shap_guided_validation import (
    build_fixed_caiso_guided_validation_split,
)
from dva.model.storage_dispatch import (
    evaluate_storage_dispatch_result,
    solve_storage_dispatch_lexicographic,
)
from dva.model.train import load_default_train_explain_split, train_model


BaselineMethod = Literal[
    "Post-InfoDVA",
    "Prediction SHAP",
    "Leave-one-feature-out",
    "Downstream permutation feature importance",
    "Greedy decision insertion",
]
RankingMode = Literal["method_default", "opposite", "local", "global", "paper_original"]
CaisoSource = Literal["legacy_holdout", "gdsi"]

DEFAULT_OUTPUT = Path("results/bootstrap_ci/additional_decision_ranking_baselines.csv")
DEFAULT_TABLE_OUTPUT = Path(
    "results/bootstrap_ci/combined_attribution_metrics_bootstrap_amended.tex"
)
DEFAULT_CAISO_LEGACY_DAILY_SHAP_OUTPUT = Path(
    "results/bootstrap_ci/caiso_legacy_recomputed_daily_shap.csv"
)
DEFAULT_EMS_RUN_DIR = Path(
    "results/ems/experiment_c_solver_dva/xgb_023/"
    "exact_vs_naive_post/design_coalitions/mask_0_exact_tau1_p8"
)


@dataclass(frozen=True, slots=True)
class AdditionalBaselineResult:
    local_rows: pd.DataFrame
    aggregate_rows: pd.DataFrame
    bootstrap_rows: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ReferenceMetricSource:
    dataset: str
    path: Path
    unit_column: str
    unit_prefix: str | None = None


def compute_additional_baselines_from_coalitions(
    *,
    dataset: str,
    unit_ids: Sequence[str],
    player_names: Sequence[str],
    decision_characteristic_by_unit: Mapping[str, Sequence[float] | np.ndarray],
    reference_scores_by_unit: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
    pfi_scores: Mapping[str, float] | None = None,
    pfi_local_scores_by_unit: Mapping[str, Mapping[str, float]] | None = None,
    ranking_mode: RankingMode = "method_default",
    bootstrap_config: BootstrapConfig | None = None,
) -> AdditionalBaselineResult:
    """Compute reference and baseline insertion AUCs from shared coalition values."""

    bootstrap_config = bootstrap_config or BootstrapConfig()
    players = tuple(str(player) for player in player_names)
    unit_list = tuple(str(unit_id) for unit_id in unit_ids)
    values_by_unit = {
        unit_id: _validate_coalition_vector(
            decision_characteristic_by_unit[unit_id],
            player_count=len(players),
            unit_id=unit_id,
        )
        for unit_id in unit_list
    }

    greedy_scores = _greedy_insertion_scores(values_by_unit, players)
    reference_scores = dict(reference_scores_by_unit or {})
    lofo_scores_by_unit = {
        unit_id: _lofo_scores(values_by_unit[unit_id], players)
        for unit_id in unit_list
    }
    local_greedy_scores_by_unit = {
        unit_id: _greedy_insertion_scores({unit_id: values_by_unit[unit_id]}, players)
        for unit_id in unit_list
    }
    local_scores_by_method: dict[str, Mapping[str, Mapping[str, float]]] = {
        **reference_scores,
        "Leave-one-feature-out": lofo_scores_by_unit,
        "Greedy decision insertion": local_greedy_scores_by_unit,
    }
    if pfi_local_scores_by_unit is not None:
        local_scores_by_method["Downstream permutation feature importance"] = (
            pfi_local_scores_by_unit
        )

    global_scores_by_method: dict[str, Mapping[str, float]] = {
        method: _mean_abs_local_scores(scores_by_unit, players)
        for method, scores_by_unit in reference_scores.items()
    }
    global_scores_by_method["Leave-one-feature-out"] = _mean_abs_local_scores(
        lofo_scores_by_unit,
        players,
    )
    global_scores_by_method["Greedy decision insertion"] = greedy_scores
    if pfi_local_scores_by_unit is not None:
        global_scores_by_method["Downstream permutation feature importance"] = (
            _mean_abs_local_scores(pfi_local_scores_by_unit, players)
        )
    elif pfi_scores is not None:
        global_scores_by_method["Downstream permutation feature importance"] = {
            str(player): abs(float(pfi_scores[player]))
            for player in players
        }
    if ranking_mode == "paper_original":
        if pfi_local_scores_by_unit is not None:
            signed_pfi_scores = _mean_signed_local_scores(
                pfi_local_scores_by_unit,
                players,
            )
            global_scores_by_method["Downstream permutation feature importance"] = (
                _scores_from_order(
                    _signed_score_order(signed_pfi_scores, players),
                    players,
                )
            )

    methods = [
        *reference_scores.keys(),
        "Leave-one-feature-out",
        "Greedy decision insertion",
    ]
    if (
        "Downstream permutation feature importance" in local_scores_by_method
        or "Downstream permutation feature importance" in global_scores_by_method
    ):
        methods.append("Downstream permutation feature importance")

    local_records: list[dict[str, Any]] = []
    aggregate_score_records: list[dict[str, Any]] = []

    for unit_id in unit_list:
        values = values_by_unit[unit_id]
        for method in methods:
            use_local = _ranking_mode_uses_local(method, ranking_mode)
            if use_local and method not in local_scores_by_method:
                use_local = False
            score_scope = "local" if use_local else "global"
            if (
                ranking_mode == "paper_original"
                and method == "Downstream permutation feature importance"
                and not use_local
            ):
                score_scope = "global_signed"
            scores = (
                local_scores_by_method[method][unit_id]
                if use_local
                else global_scores_by_method[method]
            )
            local_records.append(
                _local_record(
                    dataset=dataset,
                    unit_id=unit_id,
                    method=cast(BaselineMethod, method),
                    player_names=players,
                    scores=scores,
                    decision_characteristic_values=values,
                    score_scope=score_scope,
                    ranking_mode=ranking_mode,
                    local_scores=(
                        None
                        if method != "Downstream permutation feature importance"
                        or pfi_local_scores_by_unit is None
                        else pfi_local_scores_by_unit.get(unit_id)
                    ),
                )
            )

    for method, scores in global_scores_by_method.items():
        score_scope = "global"
        if (
            ranking_mode == "paper_original"
            and method == "Downstream permutation feature importance"
        ):
            score_scope = "global_signed"
        aggregate_score_records.append(
            _aggregate_score_record(
                dataset=dataset,
                method=cast(BaselineMethod, method),
                player_names=players,
                scores=scores,
                score_scope=score_scope,
                ranking_mode=ranking_mode,
            )
        )

    local = pd.DataFrame(local_records)
    bootstrap_input = local.loc[
        :,
        ["dataset", "unit_id", "method", "insertion_auc"],
    ].rename(columns={"insertion_auc": "value"})
    bootstrap_input["metric"] = "Decision insertion AUC \\uparrow"
    bootstrap = bootstrap_metric_table(
        bootstrap_input,
        reference_method=None,
        config=bootstrap_config,
    )
    aggregate = _merge_metric_summary(
        pd.DataFrame(aggregate_score_records),
        bootstrap,
    )
    return AdditionalBaselineResult(
        local_rows=local,
        aggregate_rows=aggregate,
        bootstrap_rows=bootstrap,
    )


def load_coalition_values(
    path: Path,
    *,
    unit_column: str,
    value_column: str = "decision_characteristic_value",
    last_n_units: int | None = None,
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    frame = pd.read_csv(path)
    _require_columns(frame, (unit_column, "coalition_mask", value_column))
    ordered_units = tuple(dict.fromkeys(frame[unit_column].astype(str)))
    if last_n_units is not None:
        ordered_units = ordered_units[-int(last_n_units) :]
    values_by_unit: dict[str, np.ndarray] = {}
    for unit_id in ordered_units:
        group = frame.loc[frame[unit_column].astype(str) == unit_id]
        group = group.sort_values("coalition_mask")
        values_by_unit[unit_id] = group[value_column].to_numpy(dtype=float, copy=True)
    return ordered_units, values_by_unit


def _legacy_caiso_holdout_config() -> CaisoShapCaseStudyConfig:
    """Original single-model CAISO recipe used for the paper attribution table."""

    return CaisoShapCaseStudyConfig(
        dataset_path=Path(
            "data/cleaned/caiso_sp15_daily_lmp_weather_2023-01-26_2026-05-07.csv"
        ),
        holdout_days=101,
        outdir=Path("/private/tmp/caiso_additional_decision_ranking_baselines"),
        model_name="xgb",
        random_state=0,
        n_jobs=1,
        mlp_hidden_layer_sizes=(256,),
        mlp_max_iter=1000,
        xgb_n_estimators=100,
        xgb_max_depth=3,
        xgb_learning_rate=0.05,
        xgb_subsample=0.9,
        xgb_colsample_bytree=0.9,
        xgb_reg_lambda=1.0,
        xgb_verbosity=0,
        learning_rate=None,
        mse_learning_rate=None,
        spo_learning_rate=None,
        training_verbose=False,
        training_log_every=None,
        spo_processes=None,
        spo_warm_start_with_mse=False,
        solver_seed=0,
        mip_gap=0.0,
        mip_gap_abs=1e-9,
        objective_tolerance=1e-6,
        max_days=None,
        background_days=365,
        compute_ead_decision_shap=True,
        storage_parameters=StorageDispatchParameters(
            energy_capacity=2.0,
            power_limit=1.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            throughput_penalty=5.0,
            initial_state_of_charge=1.0,
            terminal_state_of_charge=1.0,
        ),
        interaction_order=2,
        interaction_method="faith_shap",
        parameter_player_spec=None,
    )


def _decision_evaluation_metrics(
    attribution_values: Sequence[float] | np.ndarray,
    decision_characteristic_values: Sequence[float] | np.ndarray,
    player_names: Sequence[str],
) -> dict[str, float | None]:
    return {
        "decision_deletion_auc": compute_decision_deletion_auc(
            attribution_values,
            decision_characteristic_values,
            player_names,
        ),
        "decision_insertion_auc": compute_decision_insertion_auc(
            attribution_values,
            decision_characteristic_values,
            player_names,
        ),
        "decision_infidelity": compute_exact_decision_infidelity(
            attribution_values,
            decision_characteristic_values,
            player_names,
        ),
    }


def run_caiso_additional_baselines(
    *,
    run_dir: Path,
    last_n_units: int,
    permutation_seed: int,
    ranking_mode: RankingMode,
    bootstrap_config: BootstrapConfig,
) -> AdditionalBaselineResult:
    model_dir = _resolve_caiso_model_dir(run_dir)
    model_config = _load_json(model_dir / "model_config.json")
    experiment_metadata = _find_upward_json(model_dir, "experiment_metadata.json")
    config_payload = cast(dict[str, Any], model_config["config"])
    config = _caiso_config_from_payload(config_payload)
    split = build_fixed_caiso_guided_validation_split(
        config.dataset_path,
        validation_days=int((experiment_metadata or {}).get("validation_days", 71)),
        test_days=int((experiment_metadata or {}).get("test_days", 30)),
        background_days=int(config.background_days),
        validation_max_days=(experiment_metadata or {}).get("validation_max_days"),
        test_max_days=(experiment_metadata or {}).get("test_max_days"),
        train_months=(experiment_metadata or {}).get("train_months"),
        validation_months=(experiment_metadata or {}).get("validation_months"),
        test_rest=bool((experiment_metadata or {}).get("test_rest", False)),
    )
    explain_frame = split.test_frame.tail(int(last_n_units)).reset_index(drop=True)
    if explain_frame.empty:
        raise ValueError("No CAISO test rows remain for additional baselines.")

    artifacts = train_model(
        split.train_frame.loc[:, list(split.feature_columns)],
        split.train_frame.loc[:, list(split.target_columns)],
        model_name=config.model_name,
        feature_columns=split.feature_columns,
        target_columns=split.target_columns,
        random_state=config.random_state,
        n_jobs=1,
        mlp_hidden_layer_sizes=config.mlp_hidden_layer_sizes,
        mlp_max_iter=config.mlp_max_iter,
        mlp_dropout=config.mlp_dropout,
        mlp_weight_decay=config.mlp_weight_decay,
        mlp_batch_size=config.mlp_batch_size,
        mlp_early_stopping_patience=config.mlp_early_stopping_patience,
        mlp_activation=config.mlp_activation,
        mlp_batch_norm=config.mlp_batch_norm,
        xgb_n_estimators=config.xgb_n_estimators,
        xgb_max_depth=config.xgb_max_depth,
        xgb_learning_rate=config.xgb_learning_rate,
        xgb_subsample=config.xgb_subsample,
        xgb_colsample_bytree=config.xgb_colsample_bytree,
        xgb_reg_lambda=config.xgb_reg_lambda,
        xgb_verbosity=config.xgb_verbosity,
    )
    coalition_evaluator, _ = _build_coalition_evaluator(
        artifacts,
        config,
        split.background_frame.loc[:, list(split.feature_columns)],
    )
    solver_params = {
        "Threads": 1,
        "Seed": config.solver_seed,
        "MIPGap": config.mip_gap,
        "MIPGapAbs": config.mip_gap_abs,
    }
    values_by_unit: dict[str, np.ndarray] = {}
    full_values: dict[str, float] = {}
    reference_scores_by_unit: dict[str, dict[str, dict[str, float]]] = {
        "Post-InfoDVA": {},
        "Prediction SHAP": {},
    }
    unit_ids: list[str] = []
    for row_number, (_, row) in enumerate(explain_frame.iterrows(), start=1):
        unit_id = str(row[split.date_column])
        observation = row.loc[list(split.feature_columns)]
        true_prices = tuple(float(value) for value in row.loc[list(split.target_columns)])
        coalition_predictions, coalition_realized_values, _, _ = _solve_all_coalitions_for_day(
            observation=observation,
            true_prices=true_prices,
            date=unit_id,
            evaluator=coalition_evaluator,
            actual_parameters=config.storage_parameters,
            parameter_player_spec=config.parameter_player_spec,
            solver_params=solver_params,
            objective_tolerance=config.objective_tolerance,
        )
        decision_characteristic_values = coalition_realized_values - coalition_realized_values[0]
        predictive_hourly_shap = compute_exact_shapley_values(
            coalition_predictions,
            feature_count=len(split.feature_columns),
        )
        predictive_daily_shap = predictive_hourly_shap.sum(axis=1)
        decision_shap = compute_exact_shapley_values(
            decision_characteristic_values,
            feature_count=len(split.feature_columns),
        )
        reference_scores_by_unit["Post-InfoDVA"][unit_id] = _score_mapping(
            decision_shap,
            split.feature_columns,
        )
        reference_scores_by_unit["Prediction SHAP"][unit_id] = _score_mapping(
            predictive_daily_shap,
            split.feature_columns,
        )
        values_by_unit[unit_id] = decision_characteristic_values
        full_values[unit_id] = float(coalition_realized_values[-1])
        unit_ids.append(unit_id)
        print(
            f"[CAISO {row_number}/{len(explain_frame)}] computed coalitions for {unit_id}",
            flush=True,
        )

    pfi_scores, pfi_local = _compute_caiso_pfi_scores(
        config=config,
        model=artifacts.model,
        explain_frame=explain_frame,
        feature_columns=split.feature_columns,
        target_columns=split.target_columns,
        date_column=split.date_column,
        full_values=full_values,
        permutation_seed=permutation_seed,
        solver_params=solver_params,
    )
    return compute_additional_baselines_from_coalitions(
        dataset="CAISO",
        unit_ids=unit_ids,
        player_names=split.feature_columns,
        decision_characteristic_by_unit=values_by_unit,
        reference_scores_by_unit=reference_scores_by_unit,
        pfi_scores=pfi_scores,
        pfi_local_scores_by_unit=pfi_local,
        ranking_mode=ranking_mode,
        bootstrap_config=bootstrap_config,
    )


def run_caiso_legacy_holdout_additional_baselines(
    *,
    last_n_units: int,
    permutation_seed: int,
    ranking_mode: RankingMode,
    bootstrap_config: BootstrapConfig,
    daily_shap_output: Path | None = None,
) -> AdditionalBaselineResult:
    config = _legacy_caiso_holdout_config()
    split = load_default_train_explain_split(
        config.dataset_path,
        holdout_days=config.holdout_days,
    )
    artifacts = train_model(
        split.X_train,
        split.y_train,
        model_name=config.model_name,
        feature_columns=split.feature_columns,
        target_columns=split.target_columns,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        mlp_hidden_layer_sizes=config.mlp_hidden_layer_sizes,
        mlp_max_iter=config.mlp_max_iter,
        mlp_dropout=config.mlp_dropout,
        mlp_weight_decay=config.mlp_weight_decay,
        mlp_batch_size=config.mlp_batch_size,
        mlp_early_stopping_patience=config.mlp_early_stopping_patience,
        mlp_activation=config.mlp_activation,
        mlp_batch_norm=config.mlp_batch_norm,
        learning_rate=config.learning_rate,
        mse_learning_rate=config.mse_learning_rate,
        spo_learning_rate=config.spo_learning_rate,
        storage_parameters=config.storage_parameters,
        training_verbose=config.training_verbose,
        training_log_every=config.training_log_every,
        spo_processes=config.spo_processes,
        spo_warm_start_with_mse=config.spo_warm_start_with_mse,
        xgb_n_estimators=config.xgb_n_estimators,
        xgb_max_depth=config.xgb_max_depth,
        xgb_learning_rate=config.xgb_learning_rate,
        xgb_subsample=config.xgb_subsample,
        xgb_colsample_bytree=config.xgb_colsample_bytree,
        xgb_reg_lambda=config.xgb_reg_lambda,
        xgb_verbosity=config.xgb_verbosity,
    )
    background_frame = select_recent_background_frame(
        split.train_frame,
        split.date_column,
        config.background_days,
    )
    coalition_evaluator, _ = _build_coalition_evaluator(
        artifacts,
        config,
        background_frame.loc[:, list(split.feature_columns)],
    )
    solver_params = {
        "Threads": 1,
        "Seed": config.solver_seed,
        "MIPGap": config.mip_gap,
        "MIPGapAbs": config.mip_gap_abs,
    }
    explain_frame = split.explain_frame.iloc[-int(last_n_units) :].reset_index(drop=True)
    x_eval = split.X_explain.iloc[-int(last_n_units) :].reset_index(drop=True)
    y_eval = split.y_explain.iloc[-int(last_n_units) :].reset_index(drop=True)
    if explain_frame.empty:
        raise ValueError("No legacy CAISO holdout rows remain for additional baselines.")

    values_by_unit: dict[str, np.ndarray] = {}
    full_values: dict[str, float] = {}
    reference_scores_by_unit: dict[str, dict[str, dict[str, float]]] = {
        "Post-InfoDVA": {},
        "Prediction SHAP": {},
    }
    daily_shap_rows: list[dict[str, Any]] = []
    unit_ids: list[str] = []
    for row_idx, date_value in enumerate(explain_frame[split.date_column].astype(str)):
        true_prices = tuple(float(value) for value in y_eval.iloc[row_idx].to_numpy(dtype=float))
        coalition_predictions, coalition_realized_values, _, _ = _solve_all_coalitions_for_day(
            observation=x_eval.iloc[row_idx],
            true_prices=true_prices,
            date=f"legacy_additional_baselines_{date_value}",
            evaluator=coalition_evaluator,
            actual_parameters=config.storage_parameters,
            parameter_player_spec=config.parameter_player_spec,
            solver_params=solver_params,
            objective_tolerance=config.objective_tolerance,
        )
        decision_characteristic_values = (
            np.asarray(coalition_realized_values, dtype=float)
            - float(coalition_realized_values[0])
        )
        predictive_hourly_shap = compute_exact_shapley_values(
            coalition_predictions,
            feature_count=len(split.feature_columns),
        )
        predictive_daily_shap = predictive_hourly_shap.sum(axis=1)
        decision_shap = compute_exact_shapley_values(
            decision_characteristic_values,
            feature_count=len(split.feature_columns),
        )
        predictive_metrics = _decision_evaluation_metrics(
            predictive_daily_shap,
            decision_characteristic_values,
            split.feature_columns,
        )
        decision_metrics = _decision_evaluation_metrics(
            decision_shap,
            decision_characteristic_values,
            split.feature_columns,
        )
        predictive_ranking = build_attribution_ranking(
            predictive_daily_shap,
            split.feature_columns,
        )
        decision_ranking = build_attribution_ranking(
            decision_shap,
            split.feature_columns,
        )
        oracle_dispatch = solve_storage_dispatch_lexicographic(
            true_prices,
            config.storage_parameters,
            name=f"legacy_caiso_oracle_{date_value}",
            log_to_console=False,
            solver_params=solver_params,
            objective_tolerance=config.objective_tolerance,
        )
        daily_row: dict[str, Any] = {
            "date": date_value,
            "predictive_baseline_total": float(np.asarray(coalition_predictions[0]).sum()),
            "predictive_full_total": float(np.asarray(coalition_predictions[-1]).sum()),
            "predictive_total_gain": float(
                np.asarray(coalition_predictions[-1]).sum()
                - np.asarray(coalition_predictions[0]).sum()
            ),
            "decision_baseline_value": float(coalition_realized_values[0]),
            "decision_full_value": float(coalition_realized_values[-1]),
            "oracle_obj": float(oracle_dispatch.objective_value),
            "decision_value_gain": float(decision_characteristic_values[-1]),
            "abs_rank_spearman": compute_rank_spearman_from_rankings(
                predictive_ranking,
                decision_ranking,
            ),
            "abs_rank_kendall_tau": compute_rank_kendall_tau_from_rankings(
                predictive_ranking,
                decision_ranking,
            ),
        }
        for feature_name, shap_value in zip(
            split.feature_columns,
            predictive_daily_shap,
            strict=True,
        ):
            daily_row[f"predictive_shap_{feature_name}"] = float(shap_value)
        for feature_name, shap_value in zip(
            split.feature_columns,
            decision_shap,
            strict=True,
        ):
            daily_row[f"decision_shap_{feature_name}"] = float(shap_value)
        for metric_name, metric_value in predictive_metrics.items():
            daily_row[f"predictive_{metric_name}"] = metric_value
        for metric_name, metric_value in decision_metrics.items():
            daily_row[f"decision_{metric_name}"] = metric_value
        daily_shap_rows.append(daily_row)
        reference_scores_by_unit["Post-InfoDVA"][date_value] = _score_mapping(
            decision_shap,
            split.feature_columns,
        )
        reference_scores_by_unit["Prediction SHAP"][date_value] = _score_mapping(
            predictive_daily_shap,
            split.feature_columns,
        )
        values_by_unit[date_value] = decision_characteristic_values
        full_values[date_value] = float(coalition_realized_values[-1])
        unit_ids.append(date_value)
        print(
            f"[CAISO legacy {row_idx + 1}/{len(explain_frame)}] "
            f"computed coalitions for {date_value}",
            flush=True,
        )

    if daily_shap_output is not None:
        daily_shap_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(daily_shap_rows).to_csv(daily_shap_output, index=False)
        print(f"[CAISO legacy] wrote recomputed daily shap to {daily_shap_output}", flush=True)

    pfi_scores, pfi_local = _compute_caiso_pfi_scores(
        config=config,
        model=artifacts.model,
        explain_frame=explain_frame,
        feature_columns=split.feature_columns,
        target_columns=split.target_columns,
        date_column=split.date_column,
        full_values=full_values,
        permutation_seed=permutation_seed,
        solver_params=solver_params,
    )
    return compute_additional_baselines_from_coalitions(
        dataset="CAISO",
        unit_ids=unit_ids,
        player_names=split.feature_columns,
        decision_characteristic_by_unit=values_by_unit,
        reference_scores_by_unit=reference_scores_by_unit,
        pfi_scores=pfi_scores,
        pfi_local_scores_by_unit=pfi_local,
        ranking_mode=ranking_mode,
        bootstrap_config=bootstrap_config,
    )


def run_ems_additional_baselines(
    *,
    run_dir: Path,
    last_n_units: int,
    permutation_seed: int,
    ranking_mode: RankingMode,
    bootstrap_config: BootstrapConfig,
) -> AdditionalBaselineResult:
    metadata = _load_json(run_dir / "run_metadata.json")
    config = _ems_config_from_metadata(metadata, run_dir)
    player_names = tuple(str(player) for player in metadata["player_names"])
    unit_ids, values_by_unit = load_coalition_values(
        run_dir / "coalition_values.csv",
        unit_column=EMS_TIMESTAMP_COLUMN,
        last_n_units=last_n_units,
    )
    reference_scores_by_unit = _reference_scores_from_shap_csv(
        _resolve_ems_hourly_shap_path(config, metadata),
        unit_column=EMS_TIMESTAMP_COLUMN,
        unit_ids=unit_ids,
        player_names=player_names,
    )
    pfi_scores, pfi_local = _compute_ems_pfi_scores(
        config=config,
        run_metadata=metadata,
        selected_unit_ids=unit_ids,
        permutation_seed=permutation_seed,
    )
    return compute_additional_baselines_from_coalitions(
        dataset="EMS",
        unit_ids=unit_ids,
        player_names=player_names,
        decision_characteristic_by_unit=values_by_unit,
        reference_scores_by_unit=reference_scores_by_unit,
        pfi_scores=pfi_scores,
        pfi_local_scores_by_unit=pfi_local,
        ranking_mode=ranking_mode,
        bootstrap_config=bootstrap_config,
    )


def write_combined_outputs(
    *,
    results: Sequence[AdditionalBaselineResult],
    output: Path,
) -> pd.DataFrame:
    frames = []
    for result in results:
        frames.append(result.local_rows.assign(row_type="local"))
        frames.append(result.aggregate_rows.assign(row_type="aggregate"))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    return combined


def _prefix_result_units(
    result: AdditionalBaselineResult,
    *,
    model_id: str,
) -> AdditionalBaselineResult:
    return AdditionalBaselineResult(
        local_rows=_prefix_frame_units(result.local_rows, model_id=model_id),
        aggregate_rows=_prefix_frame_units(result.aggregate_rows, model_id=model_id),
        bootstrap_rows=result.bootstrap_rows.assign(model_id=model_id),
    )


def _prefix_frame_units(frame: pd.DataFrame, *, model_id: str) -> pd.DataFrame:
    prefixed = frame.copy()
    prefixed.insert(1, "model_id", model_id)
    prefixed["unit_id"] = (
        model_id + "::" + prefixed["unit_id"].astype(str)
    )
    return prefixed


def _resolved_output_paths(
    *,
    output: Path,
    table_output: Path,
    all_models: bool,
    ranking_mode: RankingMode,
) -> tuple[Path, Path]:
    if not all_models and ranking_mode == "method_default":
        return output, table_output
    suffix_parts = []
    if all_models:
        suffix_parts.append("all_models")
    if ranking_mode != "method_default":
        suffix_parts.append(ranking_mode)
    suffix = "_".join(suffix_parts)
    if output == DEFAULT_OUTPUT:
        output = output.with_name(f"{output.stem}_{suffix}{output.suffix}")
    if table_output == DEFAULT_TABLE_OUTPUT:
        table_output = table_output.with_name(
            f"{table_output.stem}_{suffix}{table_output.suffix}"
        )
    return output, table_output


def _discover_caiso_run_dirs(root: Path, *, evaluation_label: str) -> tuple[Path, ...]:
    pattern = f"xgb_*/models/xgb_*/{evaluation_label}"
    run_dirs = tuple(
        sorted(
            path
            for path in root.glob(pattern)
            if (path / "daily_shap.csv").exists()
            and (path.parent / "model_config.json").exists()
        )
    )
    if not run_dirs:
        raise FileNotFoundError(f"No CAISO run directories found for {root / pattern}.")
    return run_dirs


def _discover_ems_run_dirs(root: Path, *, relative_path: str) -> tuple[Path, ...]:
    run_dirs = tuple(
        sorted(
            model_dir / relative_path
            for model_dir in root.glob("xgb_*")
            if (model_dir / relative_path / "hourly_shap.csv").exists()
            and (model_dir / relative_path / "coalition_values.csv").exists()
        )
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"No EMS run directories found for {root} with relative path {relative_path}."
        )
    return run_dirs


def _model_id_from_run_dir(run_dir: Path) -> str:
    for part in reversed(run_dir.parts):
        if part.startswith("xgb_"):
            return part
    return run_dir.name


def build_amended_latex_table(
    *,
    output_csv: Path,
    caiso_reference_csv: Path,
    ems_reference_csv: Path,
    table_output: Path,
    bootstrap_config: BootstrapConfig,
    reference_sources: Sequence[ReferenceMetricSource] | None = None,
) -> str:
    additional = pd.read_csv(output_csv)
    local = additional.loc[additional["row_type"] == "local"].copy()
    local_bootstrap = _bootstrap_from_local_rows(local, bootstrap_config)
    reference_bootstrap = _reference_infidelity_bootstrap(
        caiso_reference_csv=caiso_reference_csv,
        ems_reference_csv=ems_reference_csv,
        bootstrap_config=bootstrap_config,
        sources=reference_sources,
    )
    summary = pd.concat([reference_bootstrap, local_bootstrap], ignore_index=True)
    tex = _format_latex_table(summary)
    table_output.parent.mkdir(parents=True, exist_ok=True)
    table_output.write_text(tex)
    return tex


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    ranking_mode = cast(RankingMode, args.ranking_mode)
    caiso_source = cast(CaisoSource, args.caiso_source)
    bootstrap_config = BootstrapConfig(
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        seed=args.bootstrap_seed,
    )
    output, table_output = _resolved_output_paths(
        output=args.output,
        table_output=args.table_output,
        all_models=bool(args.all_models),
        ranking_mode=ranking_mode,
    )
    caiso_reference_csv = args.caiso_reference_csv
    if caiso_reference_csv is None:
        if args.all_models:
            caiso_reference_csv = Path("")
        elif caiso_source == "legacy_holdout":
            caiso_reference_csv = DEFAULT_CAISO_LEGACY_DAILY_SHAP_OUTPUT
        elif args.caiso_run_dir is not None:
            caiso_reference_csv = args.caiso_run_dir / "daily_shap.csv"
    if caiso_reference_csv is None:
        raise SystemExit("--caiso-run-dir or --caiso-reference-csv is required.")
    ems_reference_csv = args.ems_reference_csv or (args.ems_run_dir / "hourly_shap.csv")
    reference_sources: Sequence[ReferenceMetricSource] | None = None
    if args.all_models:
        if caiso_source != "gdsi":
            raise SystemExit("--all-models currently requires --caiso-source gdsi.")
        caiso_run_dirs = _discover_caiso_run_dirs(
            args.caiso_root,
            evaluation_label=args.caiso_evaluation_label,
        )
        ems_run_dirs = _discover_ems_run_dirs(
            args.ems_root,
            relative_path=args.ems_run_relative_path,
        )
        print(
            f"Discovered {len(caiso_run_dirs)} CAISO runs and {len(ems_run_dirs)} EMS runs.",
            flush=True,
        )
        results = []
        references: list[ReferenceMetricSource] = []
        for run_dir in caiso_run_dirs:
            model_id = _model_id_from_run_dir(run_dir)
            result = run_caiso_additional_baselines(
                run_dir=run_dir,
                last_n_units=args.last_n_units,
                permutation_seed=args.permutation_seed,
                ranking_mode=ranking_mode,
                bootstrap_config=bootstrap_config,
            )
            results.append(_prefix_result_units(result, model_id=model_id))
            references.append(
                ReferenceMetricSource(
                    dataset="CAISO",
                    path=run_dir / "daily_shap.csv",
                    unit_column="date",
                    unit_prefix=model_id,
                )
            )
        for run_dir in ems_run_dirs:
            model_id = _model_id_from_run_dir(run_dir)
            result = run_ems_additional_baselines(
                run_dir=run_dir,
                last_n_units=args.last_n_units,
                permutation_seed=args.permutation_seed,
                ranking_mode=ranking_mode,
                bootstrap_config=bootstrap_config,
            )
            results.append(_prefix_result_units(result, model_id=model_id))
            references.append(
                ReferenceMetricSource(
                    dataset="EMS",
                    path=run_dir / "hourly_shap.csv",
                    unit_column=EMS_TIMESTAMP_COLUMN,
                    unit_prefix=model_id,
                )
            )
        reference_sources = references
    else:
        if caiso_source == "legacy_holdout":
            caiso_result = run_caiso_legacy_holdout_additional_baselines(
                last_n_units=args.last_n_units,
                permutation_seed=args.permutation_seed,
                ranking_mode=ranking_mode,
                bootstrap_config=bootstrap_config,
                daily_shap_output=caiso_reference_csv,
            )
        else:
            if args.caiso_run_dir is None:
                raise SystemExit("--caiso-run-dir is required with --caiso-source gdsi.")
            caiso_result = run_caiso_additional_baselines(
                run_dir=args.caiso_run_dir,
                last_n_units=args.last_n_units,
                permutation_seed=args.permutation_seed,
                ranking_mode=ranking_mode,
                bootstrap_config=bootstrap_config,
            )
        results = [
            caiso_result,
            run_ems_additional_baselines(
                run_dir=args.ems_run_dir,
                last_n_units=args.last_n_units,
                permutation_seed=args.permutation_seed,
                ranking_mode=ranking_mode,
                bootstrap_config=bootstrap_config,
            ),
        ]
    write_combined_outputs(results=results, output=output)
    tex = build_amended_latex_table(
        output_csv=output,
        caiso_reference_csv=caiso_reference_csv,
        ems_reference_csv=ems_reference_csv,
        table_output=table_output,
        bootstrap_config=bootstrap_config,
        reference_sources=reference_sources,
    )
    print(f"Wrote {output}")
    print(f"Wrote {table_output}")
    print(tex)


def _compute_caiso_pfi_scores(
    *,
    config: CaisoShapCaseStudyConfig,
    model: Any,
    explain_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    date_column: str,
    full_values: Mapping[str, float],
    permutation_seed: int,
    solver_params: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    rng = np.random.default_rng(permutation_seed)
    pfi_local: dict[str, dict[str, float]] = {
        str(unit_id): {} for unit_id in explain_frame[date_column].astype(str)
    }
    pfi_scores: dict[str, float] = {}
    base_features = explain_frame.loc[:, list(feature_columns)].reset_index(drop=True)
    true_targets = explain_frame.loc[:, list(target_columns)].reset_index(drop=True)
    unit_ids = tuple(explain_frame[date_column].astype(str))
    for feature in feature_columns:
        permuted = base_features.copy()
        permuted[feature] = permuted[feature].to_numpy()[rng.permutation(len(permuted))]
        predictions = np.asarray(model.predict(permuted), dtype=float)
        if predictions.ndim == 1:
            predictions = predictions[:, np.newaxis]
        drops = []
        for row_idx, unit_id in enumerate(unit_ids):
            dispatch = solve_storage_dispatch_lexicographic(
                predictions[row_idx],
                config.storage_parameters,
                name=f"caiso_pfi_{feature}_{unit_id}",
                log_to_console=False,
                solver_params=solver_params,
                objective_tolerance=config.objective_tolerance,
            )
            realized = evaluate_storage_dispatch_result(
                true_targets.iloc[row_idx].to_numpy(dtype=float, copy=True),
                dispatch,
                config.storage_parameters,
            ).objective_value
            drop = float(full_values[unit_id] - realized)
            pfi_local[unit_id][feature] = drop
            drops.append(drop)
        pfi_scores[str(feature)] = float(np.mean(np.abs(drops)))
        print(f"[CAISO PFI] {feature}: {pfi_scores[str(feature)]:.6g}", flush=True)
    return pfi_scores, pfi_local


def _compute_ems_pfi_scores(
    *,
    config: EmsExactShapConfig,
    run_metadata: Mapping[str, Any],
    selected_unit_ids: Sequence[str],
    permutation_seed: int,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    x_frame, y_frame, metadata = _load_ems_frames(config)
    zone_order = _load_zone_order(config.zone_order_path, tuple(_target_columns(y_frame)))
    target_columns = tuple(zone_order["target_column"].astype(str))
    zip_codes = tuple(zone_order["zip_code"].astype(str))
    y_frame = y_frame.loc[:, [EMS_TIMESTAMP_COLUMN, *target_columns]].copy()
    feature_columns = _resolve_feature_columns(x_frame, metadata)
    feature_groups = build_ems_feature_groups(feature_columns)
    time_split = _build_time_split(
        x_frame.loc[:, [EMS_TIMESTAMP_COLUMN, *feature_columns]],
        y_frame,
        test_months=config.test_months,
    )
    train_x, train_y = _maybe_sample_training_rows(
        time_split.train_x,
        time_split.train_y,
        train_sample_rows=config.train_sample_rows,
        random_state=config.random_state,
    )
    explain_x, explain_y, _ = _sample_explanation_hours(
        time_split.holdout_x,
        time_split.holdout_y,
        time_split.holdout_source_rows,
        holdout_hours=config.holdout_hours,
        max_hours=config.max_hours,
        random_state=config.random_state,
    )
    explain_x[EMS_TIMESTAMP_COLUMN] = explain_x[EMS_TIMESTAMP_COLUMN].astype(str)
    explain_y[EMS_TIMESTAMP_COLUMN] = explain_y[EMS_TIMESTAMP_COLUMN].astype(str)
    selected = set(str(unit_id) for unit_id in selected_unit_ids)
    explain_x = explain_x.loc[explain_x[EMS_TIMESTAMP_COLUMN].isin(selected)].reset_index(
        drop=True
    )
    explain_y = explain_y.loc[explain_y[EMS_TIMESTAMP_COLUMN].isin(selected)].reset_index(
        drop=True
    )
    if tuple(explain_x[EMS_TIMESTAMP_COLUMN].astype(str)) != tuple(selected_unit_ids):
        explain_x = explain_x.set_index(EMS_TIMESTAMP_COLUMN).loc[list(selected_unit_ids)].reset_index()
        explain_y = explain_y.set_index(EMS_TIMESTAMP_COLUMN).loc[list(selected_unit_ids)].reset_index()

    model = _fit_xgb_regressor(
        train_frame=train_x,
        y_train=train_y.loc[:, list(target_columns)],
        feature_columns=feature_columns,
        config=config,
    )
    distance_matrix = _load_distance_matrix(config.distance_matrix_path)
    coverage_matrix = build_coverage_matrix(
        distance_matrix,
        zip_codes,
        coverage_radius_km=config.coverage_radius_km,
    )
    hourly_path = _resolve_ems_hourly_shap_path(config, run_metadata)
    hourly = pd.read_csv(hourly_path)
    hourly[EMS_TIMESTAMP_COLUMN] = hourly[EMS_TIMESTAMP_COLUMN].astype(str)
    full_values = dict(
        zip(
            hourly[EMS_TIMESTAMP_COLUMN].astype(str),
            hourly["decision_full_value"].astype(float),
            strict=False,
        )
    )
    solver_params = _build_solver_params(config)
    rng = np.random.default_rng(permutation_seed)
    pfi_local: dict[str, dict[str, float]] = {str(unit_id): {} for unit_id in selected_unit_ids}
    pfi_scores: dict[str, float] = {}
    base_features = explain_x.loc[:, list(feature_columns)].reset_index(drop=True)
    for group in feature_groups:
        permuted = base_features.copy()
        row_order = rng.permutation(len(permuted))
        permuted.loc[:, list(group.columns)] = permuted.loc[row_order, list(group.columns)].to_numpy()
        predictions = np.maximum(
            np.asarray(
                model.predict(_frame_to_feature_matrix(permuted, feature_columns)),
                dtype=float,
            ),
            0.0,
        )
        if predictions.ndim == 1:
            predictions = predictions[:, np.newaxis]
        drops = []
        for row_idx, unit_id in enumerate(selected_unit_ids):
            solution = solve_ems_coverage(
                predictions[row_idx],
                coverage_matrix,
                zip_codes,
                facility_budget=config.facility_budget,
                solver_name=config.coverage_solver,
                name=f"ems_pfi_{group.name}_{row_idx}",
                log_to_console=False,
                solver_params=solver_params,
                optimization_solver=config.optimization_solver,
                objective_tolerance=config.objective_tolerance,
            )
            true_demand = explain_y.loc[row_idx, list(target_columns)].to_numpy(
                dtype=float,
                copy=True,
            )
            realized = _realized_coverage_value(solution.covered_zone_indices, true_demand)
            drop = float(full_values[str(unit_id)] - realized)
            pfi_local[str(unit_id)][group.name] = drop
            drops.append(drop)
        pfi_scores[group.name] = float(np.mean(np.abs(drops)))
        print(f"[EMS PFI] {group.name}: {pfi_scores[group.name]:.6g}", flush=True)
    return pfi_scores, pfi_local


def _local_record(
    *,
    dataset: str,
    unit_id: str,
    method: BaselineMethod,
    player_names: Sequence[str],
    scores: Mapping[str, float],
    decision_characteristic_values: np.ndarray,
    score_scope: str,
    ranking_mode: RankingMode,
    local_scores: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    score_vector = np.asarray([float(scores[player]) for player in player_names], dtype=float)
    insertion_auc = compute_decision_insertion_auc(
        score_vector,
        decision_characteristic_values,
        player_names,
    )
    record: dict[str, Any] = {
        "dataset": dataset,
        "unit_id": unit_id,
        "method": method,
        "metric": "Decision insertion AUC \\uparrow",
        "score_scope": score_scope,
        "ranking_mode": ranking_mode,
        "insertion_auc": insertion_auc,
        "ranking": json.dumps(_ranking_from_scores(scores, player_names)),
        "full_decision_value": float(decision_characteristic_values[-1]),
        "baseline_decision_value": float(decision_characteristic_values[0]),
    }
    for player in player_names:
        record[f"score_{player}"] = float(scores[player])
        if local_scores is not None and player in local_scores:
            record[f"local_score_{player}"] = float(local_scores[player])
    return record


def _aggregate_score_record(
    *,
    dataset: str,
    method: BaselineMethod,
    player_names: Sequence[str],
    scores: Mapping[str, float],
    score_scope: str,
    ranking_mode: RankingMode,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "dataset": dataset,
        "unit_id": "__aggregate__",
        "method": method,
        "metric": "Decision insertion AUC \\uparrow",
        "score_scope": score_scope,
        "ranking_mode": ranking_mode,
        "ranking": json.dumps(_ranking_from_scores(scores, player_names)),
    }
    for player in player_names:
        record[f"score_{player}"] = float(scores[player])
    return record


def _lofo_scores(values: np.ndarray, player_names: Sequence[str]) -> dict[str, float]:
    full_mask = len(values) - 1
    return {
        str(player): float(values[full_mask] - values[full_mask ^ (1 << idx)])
        for idx, player in enumerate(player_names)
    }


def _mean_local_lofo_scores(
    values_by_unit: Mapping[str, np.ndarray],
    player_names: Sequence[str],
) -> dict[str, float]:
    by_player = {str(player): [] for player in player_names}
    for values in values_by_unit.values():
        scores = _lofo_scores(values, player_names)
        for player, score in scores.items():
            by_player[player].append(score)
    return {player: float(np.mean(values)) for player, values in by_player.items()}


def _mean_abs_local_scores(
    scores_by_unit: Mapping[str, Mapping[str, float]],
    player_names: Sequence[str],
) -> dict[str, float]:
    return {
        str(player): float(
            np.mean(
                [
                    abs(float(scores_by_unit[str(unit_id)][str(player)]))
                    for unit_id in scores_by_unit
                ]
            )
        )
        for player in player_names
    }


def _mean_signed_local_scores(
    scores_by_unit: Mapping[str, Mapping[str, float]],
    player_names: Sequence[str],
) -> dict[str, float]:
    return {
        str(player): float(
            np.mean(
                [
                    float(scores_by_unit[str(unit_id)][str(player)])
                    for unit_id in scores_by_unit
                ]
            )
        )
        for player in player_names
    }


def _ranking_mode_uses_local(method: str, ranking_mode: RankingMode) -> bool:
    if ranking_mode == "local":
        return True
    if ranking_mode == "global":
        return False
    method_default_local = {
        "Post-InfoDVA",
        "Prediction SHAP",
        "Leave-one-feature-out",
    }
    if ranking_mode in {"method_default", "paper_original"}:
        return method in method_default_local
    return method not in method_default_local


def _greedy_insertion_scores(
    values_by_unit: Mapping[str, np.ndarray],
    player_names: Sequence[str],
) -> dict[str, float]:
    player_count = len(player_names)
    selected_mask = 0
    remaining = set(range(player_count))
    order: list[int] = []
    while remaining:
        best_idx = min(remaining)
        best_gain = -math.inf
        for idx in sorted(remaining):
            gain = float(
                np.mean(
                    [
                        values[selected_mask | (1 << idx)] - values[selected_mask]
                        for values in values_by_unit.values()
                    ]
                )
            )
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_idx = idx
        order.append(best_idx)
        selected_mask |= 1 << best_idx
        remaining.remove(best_idx)
    return _scores_from_order(order, player_names)


def _scores_from_order(order: Sequence[int], player_names: Sequence[str]) -> dict[str, float]:
    n = len(player_names)
    scores = {str(player): 0.0 for player in player_names}
    for rank, idx in enumerate(order):
        scores[str(player_names[idx])] = float(n - rank)
    return scores


def _signed_score_order(
    scores: Mapping[str, float],
    player_names: Sequence[str],
) -> tuple[int, ...]:
    indexed = [
        (idx, str(player), float(scores[str(player)]))
        for idx, player in enumerate(player_names)
    ]
    return tuple(
        idx for idx, _, _ in sorted(indexed, key=lambda item: (-item[2], item[0]))
    )


def _score_mapping(
    values: Sequence[float] | np.ndarray,
    player_names: Sequence[str],
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.shape != (len(player_names),):
        raise ValueError(
            "Attribution scores must have one value per player. "
            f"Expected {len(player_names)}, got shape {array.shape}."
        )
    return {
        str(player): float(array[idx])
        for idx, player in enumerate(player_names)
    }


def _ranking_from_scores(
    scores: Mapping[str, float],
    player_names: Sequence[str],
) -> list[str]:
    indexed = [
        (idx, str(player), abs(float(scores[str(player)])))
        for idx, player in enumerate(player_names)
    ]
    return [player for _, player, _ in sorted(indexed, key=lambda item: (-item[2], item[0]))]


def _validate_coalition_vector(
    values: Sequence[float] | np.ndarray,
    *,
    player_count: int,
    unit_id: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    expected = 1 << player_count
    if array.shape != (expected,):
        raise ValueError(
            f"Unit {unit_id!r} has {array.shape[0]} coalition values; expected {expected}."
        )
    return array


def _merge_metric_summary(
    aggregate: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    if bootstrap.empty:
        merged = aggregate.copy()
        merged["insertion_auc_mean"] = np.nan
        merged["insertion_auc_ci_low"] = np.nan
        merged["insertion_auc_ci_high"] = np.nan
        merged["n_units"] = 0
        return merged
    metric = bootstrap.loc[
        :,
        ["dataset", "method", "mean", "mean_ci_low", "mean_ci_high", "n_units"],
    ].rename(
        columns={
            "mean": "insertion_auc_mean",
            "mean_ci_low": "insertion_auc_ci_low",
            "mean_ci_high": "insertion_auc_ci_high",
        }
    )
    return aggregate.merge(metric, on=["dataset", "method"], how="left")


def _bootstrap_from_local_rows(
    local: pd.DataFrame,
    bootstrap_config: BootstrapConfig,
) -> pd.DataFrame:
    frame = local.loc[:, ["dataset", "unit_id", "method", "insertion_auc"]].rename(
        columns={"insertion_auc": "value"}
    )
    frame["metric"] = "Decision insertion AUC \\uparrow"
    return bootstrap_metric_table(frame, reference_method=None, config=bootstrap_config)


def _reference_scores_from_shap_csv(
    path: Path,
    *,
    unit_column: str,
    unit_ids: Sequence[str],
    player_names: Sequence[str],
) -> dict[str, dict[str, dict[str, float]]]:
    source = pd.read_csv(path)
    _require_columns(source, (unit_column,))
    source[unit_column] = source[unit_column].astype(str)
    source = source.set_index(unit_column, drop=False)
    references: dict[str, dict[str, dict[str, float]]] = {
        "Post-InfoDVA": {},
        "Prediction SHAP": {},
    }
    for method, prefix in [
        ("Post-InfoDVA", "decision_shap"),
        ("Prediction SHAP", "predictive_shap"),
    ]:
        columns = tuple(f"{prefix}_{player}" for player in player_names)
        _require_columns(source, columns)
        for unit_id in unit_ids:
            row = source.loc[str(unit_id)]
            references[method][str(unit_id)] = {
                str(player): float(row[f"{prefix}_{player}"])
                for player in player_names
            }
    return references


def _resolve_ems_hourly_shap_path(
    config: EmsExactShapConfig,
    run_metadata: Mapping[str, Any],
) -> Path:
    hourly_path = config.outdir / "hourly_shap.csv"
    if hourly_path.exists():
        return hourly_path
    canonical_dir = run_metadata.get("canonical_run_dir")
    if canonical_dir is not None:
        canonical_path = Path(str(canonical_dir)) / "hourly_shap.csv"
        if canonical_path.exists():
            return canonical_path
    return hourly_path


def _reference_infidelity_bootstrap(
    *,
    caiso_reference_csv: Path,
    ems_reference_csv: Path,
    bootstrap_config: BootstrapConfig,
    sources: Sequence[ReferenceMetricSource] | None = None,
) -> pd.DataFrame:
    frames = []
    reference_sources = sources or (
        ReferenceMetricSource("CAISO", caiso_reference_csv, "date"),
        ReferenceMetricSource("EMS", ems_reference_csv, EMS_TIMESTAMP_COLUMN),
    )
    for metric_source in reference_sources:
        source = pd.read_csv(metric_source.path)
        for metric, method, column in [
            (
                "Decision infidelity \\downarrow",
                "Post-InfoDVA",
                "decision_decision_infidelity",
            ),
            (
                "Decision infidelity \\downarrow",
                "Prediction SHAP",
                "predictive_decision_infidelity",
            ),
        ]:
            frame = source.loc[:, [metric_source.unit_column, column]].tail(30).copy()
            frame = frame.rename(
                columns={metric_source.unit_column: "unit_id", column: "value"}
            )
            frame["unit_id"] = frame["unit_id"].astype(str)
            if metric_source.unit_prefix is not None:
                frame["unit_id"] = metric_source.unit_prefix + "::" + frame["unit_id"]
            frame["dataset"] = metric_source.dataset
            frame["metric"] = metric
            frame["method"] = method
            frames.append(frame)
    long = pd.concat(frames, ignore_index=True)
    return bootstrap_metric_table(long, reference_method=None, config=bootstrap_config)


def _format_latex_table(summary: pd.DataFrame) -> str:
    auc_methods = [
        "Post-InfoDVA",
        "Prediction SHAP",
        "Leave-one-feature-out",
        "Greedy decision insertion",
        "Downstream permutation feature importance",
    ]
    labels = {
        "Downstream permutation feature importance": "Permutation feature importance",
    }
    lookup = summary.set_index(["metric", "dataset", "method"], drop=False)

    def fmt_auc(dataset: str, method: str) -> str:
        row = lookup.loc[("Decision insertion AUC \\uparrow", dataset, method)]
        return _fmt_ci(row, decimals=3)

    def fmt_infidelity(dataset: str, method: str) -> str:
        row = lookup.loc[("Decision infidelity \\downarrow", dataset, method)]
        if dataset == "EMS" and abs(float(row["mean"])) < 1:
            return _fmt_ci(row, decimals=4)
        return _fmt_ci(row, decimals=2, comma=True)

    def best_method(
        metric: str,
        dataset: str,
        methods: Sequence[str],
        *,
        higher_is_better: bool,
    ) -> str:
        rows_by_method = {
            method: lookup.loc[(metric, dataset, method)]
            for method in methods
        }
        key = lambda method: float(rows_by_method[method]["mean"])
        return max(methods, key=key) if higher_is_better else min(methods, key=key)

    def maybe_bold(value: str, *, enabled: bool) -> str:
        return f"\\textbf{{{value}}}" if enabled else value

    auc_best = {
        dataset: best_method(
            "Decision insertion AUC \\uparrow",
            dataset,
            auc_methods,
            higher_is_better=True,
        )
        for dataset in ("CAISO", "EMS")
    }
    infidelity_methods = ["Post-InfoDVA", "Prediction SHAP"]
    infidelity_best = {
        dataset: best_method(
            "Decision infidelity \\downarrow",
            dataset,
            infidelity_methods,
            higher_is_better=False,
        )
        for dataset in ("CAISO", "EMS")
    }

    rows = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        (
            r"\caption{Attribution metric comparison across methods for CAISO and EMS "
            r"with 95\% paired bootstrap confidence intervals on the last 30 "
            r"evaluation units. Values are mean [CI].}"
        ),
        r"\label{tab:combined_attribution_metrics_bootstrap}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Method} & \textbf{CAISO} & \textbf{EMS} \\",
        r"\midrule",
        r"\multirow{5}{*}{Decision insertion AUC $\uparrow$}",
    ]
    for method in auc_methods:
        label = labels.get(method, method)
        caiso = maybe_bold(fmt_auc("CAISO", method), enabled=method == auc_best["CAISO"])
        ems = maybe_bold(fmt_auc("EMS", method), enabled=method == auc_best["EMS"])
        rows.append(f"  & {label} & {caiso} & {ems} \\\\")
    rows.extend(
        [
            r"\midrule",
            r"\multirow{2}{*}{Decision infidelity $\downarrow$}",
        ]
    )
    for method in infidelity_methods:
        caiso = maybe_bold(
            fmt_infidelity("CAISO", method),
            enabled=method == infidelity_best["CAISO"],
        )
        ems = maybe_bold(
            fmt_infidelity("EMS", method),
            enabled=method == infidelity_best["EMS"],
        )
        rows.append(f"  & {method} & {caiso} & {ems} \\\\")
    rows.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.25em}",
            (
                r"\footnotesize\emph{Note.} Insertion AUC is computed only on "
                r"evaluation units with positive full decision-value gain, matching "
                r"the metric definition used in the case-study outputs."
            ),
            r"\end{table}",
        ]
    )
    return "\n".join(rows) + "\n"


def _fmt_ci(row: pd.Series, *, decimals: int, comma: bool = False) -> str:
    spec = f",.{decimals}f" if comma else f".{decimals}f"
    return (
        f"{float(row['mean']):{spec}} "
        f"[{float(row['mean_ci_low']):{spec}}, {float(row['mean_ci_high']):{spec}}]"
    )


def _caiso_config_from_payload(payload: Mapping[str, Any]) -> CaisoShapCaseStudyConfig:
    storage_payload = cast(Mapping[str, Any], payload.get("storage_parameters", {}))
    storage = StorageDispatchParameters(
        energy_capacity=float(storage_payload.get("energy_capacity", 2.0)),
        power_limit=float(storage_payload.get("power_limit", 1.0)),
        charge_efficiency=float(storage_payload.get("charge_efficiency", 0.95)),
        discharge_efficiency=float(storage_payload.get("discharge_efficiency", 0.95)),
        throughput_penalty=float(storage_payload.get("throughput_penalty", 0.0)),
        initial_state_of_charge=float(
            storage_payload.get("initial_state_of_charge", 1.0)
        ),
        terminal_state_of_charge=float(
            storage_payload.get("terminal_state_of_charge", 1.0)
        ),
    )
    return CaisoShapCaseStudyConfig(
        dataset_path=Path(str(payload["dataset_path"])),
        model_name=str(payload.get("model_name", "xgb")),
        random_state=int(payload.get("random_state", 0)),
        n_jobs=1,
        xgb_n_estimators=int(payload.get("xgb_n_estimators", 100)),
        xgb_max_depth=int(payload.get("xgb_max_depth", 3)),
        xgb_learning_rate=float(payload.get("xgb_learning_rate", 0.05)),
        xgb_subsample=float(payload.get("xgb_subsample", 0.9)),
        xgb_colsample_bytree=float(payload.get("xgb_colsample_bytree", 0.9)),
        xgb_reg_lambda=float(payload.get("xgb_reg_lambda", 1.0)),
        xgb_verbosity=int(payload.get("xgb_verbosity", 0)),
        background_days=int(payload.get("background_days", 365)),
        solver_seed=int(payload.get("solver_seed", 0)),
        mip_gap=float(payload.get("mip_gap", 0.0)),
        mip_gap_abs=float(payload.get("mip_gap_abs", 1e-9)),
        objective_tolerance=float(payload.get("objective_tolerance", 1e-6)),
        storage_parameters=storage,
    )


def _ems_config_from_metadata(
    metadata: Mapping[str, Any],
    run_dir: Path,
) -> EmsExactShapConfig:
    xgb_params = cast(Mapping[str, Any], metadata.get("xgb_params", {}))
    solver_params = cast(Mapping[str, Any], metadata.get("solver_params", {}))
    return EmsExactShapConfig(
        x_path=Path(str(metadata.get("x_path"))),
        y_path=Path(str(metadata.get("y_path"))),
        metadata_path=(
            None
            if metadata.get("metadata_path") is None
            else Path(str(metadata.get("metadata_path")))
        ),
        zone_order_path=Path(str(metadata.get("zone_order_path"))),
        distance_matrix_path=Path(str(metadata.get("distance_matrix_path"))),
        outdir=run_dir,
        holdout_hours=int(metadata.get("holdout_hours", 100)),
        test_months=int(metadata.get("test_months", 1)),
        max_hours=metadata.get("max_hours"),
        background_rows=int(metadata.get("background_rows", 100)),
        coalition_batch_size=int(metadata.get("coalition_batch_size", 64)),
        random_state=int(metadata.get("random_state", 0)),
        model_id=str(metadata.get("model_id", "xgb_001")),
        xgb_n_estimators=int(xgb_params.get("n_estimators", 100)),
        xgb_max_depth=int(xgb_params.get("max_depth", 3)),
        xgb_learning_rate=float(xgb_params.get("learning_rate", 0.05)),
        xgb_subsample=float(xgb_params.get("subsample", 0.9)),
        xgb_colsample_bytree=float(xgb_params.get("colsample_bytree", 0.9)),
        xgb_reg_lambda=float(xgb_params.get("reg_lambda", 1.0)),
        xgb_verbosity=int(xgb_params.get("verbosity", 0)),
        train_sample_rows=metadata.get("train_sample_rows"),
        coverage_radius_km=float(metadata.get("coverage_radius_km", 1.0)),
        facility_budget=int(metadata.get("facility_budget", 8)),
        solver_seed=int(solver_params.get("Seed", metadata.get("solver_seed", 0))),
        mip_gap=float(solver_params.get("MIPGap", 0.0)),
        mip_gap_abs=float(solver_params.get("MIPGapAbs", 1e-9)),
        gurobi_threads=int(solver_params.get("Threads", 1)),
        optimization_solver=str(metadata.get("coverage_backend_solver", "highs")),
        objective_tolerance=float(metadata.get("objective_tolerance", 1e-6)),
        coverage_solver=str(metadata.get("coverage_solver", "exact")),
        compute_cvar_decision_shap=False,
    )


def _resolve_caiso_model_dir(run_dir: Path) -> Path:
    path = run_dir
    if (path / "model_config.json").exists():
        return path
    parent = path.parent
    while parent != parent.parent:
        if (parent / "model_config.json").exists():
            return parent
        parent = parent.parent
    raise FileNotFoundError(f"Could not find model_config.json above {run_dir}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def _find_upward_json(start: Path, filename: str) -> dict[str, Any] | None:
    path = start
    while path != path.parent:
        candidate = path / filename
        if candidate.exists():
            return _load_json(candidate)
        path = path.parent
    return None


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError("Missing required columns: " + ", ".join(missing))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute LOFO, downstream permutation feature-importance, and greedy "
            "decision insertion baselines, then write local and aggregate AUCs."
        )
    )
    parser.add_argument(
        "--caiso-run-dir",
        type=Path,
        default=None,
        help=(
            "CAISO run directory for --caiso-source gdsi. The default legacy "
            "CAISO source is fully specified in this script and does not read "
            "an old results directory."
        ),
    )
    parser.add_argument(
        "--caiso-source",
        choices=("legacy_holdout", "gdsi"),
        default="legacy_holdout",
        help=(
            "CAISO recipe to use. legacy_holdout recreates the original paper "
            "CAISO setup from run_metadata.json; gdsi uses the newer guided "
            "validation split/model_config.json layout."
        ),
    )
    parser.add_argument(
        "--ems-run-dir",
        type=Path,
        default=DEFAULT_EMS_RUN_DIR,
    )
    parser.add_argument("--last-n-units", type=int, default=30)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument(
        "--ranking-mode",
        choices=("method_default", "opposite", "local", "global", "paper_original"),
        default="paper_original",
        help=(
            "Ranking scope to use before computing insertion AUC. method_default "
            "uses local DVA/Prediction/LOFO and global greedy/PFI; paper_original "
            "uses local DVA/Prediction, corrected local absolute LOFO, global "
            "same-evaluation greedy, and global signed downstream permutation; "
            "opposite flips method_default choices."
        ),
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all discovered CAISO and EMS xgb_* model directories.",
    )
    parser.add_argument("--caiso-root", type=Path, default=Path("results/caiso/gdsi"))
    parser.add_argument("--caiso-evaluation-label", default="test_baseline")
    parser.add_argument(
        "--ems-root",
        type=Path,
        default=Path("results/ems/experiment_c_solver_dva"),
    )
    parser.add_argument(
        "--ems-run-relative-path",
        default="exact_vs_naive_post/design_coalitions/mask_0_exact_tau1_p8",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE_OUTPUT)
    parser.add_argument(
        "--caiso-reference-csv",
        type=Path,
        default=None,
        help=(
            "CAISO reference metrics CSV. Defaults to a recomputed legacy "
            "daily_shap CSV for --caiso-source legacy_holdout, otherwise "
            "to --caiso-run-dir/daily_shap.csv."
        ),
    )
    parser.add_argument(
        "--ems-reference-csv",
        type=Path,
        default=None,
        help="Defaults to --ems-run-dir/hourly_shap.csv.",
    )
    return parser


if __name__ == "__main__":
    main()
