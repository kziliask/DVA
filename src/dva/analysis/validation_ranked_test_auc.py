from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from dva.analysis.additional_decision_ranking_baselines import (
    _caiso_config_from_payload,
    _compute_caiso_pfi_scores,
    _compute_ems_pfi_scores,
    _discover_caiso_run_dirs,
    _discover_ems_run_dirs,
    _ems_config_from_metadata,
    _find_upward_json,
    _greedy_insertion_scores,
    _load_json,
    _lofo_scores,
    _mean_abs_local_scores,
    _model_id_from_run_dir,
    _reference_scores_from_shap_csv,
    _resolve_caiso_model_dir,
    load_coalition_values,
)
from dva.analysis.caiso_shap import (
    _build_coalition_evaluator,
    _solve_all_coalitions_for_day,
    compute_exact_shapley_values,
)
from dva.analysis.evaluation_metrics import compute_decision_insertion_auc
from dva.analysis.ems_exact_shap import (
    EMS_TIMESTAMP_COLUMN,
    GroupedBackgroundCoalitionPredictor,
    _build_solver_params,
    _build_time_split,
    _fit_xgb_regressor,
    _load_distance_matrix,
    _load_ems_frames,
    _load_zone_order,
    _maybe_sample_training_rows,
    _realized_coverage_value,
    _resolve_feature_columns,
    _sample_background_frame,
    _target_columns,
    build_coverage_matrix,
    build_ems_feature_groups,
    solve_ems_coverage,
)
from dva.analysis.paired_bootstrap import BootstrapConfig, bootstrap_metric_table
from dva.analysis.run_caiso_decision_shap_guided_validation import (
    build_fixed_caiso_guided_validation_split,
)
from dva.model.train import train_model


DEFAULT_OUTPUT = Path("results/bootstrap_ci/validation_ranked_test_auc.csv")
DEFAULT_TABLE_OUTPUT = Path(
    "results/bootstrap_ci/validation_ranked_test_auc.tex"
)


def run_validation_ranked_test_auc(
    *,
    caiso_root: Path,
    evaluation_label: str,
    ems_root: Path,
    ems_run_relative_path: str,
    last_n_units: int,
    permutation_seed: int,
    ems_test_max_units: int | None,
    bootstrap_config: BootstrapConfig,
    output: Path,
    table_output: Path,
) -> str:
    rows: list[dict[str, Any]] = []
    run_dirs = _discover_caiso_run_dirs(caiso_root, evaluation_label=evaluation_label)
    print(f"Discovered {len(run_dirs)} CAISO runs.", flush=True)
    for run_index, run_dir in enumerate(run_dirs, start=1):
        model_id = _model_id_from_run_dir(run_dir)
        print(f"[{run_index}/{len(run_dirs)}] learning validation rankings for {model_id}", flush=True)
        rows.extend(
            _run_caiso_model(
                run_dir=run_dir,
                model_id=model_id,
                last_n_units=last_n_units,
                permutation_seed=permutation_seed,
            )
        )
    ems_run_dirs = _discover_ems_run_dirs(ems_root, relative_path=ems_run_relative_path)
    print(f"Discovered {len(ems_run_dirs)} EMS runs.", flush=True)
    for run_index, run_dir in enumerate(ems_run_dirs, start=1):
        model_id = _model_id_from_run_dir(run_dir)
        print(f"[{run_index}/{len(ems_run_dirs)}] learning EMS validation rankings for {model_id}", flush=True)
        rows.extend(
            _run_ems_model(
                run_dir=run_dir,
                model_id=model_id,
                last_n_units=last_n_units,
                permutation_seed=permutation_seed,
                ems_test_max_units=ems_test_max_units,
            )
        )
    local = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    local.to_csv(output, index=False)

    bootstrap = _bootstrap_auc(local, bootstrap_config)
    tex = _format_auc_table(bootstrap)
    table_output.parent.mkdir(parents=True, exist_ok=True)
    table_output.write_text(tex)
    print(f"Wrote {output}", flush=True)
    print(f"Wrote {table_output}", flush=True)
    print(tex)
    return tex


def _run_caiso_model(
    *,
    run_dir: Path,
    model_id: str,
    last_n_units: int,
    permutation_seed: int,
) -> list[dict[str, Any]]:
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
    validation_frame = split.validation_frame.tail(int(last_n_units)).reset_index(drop=True)
    test_frame = split.test_frame.tail(int(last_n_units)).reset_index(drop=True)
    if validation_frame.empty or test_frame.empty:
        raise ValueError(f"{model_id} has an empty validation or test frame.")

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

    validation = _compute_caiso_coalitions(
        frame=validation_frame,
        frame_label="validation",
        model_id=model_id,
        feature_columns=split.feature_columns,
        target_columns=split.target_columns,
        date_column=split.date_column,
        evaluator=coalition_evaluator,
        config=config,
        solver_params=solver_params,
    )
    test = _compute_caiso_coalitions(
        frame=test_frame,
        frame_label="test",
        model_id=model_id,
        feature_columns=split.feature_columns,
        target_columns=split.target_columns,
        date_column=split.date_column,
        evaluator=coalition_evaluator,
        config=config,
        solver_params=solver_params,
    )
    pfi_scores, _ = _compute_caiso_pfi_scores(
        config=config,
        model=artifacts.model,
        explain_frame=validation_frame,
        feature_columns=split.feature_columns,
        target_columns=split.target_columns,
        date_column=split.date_column,
        full_values=validation.full_values_by_unit,
        permutation_seed=permutation_seed,
        solver_params=solver_params,
    )
    scores_by_method: dict[str, Mapping[str, float]] = {
        "Post-InfoDVA": _mean_abs_local_scores(
            validation.decision_shap_by_unit,
            split.feature_columns,
        ),
        "Prediction SHAP": _mean_abs_local_scores(
            validation.predictive_shap_by_unit,
            split.feature_columns,
        ),
        "Leave-one-feature-out": _mean_abs_local_scores(
            {
                unit_id: _lofo_scores(values, split.feature_columns)
                for unit_id, values in validation.values_by_unit.items()
            },
            split.feature_columns,
        ),
        "Greedy decision insertion": _greedy_insertion_scores(
            validation.values_by_unit,
            split.feature_columns,
        ),
        "Downstream permutation feature importance": pfi_scores,
    }
    rows: list[dict[str, Any]] = []
    for unit_id, values in test.values_by_unit.items():
        for method, scores in scores_by_method.items():
            score_vector = np.asarray(
                [float(scores[feature]) for feature in split.feature_columns],
                dtype=float,
            )
            insertion_auc = compute_decision_insertion_auc(
                score_vector,
                values,
                split.feature_columns,
            )
            rows.append(
                {
                    "dataset": "CAISO",
                    "model_id": model_id,
                    "unit_id": f"{model_id}::{unit_id}",
                    "date": unit_id,
                    "metric": "Decision insertion AUC \\uparrow",
                    "method": method,
                    "ranking_source": "validation",
                    "validation_units": len(validation.unit_ids),
                    "test_units": len(test.unit_ids),
                    "insertion_auc": insertion_auc,
                    "ranking": json.dumps(
                        _ranking_from_scores(scores, split.feature_columns)
                    ),
                    "full_decision_value": float(values[-1]),
                }
            )
    return rows


def _run_ems_model(
    *,
    run_dir: Path,
    model_id: str,
    last_n_units: int,
    permutation_seed: int,
    ems_test_max_units: int | None,
) -> list[dict[str, Any]]:
    metadata = _load_json(run_dir / "run_metadata.json")
    config = _ems_config_from_metadata(metadata, run_dir)
    player_names = tuple(str(player) for player in metadata["player_names"])
    unit_ids, values_by_unit = load_coalition_values(
        run_dir / "coalition_values.csv",
        unit_column=EMS_TIMESTAMP_COLUMN,
    )
    validation_ids = tuple(unit_ids)
    reference_scores = _reference_scores_from_shap_csv(
        run_dir / "hourly_shap.csv",
        unit_column=EMS_TIMESTAMP_COLUMN,
        unit_ids=validation_ids,
        player_names=player_names,
    )
    validation_values = {unit_id: values_by_unit[unit_id] for unit_id in validation_ids}
    pfi_scores, _ = _compute_ems_pfi_scores(
        config=config,
        run_metadata=metadata,
        selected_unit_ids=validation_ids,
        permutation_seed=permutation_seed,
    )
    scores_by_method: dict[str, Mapping[str, float]] = {
        "Post-InfoDVA": _mean_abs_local_scores(
            reference_scores["Post-InfoDVA"],
            player_names,
        ),
        "Prediction SHAP": _mean_abs_local_scores(
            reference_scores["Prediction SHAP"],
            player_names,
        ),
        "Leave-one-feature-out": _mean_abs_local_scores(
            {
                unit_id: _lofo_scores(values_by_unit[unit_id], player_names)
                for unit_id in validation_ids
            },
            player_names,
        ),
        "Greedy decision insertion": _greedy_insertion_scores(
            validation_values,
            player_names,
        ),
        "Downstream permutation feature importance": pfi_scores,
    }
    test_context = _prepare_ems_unsampled_test_context(
        config=config,
        validation_unit_ids=validation_ids,
        ems_test_max_units=ems_test_max_units,
    )
    prefix_masks_by_method = {
        method: _prefix_masks_for_scores(scores, player_names)
        for method, scores in scores_by_method.items()
    }
    required_masks = tuple(
        sorted(
            {
                mask
                for prefix_masks in prefix_masks_by_method.values()
                for mask in prefix_masks
            }
        )
    )
    rows: list[dict[str, Any]] = []
    for row_idx, (_, row) in enumerate(test_context.test_x.iterrows(), start=1):
        unit_id = str(row[EMS_TIMESTAMP_COLUMN])
        true_demand = test_context.test_y.loc[
            row_idx - 1,
            list(test_context.target_columns),
        ].to_numpy(dtype=float, copy=True)
        coalition_predictions = _predict_selected_ems_coalitions(
            predictor=test_context.coalition_predictor,
            observation=row.loc[list(test_context.feature_columns)],
            coalition_masks=required_masks,
        )
        realized_values: dict[int, float] = {}
        for coalition_mask in required_masks:
            solution = solve_ems_coverage(
                coalition_predictions[coalition_mask],
                test_context.coverage_matrix,
                test_context.zip_codes,
                facility_budget=config.facility_budget,
                solver_name=config.coverage_solver,
                name=f"ems_oos_{model_id}_{row_idx}_{coalition_mask}",
                log_to_console=False,
                solver_params=test_context.solver_params,
                optimization_solver=config.optimization_solver,
                objective_tolerance=config.objective_tolerance,
            )
            realized_values[coalition_mask] = _realized_coverage_value(
                solution.covered_zone_indices,
                true_demand,
            )
        baseline_value = realized_values[0]
        values = np.full(1 << len(player_names), np.nan, dtype=float)
        for coalition_mask, realized in realized_values.items():
            values[coalition_mask] = float(realized - baseline_value)

        for method, scores in scores_by_method.items():
            score_vector = np.asarray(
                [float(scores[player]) for player in player_names],
                dtype=float,
            )
            insertion_auc = compute_decision_insertion_auc(
                score_vector,
                values,
                player_names,
            )
            rows.append(
                {
                    "dataset": "EMS",
                    "model_id": model_id,
                    "unit_id": f"{model_id}::{unit_id}",
                    "timestamp_hour": unit_id,
                    "metric": "Decision insertion AUC \\uparrow",
                    "method": method,
                    "ranking_source": "validation",
                    "validation_units": len(validation_ids),
                    "test_units": len(test_context.test_x),
                    "insertion_auc": insertion_auc,
                    "ranking": json.dumps(_ranking_from_scores(scores, player_names)),
                    "full_decision_value": float(values[-1]),
                }
            )
        print(
            f"[{model_id} EMS test {row_idx}/{len(test_context.test_x)}] "
            f"evaluated {unit_id}",
            flush=True,
        )
    return rows


@dataclass(frozen=True, slots=True)
class _EmsUnsampledTestContext:
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    zip_codes: tuple[str, ...]
    test_x: pd.DataFrame
    test_y: pd.DataFrame
    coalition_predictor: GroupedBackgroundCoalitionPredictor
    coverage_matrix: np.ndarray
    solver_params: dict[str, Any]


def _prepare_ems_unsampled_test_context(
    *,
    config: Any,
    validation_unit_ids: Sequence[str],
    ems_test_max_units: int | None,
) -> _EmsUnsampledTestContext:
    x_frame, y_frame, metadata = _load_ems_frames(config)
    zone_order = _load_zone_order(config.zone_order_path, tuple(_target_columns(y_frame)))
    target_columns = tuple(zone_order["target_column"].astype(str))
    zip_codes = tuple(zone_order["zip_code"].astype(str))
    y_frame = y_frame.loc[:, [EMS_TIMESTAMP_COLUMN, *target_columns]].copy()
    feature_columns = tuple(_resolve_feature_columns(x_frame, metadata))
    feature_groups = tuple(build_ems_feature_groups(feature_columns))
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
    model = _fit_xgb_regressor(
        train_frame=train_x,
        y_train=train_y.loc[:, list(target_columns)],
        feature_columns=feature_columns,
        config=config,
    )
    background_frame = _sample_background_frame(
        train_frame=train_x,
        feature_columns=feature_columns,
        rows=config.background_rows,
        random_state=config.random_state,
    )
    coalition_predictor = GroupedBackgroundCoalitionPredictor(
        model=model,
        feature_names=feature_columns,
        feature_groups=feature_groups,
        background_frame=background_frame,
        coalition_batch_size=config.coalition_batch_size,
    )
    distance_matrix = _load_distance_matrix(config.distance_matrix_path)
    coverage_matrix = build_coverage_matrix(
        distance_matrix,
        zip_codes,
        coverage_radius_km=config.coverage_radius_km,
    )
    holdout_x = time_split.holdout_x.reset_index(drop=True).copy()
    holdout_y = time_split.holdout_y.reset_index(drop=True).copy()
    holdout_x[EMS_TIMESTAMP_COLUMN] = holdout_x[EMS_TIMESTAMP_COLUMN].astype(str)
    holdout_y[EMS_TIMESTAMP_COLUMN] = holdout_y[EMS_TIMESTAMP_COLUMN].astype(str)
    sampled_validation = {str(unit_id) for unit_id in validation_unit_ids}
    unsampled_mask = ~holdout_x[EMS_TIMESTAMP_COLUMN].isin(sampled_validation)
    test_x = holdout_x.loc[unsampled_mask].reset_index(drop=True)
    test_y = holdout_y.loc[unsampled_mask].reset_index(drop=True)
    if ems_test_max_units is not None:
        if ems_test_max_units <= 0:
            raise ValueError("--ems-test-max-units must be positive when provided.")
        test_x = test_x.head(ems_test_max_units).reset_index(drop=True)
        test_y = test_y.head(ems_test_max_units).reset_index(drop=True)
    if test_x.empty:
        raise ValueError("No EMS unsampled holdout units remain for test evaluation.")
    if not test_x[EMS_TIMESTAMP_COLUMN].equals(test_y[EMS_TIMESTAMP_COLUMN]):
        raise ValueError("EMS unsampled test X/y timestamp columns are not aligned.")
    return _EmsUnsampledTestContext(
        feature_columns=feature_columns,
        target_columns=target_columns,
        zip_codes=zip_codes,
        test_x=test_x,
        test_y=test_y,
        coalition_predictor=coalition_predictor,
        coverage_matrix=coverage_matrix,
        solver_params=_build_solver_params(config),
    )


def _prefix_masks_for_scores(
    scores: Mapping[str, float],
    player_names: Sequence[str],
) -> tuple[int, ...]:
    feature_index = {str(player): idx for idx, player in enumerate(player_names)}
    coalition_mask = 0
    masks = [0]
    for player in _ranking_from_scores(scores, player_names):
        coalition_mask |= 1 << feature_index[player]
        masks.append(coalition_mask)
    return tuple(masks)


def _predict_selected_ems_coalitions(
    *,
    predictor: GroupedBackgroundCoalitionPredictor,
    observation: pd.Series | Sequence[float] | np.ndarray,
    coalition_masks: Sequence[int],
) -> dict[int, np.ndarray]:
    observation_values = _ems_observation_values(observation, predictor.feature_names)
    predictions_by_mask: dict[int, np.ndarray] = {}
    masks = tuple(int(mask) for mask in coalition_masks)
    for batch_start in range(0, len(masks), predictor.coalition_batch_size):
        batch_masks = masks[batch_start : batch_start + predictor.coalition_batch_size]
        batch_values = np.tile(predictor.background_values, (len(batch_masks), 1))
        for local_idx, coalition_mask in enumerate(batch_masks):
            included_indices = predictor._included_indices_by_mask[coalition_mask]
            if included_indices.size == 0:
                continue
            row_slice = slice(
                local_idx * predictor.background_count,
                (local_idx + 1) * predictor.background_count,
            )
            batch_values[row_slice, included_indices] = observation_values[
                included_indices
            ]
        batch_predictions = np.asarray(
            predictor.model.predict(np.ascontiguousarray(batch_values, dtype=np.float32)),
            dtype=float,
        )
        if batch_predictions.ndim == 1:
            batch_predictions = batch_predictions[:, np.newaxis]
        batch_predictions = batch_predictions.reshape(
            len(batch_masks),
            predictor.background_count,
            predictor.output_count,
        )
        averaged = np.maximum(batch_predictions.mean(axis=1), 0.0)
        for local_idx, coalition_mask in enumerate(batch_masks):
            predictions_by_mask[int(coalition_mask)] = averaged[local_idx]
    return predictions_by_mask


def _ems_observation_values(
    observation: pd.Series | Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    if isinstance(observation, pd.Series):
        return observation.loc[list(feature_names)].to_numpy(dtype=float, copy=True)
    array = np.asarray(observation, dtype=float)
    if array.shape != (len(feature_names),):
        raise ValueError(
            "EMS observation must contain one value per raw feature. "
            f"Expected {len(feature_names)}, got shape {array.shape}."
        )
    return array


class _CaisoCoalitions:
    def __init__(
        self,
        *,
        unit_ids: tuple[str, ...],
        values_by_unit: dict[str, np.ndarray],
        full_values_by_unit: dict[str, float],
        decision_shap_by_unit: dict[str, dict[str, float]],
        predictive_shap_by_unit: dict[str, dict[str, float]],
    ) -> None:
        self.unit_ids = unit_ids
        self.values_by_unit = values_by_unit
        self.full_values_by_unit = full_values_by_unit
        self.decision_shap_by_unit = decision_shap_by_unit
        self.predictive_shap_by_unit = predictive_shap_by_unit


def _compute_caiso_coalitions(
    *,
    frame: pd.DataFrame,
    frame_label: str,
    model_id: str,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    date_column: str,
    evaluator: Any,
    config: Any,
    solver_params: dict[str, Any],
) -> _CaisoCoalitions:
    unit_ids: list[str] = []
    values_by_unit: dict[str, np.ndarray] = {}
    full_values_by_unit: dict[str, float] = {}
    decision_shap_by_unit: dict[str, dict[str, float]] = {}
    predictive_shap_by_unit: dict[str, dict[str, float]] = {}
    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        unit_id = str(row[date_column])
        observation = row.loc[list(feature_columns)]
        true_prices = tuple(float(value) for value in row.loc[list(target_columns)])
        coalition_predictions, coalition_realized_values, _, _ = _solve_all_coalitions_for_day(
            observation=observation,
            true_prices=true_prices,
            date=unit_id,
            evaluator=evaluator,
            actual_parameters=config.storage_parameters,
            parameter_player_spec=config.parameter_player_spec,
            solver_params=solver_params,
            objective_tolerance=config.objective_tolerance,
        )
        decision_characteristic_values = (
            coalition_realized_values - coalition_realized_values[0]
        )
        decision_shap = compute_exact_shapley_values(
            decision_characteristic_values,
            feature_count=len(feature_columns),
        )
        predictive_hourly_shap = compute_exact_shapley_values(
            coalition_predictions,
            feature_count=len(feature_columns),
        )
        predictive_daily_shap = predictive_hourly_shap.sum(axis=1)
        unit_ids.append(unit_id)
        values_by_unit[unit_id] = decision_characteristic_values
        full_values_by_unit[unit_id] = float(coalition_realized_values[-1])
        decision_shap_by_unit[unit_id] = {
            str(feature): float(decision_shap[idx])
            for idx, feature in enumerate(feature_columns)
        }
        predictive_shap_by_unit[unit_id] = {
            str(feature): float(predictive_daily_shap[idx])
            for idx, feature in enumerate(feature_columns)
        }
        print(
            f"[{model_id} {frame_label} {row_number}/{len(frame)}] computed {unit_id}",
            flush=True,
        )
    return _CaisoCoalitions(
        unit_ids=tuple(unit_ids),
        values_by_unit=values_by_unit,
        full_values_by_unit=full_values_by_unit,
        decision_shap_by_unit=decision_shap_by_unit,
        predictive_shap_by_unit=predictive_shap_by_unit,
    )


def _ranking_from_scores(
    scores: Mapping[str, float],
    player_names: Sequence[str],
) -> list[str]:
    indexed = [
        (idx, str(player), abs(float(scores[str(player)])))
        for idx, player in enumerate(player_names)
    ]
    return [player for _, player, _ in sorted(indexed, key=lambda item: (-item[2], item[0]))]


def _bootstrap_auc(local: pd.DataFrame, config: BootstrapConfig) -> pd.DataFrame:
    frame = local.loc[:, ["dataset", "unit_id", "metric", "method", "insertion_auc"]]
    frame = frame.rename(columns={"insertion_auc": "value"})
    return bootstrap_metric_table(frame, reference_method=None, config=config)


def _format_auc_table(summary: pd.DataFrame) -> str:
    methods = [
        "Post-InfoDVA",
        "Prediction SHAP",
        "Leave-one-feature-out",
        "Greedy decision insertion",
        "Downstream permutation feature importance",
    ]
    labels = {
        "Downstream permutation feature importance": "Permutation feature importance",
    }
    lookup = summary.set_index(["dataset", "method"], drop=False)
    rows = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        (
            r"\caption{Out-of-sample decision insertion AUC using "
            r"validation-learned rankings and held-out test units across 25 models. "
            r"Values are mean [95\% paired bootstrap CI].}"
        ),
        r"\label{tab:validation_ranked_test_auc}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{CAISO} & \textbf{EMS} \\",
        r"\midrule",
    ]
    best_by_dataset = {
        dataset: max(
            methods,
            key=lambda method: float(lookup.loc[(dataset, method), "mean"]),
        )
        for dataset in ("CAISO", "EMS")
    }
    for method in methods:
        values = []
        for dataset in ("CAISO", "EMS"):
            row = lookup.loc[(dataset, method)]
            value = (
                f"{float(row['mean']):.3f} "
                f"[{float(row['mean_ci_low']):.3f}, {float(row['mean_ci_high']):.3f}]"
            )
            if method == best_by_dataset[dataset]:
                value = f"\\textbf{{{value}}}"
            values.append(value)
        rows.append(f"  {labels.get(method, method)} & {values[0]} & {values[1]} \\\\")
    rows.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.25em}",
            (
                r"\footnotesize\emph{Note.} Feature rankings are learned only from "
                r"validation units. Test AUC is computed only on held-out test units "
                r"with positive full decision-value gain."
            ),
            r"\end{table}",
        ]
    )
    return "\n".join(rows) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Learn CAISO attribution rankings on validation units and evaluate "
            "decision insertion AUC on held-out test units."
        )
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
    parser.add_argument(
        "--last-n-units",
        type=int,
        default=30,
        help=(
            "Number of CAISO validation and test tail units to recompute. "
            "EMS uses all sampled holdout units as validation."
        ),
    )
    parser.add_argument(
        "--ems-test-max-units",
        type=int,
        default=None,
        help=(
            "Optional cap for EMS unsampled holdout test units. By default, "
            "all unsampled EMS holdout units are evaluated."
        ),
    )
    parser.add_argument("--permutation-seed", type=int, default=1)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    bootstrap_config = BootstrapConfig(
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        seed=args.bootstrap_seed,
    )
    run_validation_ranked_test_auc(
        caiso_root=args.caiso_root,
        evaluation_label=args.caiso_evaluation_label,
        ems_root=args.ems_root,
        ems_run_relative_path=args.ems_run_relative_path,
        last_n_units=args.last_n_units,
        permutation_seed=args.permutation_seed,
        ems_test_max_units=args.ems_test_max_units,
        bootstrap_config=bootstrap_config,
        output=args.output,
        table_output=args.table_output,
    )


if __name__ == "__main__":
    main()
