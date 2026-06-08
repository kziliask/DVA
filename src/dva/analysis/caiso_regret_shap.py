from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dva.analysis.caiso_shap import (
    BackgroundMarginalCoalitionEvaluator,
    DEFAULT_BACKGROUND_DAYS,
    _build_daily_full_dispatch_rows,
    _solve_all_coalitions_for_day,
    build_default_storage_parameters,
    compute_exact_shapley_values,
    select_recent_background_frame,
)
from dva.analysis.evaluation_metrics import (
    build_attribution_ranking,
    compute_kendall_tau_correlation,
    compute_rank_kendall_tau_from_rankings,
    compute_rank_spearman_from_rankings,
    compute_spearman_rank_correlation,
)
from dva.model.storage_dispatch import (
    StorageDispatchParameters,
    evaluate_storage_dispatch_result,
    solve_storage_dispatch_lexicographic,
)
from dva.model.train import (
    DEFAULT_DATASET_PATH,
    DEFAULT_HOLDOUT_DAYS,
    DEFAULT_MODEL_NAME,
    ModelTrainingArtifacts,
    SUPPORTED_MODEL_NAMES,
    load_default_train_explain_split,
    train_model,
)


DEFAULT_RANDOM_STATE = 0
DEFAULT_RF_N_JOBS = 1
DEFAULT_SOLVER_SEED = 0
DEFAULT_MIP_GAP = 0.0
DEFAULT_MIP_GAP_ABS = 1e-9
DEFAULT_OBJECTIVE_TOLERANCE = 1e-6
DEFAULT_OUTPUT_DIR = Path("results/caiso_regret_shap_case_study")
REGRET_TARGET_COLUMN = "actual_daily_regret"


@dataclass(frozen=True, slots=True)
class CaisoRegretShapCaseStudyConfig:
    dataset_path: Path = DEFAULT_DATASET_PATH
    holdout_days: int = DEFAULT_HOLDOUT_DAYS
    outdir: Path = DEFAULT_OUTPUT_DIR
    model_name: str = DEFAULT_MODEL_NAME
    random_state: int = DEFAULT_RANDOM_STATE
    n_jobs: int = DEFAULT_RF_N_JOBS
    solver_seed: int = DEFAULT_SOLVER_SEED
    mip_gap: float = DEFAULT_MIP_GAP
    mip_gap_abs: float = DEFAULT_MIP_GAP_ABS
    objective_tolerance: float = DEFAULT_OBJECTIVE_TOLERANCE
    max_days: int | None = None
    background_days: int = DEFAULT_BACKGROUND_DAYS
    max_train_days: int | None = None
    storage_parameters: StorageDispatchParameters = field(
        default_factory=lambda: build_default_storage_parameters()
    )


@dataclass(frozen=True, slots=True)
class CaisoRegretShapCaseStudyOutputs:
    daily_shap: pd.DataFrame
    regret_feature_shap: pd.DataFrame
    regret_predictions: pd.DataFrame
    daily_full_dispatch: pd.DataFrame
    summary_shap: pd.DataFrame
    prediction_metrics: dict[str, Any]
    comparison_metrics: dict[str, Any]
    run_metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _RegretLabelResult:
    labels: pd.DataFrame
    daily_full_dispatch: pd.DataFrame


def run_caiso_regret_shap_case_study(
    config: CaisoRegretShapCaseStudyConfig,
) -> CaisoRegretShapCaseStudyOutputs:
    if config.max_days is not None and config.max_days <= 0:
        raise ValueError("max_days must be strictly positive when provided.")
    if config.background_days <= 0:
        raise ValueError("background_days must be strictly positive.")
    if config.max_train_days is not None and config.max_train_days <= 0:
        raise ValueError("max_train_days must be strictly positive when provided.")
    if config.model_name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            "model_name must be one of: " + ", ".join(SUPPORTED_MODEL_NAMES)
        )

    split = load_default_train_explain_split(
        dataset_path=config.dataset_path,
        holdout_days=config.holdout_days,
    )
    explain_frame = split.explain_frame
    if config.max_days is not None:
        explain_frame = explain_frame.iloc[: config.max_days].reset_index(drop=True)
    if explain_frame.empty:
        raise ValueError("No explanation rows remain after applying max_days.")

    regret_train_frame = split.train_frame
    if config.max_train_days is not None:
        regret_train_frame = regret_train_frame.iloc[-config.max_train_days :].reset_index(
            drop=True
        )
    if regret_train_frame.empty:
        raise ValueError("No regret-training rows remain after applying max_train_days.")

    background_frame = select_recent_background_frame(
        split.train_frame,
        split.date_column,
        config.background_days,
    )
    regret_background_frame = select_recent_background_frame(
        regret_train_frame,
        split.date_column,
        config.background_days,
    )
    X_background = background_frame.loc[:, list(split.feature_columns)]
    X_regret_background = regret_background_frame.loc[:, list(split.feature_columns)]
    X_regret_train = regret_train_frame.loc[:, list(split.feature_columns)]
    X_explain = explain_frame.loc[:, list(split.feature_columns)]
    y_explain = explain_frame.loc[:, list(split.target_columns)]
    explain_dates = explain_frame.loc[:, split.date_column].astype(str).tolist()

    solver_params = {
        "Threads": 1,
        "Seed": config.solver_seed,
        "MIPGap": config.mip_gap,
        "MIPGapAbs": config.mip_gap_abs,
    }
    started_at = time.perf_counter()

    base_artifacts = train_model(
        split.X_train,
        split.y_train,
        model_name=config.model_name,
        feature_columns=split.feature_columns,
        target_columns=split.target_columns,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )

    train_label_result = _build_regret_labels_for_frame(
        frame=regret_train_frame,
        split_name="train",
        base_artifacts=base_artifacts,
        date_column=split.date_column,
        feature_columns=split.feature_columns,
        target_columns=split.target_columns,
        storage_parameters=config.storage_parameters,
        solver_params=solver_params,
        objective_tolerance=config.objective_tolerance,
        include_full_dispatch=False,
    )
    y_regret_train = train_label_result.labels.loc[:, [REGRET_TARGET_COLUMN]]
    regret_model_name = resolve_regret_model_name(config.model_name)
    regret_artifacts = train_model(
        X_regret_train,
        y_regret_train,
        model_name=regret_model_name,
        feature_columns=split.feature_columns,
        target_columns=(REGRET_TARGET_COLUMN,),
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )

    base_coalition_evaluator = BackgroundMarginalCoalitionEvaluator(
        base_artifacts.model,
        split.feature_columns,
        X_background,
    )
    regret_coalition_evaluator = BackgroundMarginalCoalitionEvaluator(
        regret_artifacts.model,
        split.feature_columns,
        X_regret_background,
    )

    daily_rows: list[dict[str, Any]] = []
    regret_feature_rows: list[dict[str, Any]] = []
    explain_prediction_rows: list[dict[str, Any]] = []
    daily_full_dispatch_rows: list[dict[str, Any]] = []

    for row_idx, date in enumerate(explain_dates):
        observation = X_explain.iloc[row_idx]
        true_prices = tuple(
            float(value)
            for value in y_explain.iloc[row_idx].to_numpy(dtype=float, copy=True)
        )
        regret_coalition_predictions = _as_1d_coalition_values(
            regret_coalition_evaluator.evaluate_all_coalitions(observation),
            coalition_count=regret_coalition_evaluator.coalition_count,
        )
        regret_predictive_shap = compute_exact_shapley_values(
            regret_coalition_predictions,
            feature_count=len(split.feature_columns),
        )
        (
            price_coalition_predictions,
            coalition_realized_values,
            _coalition_dispatch_results,
            full_dispatch_result,
        ) = _solve_all_coalitions_for_day(
            observation=observation,
            true_prices=true_prices,
            date=date,
            evaluator=base_coalition_evaluator,
            actual_parameters=config.storage_parameters,
            parameter_player_spec=None,
            solver_params=solver_params,
            objective_tolerance=config.objective_tolerance,
        )
        if full_dispatch_result is None:
            raise RuntimeError("Expected to capture the full-coalition dispatch result.")

        price_predictive_hourly_shap = compute_exact_shapley_values(
            price_coalition_predictions,
            feature_count=len(split.feature_columns),
        )
        price_predictive_daily_shap = price_predictive_hourly_shap.sum(axis=1)
        oracle_dispatch_result = solve_storage_dispatch_lexicographic(
            true_prices,
            config.storage_parameters,
            name=f"storage_dispatch_regret_oracle_{date}",
            log_to_console=False,
            solver_params=solver_params,
            objective_tolerance=config.objective_tolerance,
        )
        oracle_objective_value = float(oracle_dispatch_result.objective_value)
        coalition_regrets = oracle_objective_value - coalition_realized_values
        decision_characteristic_values = coalition_realized_values - coalition_realized_values[0]
        first_stage_decision_shap = compute_exact_shapley_values(
            decision_characteristic_values,
            feature_count=len(split.feature_columns),
        )
        daily_abs_rank_spearman = compute_rank_spearman_from_rankings(
            build_attribution_ranking(price_predictive_daily_shap, split.feature_columns),
            build_attribution_ranking(first_stage_decision_shap, split.feature_columns),
        )
        daily_abs_rank_kendall_tau = compute_rank_kendall_tau_from_rankings(
            build_attribution_ranking(price_predictive_daily_shap, split.feature_columns),
            build_attribution_ranking(first_stage_decision_shap, split.feature_columns),
        )
        regret_daily_abs_rank_spearman = compute_rank_spearman_from_rankings(
            build_attribution_ranking(regret_predictive_shap, split.feature_columns),
            build_attribution_ranking(first_stage_decision_shap, split.feature_columns),
        )
        regret_daily_abs_rank_kendall_tau = compute_rank_kendall_tau_from_rankings(
            build_attribution_ranking(regret_predictive_shap, split.feature_columns),
            build_attribution_ranking(first_stage_decision_shap, split.feature_columns),
        )

        predicted_daily_regret = float(regret_coalition_predictions[-1])
        baseline_predicted_regret = float(regret_coalition_predictions[0])
        price_predictive_baseline_total = float(price_coalition_predictions[0].sum())
        price_predictive_full_total = float(price_coalition_predictions[-1].sum())
        actual_daily_regret = float(coalition_regrets[-1])
        baseline_daily_regret = float(coalition_regrets[0])
        first_stage_decision_baseline_value = float(coalition_realized_values[0])
        first_stage_decision_full_value = float(coalition_realized_values[-1])
        row: dict[str, Any] = {
            "date": date,
            "predictive_baseline_total": price_predictive_baseline_total,
            "predictive_full_total": price_predictive_full_total,
            "predictive_total_gain": (
                price_predictive_full_total - price_predictive_baseline_total
            ),
            "regret_predictive_baseline_value": baseline_predicted_regret,
            "regret_predictive_full_value": predicted_daily_regret,
            "regret_predictive_value_gain": (
                predicted_daily_regret - baseline_predicted_regret
            ),
            "decision_baseline_value": first_stage_decision_baseline_value,
            "decision_full_value": first_stage_decision_full_value,
            "oracle_obj": oracle_objective_value,
            "base_decision_baseline_value": first_stage_decision_baseline_value,
            "base_decision_full_value": first_stage_decision_full_value,
            REGRET_TARGET_COLUMN: actual_daily_regret,
            "predicted_daily_regret": predicted_daily_regret,
            "baseline_daily_regret": baseline_daily_regret,
            "decision_value_gain": (
                first_stage_decision_full_value - first_stage_decision_baseline_value
            ),
            "abs_rank_spearman": daily_abs_rank_spearman,
            "abs_rank_kendall_tau": daily_abs_rank_kendall_tau,
            "regret_abs_rank_spearman": regret_daily_abs_rank_spearman,
            "regret_abs_rank_kendall_tau": regret_daily_abs_rank_kendall_tau,
        }
        for feature_name, shap_value in zip(
            split.feature_columns,
            price_predictive_daily_shap,
            strict=True,
        ):
            row[f"predictive_shap_{feature_name}"] = float(shap_value)
        for feature_name, shap_value in zip(
            split.feature_columns,
            regret_predictive_shap,
            strict=True,
        ):
            row[f"regret_predictive_shap_{feature_name}"] = float(shap_value)
            regret_feature_rows.append(
                {
                    "date": date,
                    "feature": feature_name,
                    "shap_value": float(shap_value),
                }
            )
        for feature_name, shap_value in zip(
            split.feature_columns,
            first_stage_decision_shap,
            strict=True,
        ):
            row[f"decision_shap_{feature_name}"] = float(shap_value)
        daily_rows.append(row)
        explain_prediction_rows.append(
            {
                "split": "explain",
                "date": date,
                "oracle_obj": oracle_objective_value,
                "base_decision_value": first_stage_decision_full_value,
                REGRET_TARGET_COLUMN: actual_daily_regret,
                "predicted_daily_regret": predicted_daily_regret,
            }
        )
        daily_full_dispatch_rows.extend(
            _build_daily_full_dispatch_rows(date, full_dispatch_result)
        )

    daily_shap = pd.DataFrame(daily_rows)
    regret_feature_shap = pd.DataFrame(regret_feature_rows)
    daily_full_dispatch = pd.DataFrame(daily_full_dispatch_rows)
    summary_shap = _build_summary_shap_frame(daily_shap, split.feature_columns)

    train_prediction_rows = _build_train_prediction_rows(
        train_label_result.labels,
        _predict_scalar(regret_artifacts.model, X_regret_train),
    )
    regret_predictions = pd.concat(
        [
            train_prediction_rows,
            pd.DataFrame(explain_prediction_rows),
        ],
        ignore_index=True,
    )
    y_true_explain_regret = daily_shap[REGRET_TARGET_COLUMN].to_numpy(dtype=float)
    y_pred_explain_regret = daily_shap["predicted_daily_regret"].to_numpy(dtype=float)
    y_pred_price_explain = _as_2d_predictions(base_artifacts.model.predict(X_explain))
    prediction_metrics = _build_prediction_metrics(
        y_true_train=y_regret_train[REGRET_TARGET_COLUMN].to_numpy(dtype=float),
        y_pred_train=train_prediction_rows["predicted_daily_regret"].to_numpy(dtype=float),
        y_true_explain=y_true_explain_regret,
        y_pred_explain=y_pred_explain_regret,
        y_true_price_explain=y_explain.to_numpy(dtype=float, copy=True),
        y_pred_price_explain=y_pred_price_explain,
    )
    comparison_metrics = _build_comparison_metrics(
        daily_shap=daily_shap,
        summary_shap=summary_shap,
    )
    total_runtime_seconds = time.perf_counter() - started_at
    run_metadata = {
        "experiment_type": "caiso_regret_predictor_shap",
        "dataset_path": str(config.dataset_path),
        "model_name": regret_artifacts.model_name,
        "model_description": regret_artifacts.model_description,
        "base_model_name": base_artifacts.model_name,
        "base_model_description": base_artifacts.model_description,
        "regret_model_name": regret_artifacts.model_name,
        "regret_model_description": regret_artifacts.model_description,
        "regret_model_architecture_source": config.model_name,
        "regret_target_column": REGRET_TARGET_COLUMN,
        "predictive_shap_family": "first_stage_price_model_prediction",
        "regret_predictive_shap_family": "second_stage_regret_model_prediction",
        "decision_shap_family": "first_stage_price_model_decision_value",
        "coalition_expectation_method": "empirical_background_marginalization",
        "feature_columns": list(split.feature_columns),
        "player_names": list(split.feature_columns),
        "target_columns": list(split.target_columns),
        "train_date_start": split.train_dates.iloc[0],
        "train_date_end": split.train_dates.iloc[-1],
        "background_date_start": str(background_frame[split.date_column].iloc[0]),
        "background_date_end": str(background_frame[split.date_column].iloc[-1]),
        "regret_train_date_start": str(regret_train_frame[split.date_column].iloc[0]),
        "regret_train_date_end": str(regret_train_frame[split.date_column].iloc[-1]),
        "regret_background_date_start": str(
            regret_background_frame[split.date_column].iloc[0]
        ),
        "regret_background_date_end": str(
            regret_background_frame[split.date_column].iloc[-1]
        ),
        "explain_date_start": explain_dates[0],
        "explain_date_end": explain_dates[-1],
        "train_rows": int(len(split.X_train)),
        "background_rows": int(len(X_background)),
        "regret_train_rows": int(len(X_regret_train)),
        "regret_background_rows": int(len(X_regret_background)),
        "explain_rows": int(len(X_explain)),
        "holdout_days": config.holdout_days,
        "background_days": config.background_days,
        "max_days": config.max_days,
        "max_train_days": config.max_train_days,
        "coalitions_per_day": regret_coalition_evaluator.coalition_count,
        "storage_parameters": dataclasses.asdict(config.storage_parameters),
        "random_state": config.random_state,
        "n_jobs": config.n_jobs,
        "solver_params": solver_params,
        "objective_tolerance": config.objective_tolerance,
        "runtime_seconds": total_runtime_seconds,
    }

    return CaisoRegretShapCaseStudyOutputs(
        daily_shap=daily_shap,
        regret_feature_shap=regret_feature_shap,
        regret_predictions=regret_predictions,
        daily_full_dispatch=daily_full_dispatch,
        summary_shap=summary_shap,
        prediction_metrics=prediction_metrics,
        comparison_metrics=comparison_metrics,
        run_metadata=run_metadata,
    )


def write_caiso_regret_shap_case_study_outputs(
    outputs: CaisoRegretShapCaseStudyOutputs,
    outdir: Path | str,
) -> None:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    outputs.daily_shap.to_csv(outdir_path / "daily_shap.csv", index=False)
    outputs.regret_feature_shap.to_csv(
        outdir_path / "regret_feature_shap.csv",
        index=False,
    )
    outputs.regret_predictions.to_csv(
        outdir_path / "regret_predictions.csv",
        index=False,
    )
    outputs.daily_full_dispatch.to_csv(
        outdir_path / "daily_full_dispatch.csv",
        index=False,
    )
    outputs.summary_shap.to_csv(outdir_path / "summary_shap.csv", index=False)

    with (outdir_path / "prediction_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(outputs.prediction_metrics, handle, indent=2, sort_keys=True)

    with (outdir_path / "comparison_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(outputs.comparison_metrics, handle, indent=2, sort_keys=True)

    with (outdir_path / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(outputs.run_metadata, handle, indent=2, sort_keys=True)


def resolve_regret_model_name(base_model_name: str) -> str:
    if base_model_name != "xgb":
        raise ValueError("Only model_name='xgb' is supported for regret SHAP.")
    return "xgb"


def _build_regret_labels_for_frame(
    *,
    frame: pd.DataFrame,
    split_name: str,
    base_artifacts: ModelTrainingArtifacts,
    date_column: str,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    storage_parameters: StorageDispatchParameters,
    solver_params: dict[str, Any],
    objective_tolerance: float,
    include_full_dispatch: bool,
) -> _RegretLabelResult:
    label_rows: list[dict[str, Any]] = []
    daily_full_dispatch_rows: list[dict[str, Any]] = []
    X = frame.loc[:, list(feature_columns)]
    y = frame.loc[:, list(target_columns)]
    price_predictions = _as_2d_predictions(base_artifacts.model.predict(X))

    for row_idx, (_, row) in enumerate(frame.reset_index(drop=True).iterrows()):
        date = str(row[date_column])
        true_prices = tuple(
            float(value)
            for value in y.iloc[row_idx].to_numpy(dtype=float, copy=True)
        )
        predicted_prices = tuple(float(value) for value in price_predictions[row_idx])
        base_dispatch_result = solve_storage_dispatch_lexicographic(
            predicted_prices,
            storage_parameters,
            name=f"storage_dispatch_regret_base_{split_name}_{date}",
            log_to_console=False,
            solver_params=solver_params,
            objective_tolerance=objective_tolerance,
        )
        base_evaluation = evaluate_storage_dispatch_result(
            true_prices,
            base_dispatch_result,
            storage_parameters,
        )
        oracle_dispatch_result = solve_storage_dispatch_lexicographic(
            true_prices,
            storage_parameters,
            name=f"storage_dispatch_regret_oracle_{split_name}_{date}",
            log_to_console=False,
            solver_params=solver_params,
            objective_tolerance=objective_tolerance,
        )
        oracle_objective_value = float(oracle_dispatch_result.objective_value)
        base_decision_value = float(base_evaluation.objective_value)
        label_rows.append(
            {
                "split": split_name,
                "date": date,
                "oracle_obj": oracle_objective_value,
                "base_decision_value": base_decision_value,
                REGRET_TARGET_COLUMN: oracle_objective_value - base_decision_value,
            }
        )
        if include_full_dispatch:
            daily_full_dispatch_rows.extend(
                _build_daily_full_dispatch_rows(date, base_dispatch_result)
            )

    return _RegretLabelResult(
        labels=pd.DataFrame(label_rows),
        daily_full_dispatch=pd.DataFrame(daily_full_dispatch_rows),
    )


def _build_train_prediction_rows(
    train_labels: pd.DataFrame,
    train_predictions: np.ndarray,
) -> pd.DataFrame:
    frame = train_labels.copy()
    frame["predicted_daily_regret"] = train_predictions.astype(float)
    return frame


def _build_summary_shap_frame(
    daily_shap: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    predictive_abs_means: dict[str, float] = {}
    decision_abs_means: dict[str, float] = {}

    for feature_name in feature_names:
        predictive_column = daily_shap[f"predictive_shap_{feature_name}"]
        decision_column = daily_shap[f"decision_shap_{feature_name}"]
        predictive_abs_means[feature_name] = float(predictive_column.abs().mean())
        decision_abs_means[feature_name] = float(decision_column.abs().mean())

    predictive_ranks = _descending_rank_map(predictive_abs_means)
    decision_ranks = _descending_rank_map(decision_abs_means)
    for feature_name in feature_names:
        predictive_column = daily_shap[f"predictive_shap_{feature_name}"]
        decision_column = daily_shap[f"decision_shap_{feature_name}"]
        summary_rows.append(
            {
                "feature": feature_name,
                "predictive_mean_signed_shap": float(predictive_column.mean()),
                "predictive_mean_abs_shap": predictive_abs_means[feature_name],
                "predictive_rank": predictive_ranks[feature_name],
                "decision_mean_signed_shap": float(decision_column.mean()),
                "decision_mean_abs_shap": decision_abs_means[feature_name],
                "decision_rank": decision_ranks[feature_name],
            }
        )
    return pd.DataFrame(summary_rows).sort_values("predictive_rank").reset_index(drop=True)


def _build_prediction_metrics(
    *,
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_explain: np.ndarray,
    y_pred_explain: np.ndarray,
    y_true_price_explain: np.ndarray,
    y_pred_price_explain: np.ndarray,
) -> dict[str, Any]:
    if y_true_price_explain.shape != y_pred_price_explain.shape:
        raise ValueError(
            "Price holdout y_true and y_pred must share the same shape. "
            f"Got {y_true_price_explain.shape} and {y_pred_price_explain.shape}."
        )
    price_mse = float(mean_squared_error(y_true_price_explain, y_pred_price_explain))
    return {
        "train": _scalar_regression_metrics(y_true_train, y_pred_train),
        "holdout": _scalar_regression_metrics(y_true_explain, y_pred_explain),
        "base_price_model_holdout": {
            "mae": float(mean_absolute_error(y_true_price_explain, y_pred_price_explain)),
            "mse": price_mse,
            "rmse": float(np.sqrt(price_mse)),
            "days": int(y_true_price_explain.shape[0]),
            "targets_per_day": (
                int(y_true_price_explain.shape[1])
                if y_true_price_explain.ndim > 1
                else 1
            ),
            "predictions": int(y_true_price_explain.size),
        },
    }


def _scalar_regression_metrics(
    y_true: np.ndarray | Sequence[float],
    y_pred: np.ndarray | Sequence[float],
) -> dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_array = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            "Scalar metric arrays must share the same shape. "
            f"Got {y_true_array.shape} and {y_pred_array.shape}."
        )
    mse = float(mean_squared_error(y_true_array, y_pred_array))
    return {
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "days": int(y_true_array.size),
        "predictions": int(y_true_array.size),
        "mean_actual_daily_regret": float(np.mean(y_true_array)),
        "mean_predicted_daily_regret": float(np.mean(y_pred_array)),
    }


def _build_comparison_metrics(
    *,
    daily_shap: pd.DataFrame,
    summary_shap: pd.DataFrame,
) -> dict[str, Any]:
    daily_spearman_by_date = {
        str(row["date"]): (
            None
            if pd.isna(row["abs_rank_spearman"])
            else float(row["abs_rank_spearman"])
        )
        for row in daily_shap.to_dict(orient="records")
    }
    daily_kendall_tau_by_date = {
        str(row["date"]): (
            None
            if pd.isna(row["abs_rank_kendall_tau"])
            else float(row["abs_rank_kendall_tau"])
        )
        for row in daily_shap.to_dict(orient="records")
    }
    return {
        "daily_abs_rank_spearman": _build_rank_agreement_summary(
            daily_spearman_by_date,
        ),
        "daily_abs_rank_kendall_tau": _build_rank_agreement_summary(
            daily_kendall_tau_by_date,
        ),
        "global_abs_rank_spearman": compute_spearman_rank_correlation(
            summary_shap["predictive_mean_abs_shap"].to_numpy(dtype=float),
            summary_shap["decision_mean_abs_shap"].to_numpy(dtype=float),
        ),
        "global_abs_rank_kendall_tau": compute_kendall_tau_correlation(
            summary_shap["predictive_mean_abs_shap"].to_numpy(dtype=float),
            summary_shap["decision_mean_abs_shap"].to_numpy(dtype=float),
        ),
    }


def _build_rank_agreement_summary(
    values_by_date: dict[str, float | None],
) -> dict[str, Any]:
    valid_values = [value for value in values_by_date.values() if value is not None]
    return {
        "values_by_date": values_by_date,
        "mean": float(np.mean(valid_values)) if valid_values else None,
        "median": float(np.median(valid_values)) if valid_values else None,
        "valid_days": len(valid_values),
        "total_days": len(values_by_date),
    }


def _descending_rank_map(values_by_feature: dict[str, float]) -> dict[str, int]:
    ranked = (
        pd.Series(values_by_feature, dtype=float)
        .rank(method="dense", ascending=False)
        .astype(int)
        .to_dict()
    )
    return {feature_name: int(rank) for feature_name, rank in ranked.items()}


def _as_2d_predictions(predictions: Any) -> np.ndarray:
    prediction_array = np.asarray(predictions, dtype=float)
    if prediction_array.ndim == 1:
        return prediction_array[:, np.newaxis]
    if prediction_array.ndim != 2:
        raise ValueError(
            "Model predictions must be a 1D or 2D array. "
            f"Got shape {prediction_array.shape}."
        )
    return prediction_array


def _predict_scalar(model: Any, X: pd.DataFrame) -> np.ndarray:
    predictions = _as_2d_predictions(model.predict(X))
    if predictions.shape[1] != 1:
        raise ValueError(
            "Expected a scalar regret prediction, got "
            f"{predictions.shape[1]} outputs."
        )
    return predictions[:, 0]


def _as_1d_coalition_values(
    coalition_values: Any,
    *,
    coalition_count: int,
) -> np.ndarray:
    values = np.asarray(coalition_values, dtype=float)
    if values.shape == (coalition_count,):
        return values
    if values.shape == (coalition_count, 1):
        return values[:, 0]
    raise ValueError(
        "Expected one scalar value per coalition. "
        f"Got shape {values.shape} for {coalition_count} coalitions."
    )
