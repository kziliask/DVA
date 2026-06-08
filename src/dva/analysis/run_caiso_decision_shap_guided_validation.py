from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from dva.analysis.caiso_shap import (
    BackgroundMarginalCoalitionEvaluator,
    CaisoShapCaseStudyConfig,
    build_default_storage_parameters,
    compute_exact_shapley_values,
    run_caiso_shap_case_study_with_artifacts,
    select_recent_background_frame,
    write_caiso_shap_case_study_outputs,
)
from dva.model.train import (
    DEFAULT_DATASET_PATH,
    DEFAULT_DATE_COLUMN,
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_TARGET_COLUMNS,
    load_default_training_frame,
    train_model,
)


DEFAULT_OUTDIR = Path("results/caiso_decision_shap_guided_validation")
DEFAULT_VALIDATION_DAYS = 71
DEFAULT_TEST_DAYS = 30
DEFAULT_BACKGROUND_DAYS = 365
DEFAULT_BOOTSTRAP_REPLICATES = 1000
DEFAULT_MIN_MEAN_ABS_FRACTION = 0.01
DEFAULT_CI_LEVEL = 0.95
DEFAULT_TOP_K_CANDIDATES = 3
DEFAULT_THROUGHPUT_PENALTY = 5.0
DEFAULT_RANDOM_MASK_SEED = 20260510
SAGE_RESULT_STATUS_KEY = "sage_selection_status"
LOFO_RESULT_STATUS_KEY = "lofo_selection_status"
DEFAULT_STORAGE_PARAMETER_TEMPLATE = build_default_storage_parameters()
METRIC_KEYS = (
    "mae",
    "mse",
    "rmse",
    "mean_actual_daily_regret",
    "mean_decision_value_gain",
    "mean_oracle_value",
    "mean_realized_decision_value",
    "mean_distance_from_oracle",
    "days",
    "targets_per_day",
    "predictions",
)


@dataclass(frozen=True, slots=True)
class CaisoGuidedValidationSplit:
    dataset_path: Path
    date_column: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    background_frame: pd.DataFrame
    validation_days: int
    test_days: int
    background_days: int


def _date_string(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%d")


def make_L25_OA_symbols() -> np.ndarray:
    p = 5
    rows = []
    for a in range(p):
        for b in range(p):
            rows.append(
                [
                    a,
                    b,
                    (a + b) % p,
                    (a + 2 * b) % p,
                    (a + 3 * b) % p,
                    (a + 4 * b) % p,
                ]
            )
    return np.array(rows, dtype=int)


def make_model_manifest() -> pd.DataFrame:
    oa = make_L25_OA_symbols()
    xgb_levels = {
        "n_estimators": [100, 50, 150, 250, 350],
        "max_depth": [3, 2, 4, 5, 6],
        "learning_rate": [0.05, 0.01, 0.03, 0.10, 0.15],
        "subsample": [0.90, 0.60, 0.75, 0.95, 1.00],
        "colsample_bytree": [0.90, 0.60, 0.75, 0.95, 1.00],
        "reg_lambda": [1.0, 0.3, 3.0, 10.0, 30.0],
    }
    xgb_design = pd.DataFrame(
        {
            name: [levels[symbol] for symbol in oa[:, factor_idx]]
            for factor_idx, (name, levels) in enumerate(xgb_levels.items())
        }
    )
    xgb_design.insert(0, "run", np.arange(1, len(xgb_design) + 1))
    xgb_design.insert(0, "model_id", [f"xgb_{run:03d}" for run in xgb_design["run"]])
    xgb_design.insert(1, "model_name", "xgb")
    return xgb_design


def _filter_manifest_by_model_family(
    manifest: pd.DataFrame,
    model_family: str,
) -> pd.DataFrame:
    if model_family in {"all", "xgb"}:
        return manifest.reset_index(drop=True)
    raise ValueError("model_family must be one of: all, xgb.")


def build_fixed_caiso_guided_validation_split(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    *,
    validation_days: int = DEFAULT_VALIDATION_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
    background_days: int = DEFAULT_BACKGROUND_DAYS,
    validation_max_days: int | None = None,
    test_max_days: int | None = None,
    train_months: int | None = None,
    validation_months: int | None = None,
    test_rest: bool = False,
    date_column: str = DEFAULT_DATE_COLUMN,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS,
) -> CaisoGuidedValidationSplit:
    if validation_days <= 0:
        raise ValueError("validation_days must be strictly positive.")
    if test_days <= 0:
        raise ValueError("test_days must be strictly positive.")
    if validation_max_days is not None and validation_max_days <= 0:
        raise ValueError("validation_max_days must be strictly positive when provided.")
    if test_max_days is not None and test_max_days <= 0:
        raise ValueError("test_max_days must be strictly positive when provided.")
    if train_months is not None and train_months <= 0:
        raise ValueError("train_months must be strictly positive when provided.")
    if validation_months is not None and validation_months <= 0:
        raise ValueError("validation_months must be strictly positive when provided.")
    if validation_months is not None and train_months is None:
        raise ValueError("validation_months requires train_months.")
    if test_rest and train_months is None:
        raise ValueError("test_rest requires train_months.")

    frame = load_default_training_frame(
        dataset_path,
        date_column=date_column,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )
    if train_months is not None:
        dates = pd.to_datetime(frame[date_column], errors="raise")
        train_start = pd.Timestamp(dates.min()).normalize()
        validation_start = train_start + pd.DateOffset(months=train_months)
        test_start = (
            validation_start + pd.DateOffset(months=validation_months)
            if validation_months is not None
            else validation_start + pd.Timedelta(days=validation_days)
        )
        if test_start > pd.Timestamp(dates.max()).normalize():
            raise ValueError(
                "Calendar split leaves no test rows. "
                f"Computed test_start={_date_string(test_start)} but dataset ends at "
                f"{_date_string(pd.Timestamp(dates.max()).normalize())}."
            )
        train_frame = frame.loc[dates < validation_start].reset_index(drop=True)
        validation_frame = frame.loc[
            (dates >= validation_start) & (dates < test_start)
        ].reset_index(drop=True)
        if test_rest:
            test_frame = frame.loc[dates >= test_start].reset_index(drop=True)
        else:
            test_end = test_start + pd.Timedelta(days=test_days)
            test_frame = frame.loc[
                (dates >= test_start) & (dates < test_end)
            ].reset_index(drop=True)
        if train_frame.empty or validation_frame.empty or test_frame.empty:
            raise ValueError(
                "Calendar split must produce nonempty train, validation, and test frames."
            )
        if validation_max_days is not None:
            validation_frame = validation_frame.iloc[:validation_max_days].reset_index(
                drop=True
            )
        if test_max_days is not None:
            test_frame = test_frame.iloc[:test_max_days].reset_index(drop=True)
        background_frame = select_recent_background_frame(
            train_frame,
            date_column,
            background_days,
        )
        return CaisoGuidedValidationSplit(
            dataset_path=Path(dataset_path),
            date_column=date_column,
            feature_columns=tuple(feature_columns),
            target_columns=tuple(target_columns),
            train_frame=train_frame,
            validation_frame=validation_frame,
            test_frame=test_frame,
            background_frame=background_frame,
            validation_days=len(validation_frame),
            test_days=len(test_frame),
            background_days=background_days,
        )

    holdout_days = validation_days + test_days
    if holdout_days >= len(frame):
        raise ValueError("validation_days + test_days must leave at least one train row.")

    train_frame = frame.iloc[:-holdout_days].reset_index(drop=True)
    validation_frame = frame.iloc[-holdout_days:-test_days].reset_index(drop=True)
    test_frame = frame.iloc[-test_days:].reset_index(drop=True)
    if validation_max_days is not None:
        validation_frame = validation_frame.iloc[:validation_max_days].reset_index(
            drop=True
        )
    if test_max_days is not None:
        test_frame = test_frame.iloc[:test_max_days].reset_index(drop=True)
    background_frame = select_recent_background_frame(
        train_frame,
        date_column,
        background_days,
    )
    return CaisoGuidedValidationSplit(
        dataset_path=Path(dataset_path),
        date_column=date_column,
        feature_columns=tuple(feature_columns),
        target_columns=tuple(target_columns),
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        background_frame=background_frame,
        validation_days=validation_days,
        test_days=test_days,
        background_days=background_days,
    )


def compute_background_feature_replacements(
    background_frame: pd.DataFrame,
    features: Sequence[str],
) -> dict[str, float]:
    return {
        feature_name: float(background_frame.loc[:, feature_name].mean())
        for feature_name in features
    }


def select_harmful_feature_candidates(
    daily_shap: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    random_state: int = 0,
    min_mean_abs_fraction: float = DEFAULT_MIN_MEAN_ABS_FRACTION,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> pd.DataFrame:
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative.")
    if not 0.0 <= min_mean_abs_fraction <= 1.0:
        raise ValueError("min_mean_abs_fraction must be in [0, 1].")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1).")

    feature_columns = tuple(feature_columns)
    mean_abs_values = {
        feature_name: float(
            daily_shap[f"decision_shap_{feature_name}"].abs().mean()
        )
        for feature_name in feature_columns
    }
    mean_abs_threshold = min_mean_abs_fraction * max(
        mean_abs_values.values(),
        default=0.0,
    )
    alpha = (1.0 - ci_level) / 2.0
    rows: list[dict[str, Any]] = []
    for feature_idx, feature_name in enumerate(feature_columns):
        values = daily_shap[f"decision_shap_{feature_name}"].to_numpy(dtype=float)
        mean_signed = float(np.mean(values))
        mean_abs = mean_abs_values[feature_name]
        if bootstrap_replicates > 0:
            rng = np.random.default_rng(random_state + feature_idx * 1009)
            sample_indices = rng.integers(
                0,
                len(values),
                size=(bootstrap_replicates, len(values)),
            )
            bootstrap_means = values[sample_indices].mean(axis=1)
            ci_lower = float(np.quantile(bootstrap_means, alpha))
            ci_upper = float(np.quantile(bootstrap_means, 1.0 - alpha))
        else:
            ci_lower = mean_signed
            ci_upper = mean_signed

        reasons = []
        if mean_signed >= 0.0:
            reasons.append("non_negative_mean_decision_shap")
        if mean_abs < mean_abs_threshold:
            reasons.append("tiny_mean_abs_decision_shap")
        if ci_upper >= 0.0:
            reasons.append("bootstrap_upper_ci_not_negative")
        rows.append(
            {
                "feature": feature_name,
                "mean_decision_shap": mean_signed,
                "mean_abs_decision_shap": mean_abs,
                "mean_abs_threshold": mean_abs_threshold,
                "bootstrap_ci_lower": ci_lower,
                "bootstrap_ci_upper": ci_upper,
                "eligible": not reasons,
                "ineligibility_reason": ";".join(reasons),
                "ineligibility_reason_count": len(reasons),
                "candidate_rank": None,
            }
        )

    candidates = pd.DataFrame(rows)
    eligible = candidates.loc[candidates["eligible"]].sort_values(
        ["mean_abs_decision_shap", "feature"],
        ascending=[False, True],
    )
    for rank, row_index in enumerate(eligible.index, start=1):
        candidates.loc[row_index, "candidate_rank"] = rank
    return candidates.sort_values(
        ["eligible", "candidate_rank", "feature"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def compute_daily_sage_values(
    *,
    model: Any,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    background_frame: pd.DataFrame,
    explain_frame: pd.DataFrame,
    date_column: str,
) -> pd.DataFrame:
    """Compute exact marginal-imputation SAGE contributions for each day."""
    feature_columns = tuple(feature_columns)
    target_columns = tuple(target_columns)
    if not feature_columns:
        raise ValueError("feature_columns must contain at least one feature.")
    if not target_columns:
        raise ValueError("target_columns must contain at least one target.")
    if background_frame.empty:
        raise ValueError("background_frame must contain at least one row.")
    if explain_frame.empty:
        raise ValueError("explain_frame must contain at least one row.")

    required_columns = {date_column, *feature_columns, *target_columns}
    missing_explain_columns = sorted(required_columns - set(explain_frame.columns))
    if missing_explain_columns:
        raise KeyError(
            "explain_frame is missing required column(s): "
            + ", ".join(missing_explain_columns)
        )
    missing_background_columns = sorted(
        set(feature_columns) - set(background_frame.columns)
    )
    if missing_background_columns:
        raise KeyError(
            "background_frame is missing required feature column(s): "
            + ", ".join(missing_background_columns)
        )

    evaluator = BackgroundMarginalCoalitionEvaluator(
        model,
        feature_columns,
        background_frame.loc[:, list(feature_columns)],
    )
    feature_count = len(feature_columns)
    full_mask = (1 << feature_count) - 1
    rows: list[dict[str, Any]] = []
    for row_idx, date in enumerate(explain_frame.loc[:, date_column].tolist()):
        observation = explain_frame.loc[:, list(feature_columns)].iloc[row_idx]
        y_true = explain_frame.loc[:, list(target_columns)].iloc[row_idx].to_numpy(
            dtype=float,
            copy=True,
        )
        coalition_predictions = np.asarray(
            evaluator.evaluate_all_coalitions(observation),
            dtype=float,
        )
        if coalition_predictions.ndim == 1:
            coalition_predictions = coalition_predictions[:, np.newaxis]
        if coalition_predictions.shape[1] != len(target_columns):
            raise ValueError(
                "model prediction width must match target_columns. "
                f"Expected {len(target_columns)}, got {coalition_predictions.shape[1]}."
            )

        coalition_losses = np.mean(
            (coalition_predictions - y_true[np.newaxis, :]) ** 2,
            axis=1,
        )
        sage_characteristic_values = coalition_losses[0] - coalition_losses
        sage_values = compute_exact_shapley_values(
            sage_characteristic_values,
            feature_count=feature_count,
        )
        row: dict[str, Any] = {
            "date": date,
            "sage_loss_null": float(coalition_losses[0]),
            "sage_loss_full": float(coalition_losses[full_mask]),
            "sage_value_full": float(sage_characteristic_values[full_mask]),
            "sage_efficiency_gap": float(
                np.sum(sage_values) - sage_characteristic_values[full_mask]
            ),
        }
        for feature_name, sage_value in zip(feature_columns, sage_values, strict=True):
            row[f"sage_shap_{feature_name}"] = float(sage_value)
        rows.append(row)
    return pd.DataFrame(rows)


def select_sage_disruptive_feature_candidates(
    daily_sage: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    random_state: int = 0,
    min_mean_abs_fraction: float = DEFAULT_MIN_MEAN_ABS_FRACTION,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> pd.DataFrame:
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative.")
    if not 0.0 <= min_mean_abs_fraction <= 1.0:
        raise ValueError("min_mean_abs_fraction must be in [0, 1].")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1).")

    feature_columns = tuple(feature_columns)
    if daily_sage.empty:
        raise ValueError("daily_sage must contain at least one row.")
    missing_columns = sorted(
        f"sage_shap_{feature_name}"
        for feature_name in feature_columns
        if f"sage_shap_{feature_name}" not in daily_sage.columns
    )
    if missing_columns:
        raise KeyError(
            "daily_sage is missing required SAGE column(s): "
            + ", ".join(missing_columns)
        )

    mean_abs_values = {
        feature_name: float(daily_sage[f"sage_shap_{feature_name}"].abs().mean())
        for feature_name in feature_columns
    }
    mean_abs_threshold = min_mean_abs_fraction * max(
        mean_abs_values.values(),
        default=0.0,
    )
    alpha = (1.0 - ci_level) / 2.0
    rows: list[dict[str, Any]] = []
    for feature_idx, feature_name in enumerate(feature_columns):
        values = daily_sage[f"sage_shap_{feature_name}"].to_numpy(dtype=float)
        mean_signed = float(np.mean(values))
        mean_abs = mean_abs_values[feature_name]
        if bootstrap_replicates > 0:
            rng = np.random.default_rng(random_state + feature_idx * 1009)
            sample_indices = rng.integers(
                0,
                len(values),
                size=(bootstrap_replicates, len(values)),
            )
            bootstrap_means = values[sample_indices].mean(axis=1)
            ci_lower = float(np.quantile(bootstrap_means, alpha))
            ci_upper = float(np.quantile(bootstrap_means, 1.0 - alpha))
        else:
            ci_lower = mean_signed
            ci_upper = mean_signed

        mean_is_negative = mean_signed < 0.0
        mean_abs_is_not_tiny = mean_abs >= mean_abs_threshold
        upper_ci_is_negative = ci_upper < 0.0
        failed_conditions = []
        if not mean_is_negative:
            failed_conditions.append("non_negative_mean_sage")
        if not mean_abs_is_not_tiny:
            failed_conditions.append("tiny_mean_abs_sage")
        if not upper_ci_is_negative:
            failed_conditions.append("bootstrap_upper_ci_not_negative")
        rows.append(
            {
                "feature": feature_name,
                "mean_sage": mean_signed,
                "mean_abs_sage": mean_abs,
                "mean_abs_threshold": mean_abs_threshold,
                "bootstrap_ci_lower": ci_lower,
                "bootstrap_ci_upper": ci_upper,
                "mean_sage_negative": mean_is_negative,
                "mean_abs_sage_not_tiny": mean_abs_is_not_tiny,
                "bootstrap_upper_ci_negative": upper_ci_is_negative,
                "disruptive": upper_ci_is_negative,
                "failed_conditions": ";".join(failed_conditions),
                "failed_condition_count": len(failed_conditions),
                "candidate_rank": None,
            }
        )

    candidates = pd.DataFrame(rows)
    disruptive = candidates.loc[candidates["disruptive"]].sort_values(
        ["mean_sage", "feature"],
        ascending=[True, True],
    )
    for rank, row_index in enumerate(disruptive.index, start=1):
        candidates.loc[row_index, "candidate_rank"] = rank
    return candidates.sort_values(
        ["disruptive", "candidate_rank", "feature"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def select_random_ineligible_feature(
    candidates: pd.DataFrame,
    *,
    random_mask_seed: int = DEFAULT_RANDOM_MASK_SEED,
    model_id: str = "",
) -> str | None:
    ineligible = candidates.loc[~candidates["eligible"]].copy()
    if ineligible.empty:
        return None
    if "ineligibility_reason_count" not in ineligible.columns:
        ineligible["ineligibility_reason_count"] = ineligible[
            "ineligibility_reason"
        ].map(_count_ineligibility_reasons)
    max_reason_count = int(ineligible["ineligibility_reason_count"].max())
    priority_features = sorted(
        str(feature_name)
        for feature_name in ineligible.loc[
            ineligible["ineligibility_reason_count"] == max_reason_count,
            "feature",
        ].tolist()
    )
    rng = np.random.default_rng(
        int(random_mask_seed) + _stable_model_seed_offset(model_id)
    )
    return str(rng.choice(priority_features))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CAISO decision-SHAP guided validation feature hiding sweep.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--model-id", action="append", default=None)
    parser.add_argument(
        "--model-family",
        choices=("all", "xgb"),
        default="all",
        help="Restrict the manifest to the XGBoost family.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--train-months",
        type=int,
        default=None,
        help=(
            "Use a calendar split starting at the first dataset date with this many "
            "months for training. When omitted, the legacy end-holdout split is used."
        ),
    )
    parser.add_argument(
        "--validation-months",
        type=int,
        default=None,
        help=(
            "Use this many calendar months for validation after --train-months. "
            "When omitted with --train-months, --validation-days is used instead."
        ),
    )
    parser.add_argument(
        "--validation-days",
        type=int,
        default=DEFAULT_VALIDATION_DAYS,
        help="Validation days for the legacy split, or for train-month splits without validation-months.",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=DEFAULT_TEST_DAYS,
        help="Test days for legacy splits and train-month splits without --test-rest.",
    )
    parser.add_argument(
        "--test-rest",
        action="store_true",
        help="With --train-months, use all remaining rows after validation as test.",
    )
    parser.add_argument("--validation-max-days", type=int, default=None)
    parser.add_argument("--test-max-days", type=int, default=None)
    parser.add_argument("--background-days", type=int, default=DEFAULT_BACKGROUND_DAYS)
    parser.add_argument(
        "--throughput-penalty",
        type=float,
        default=DEFAULT_THROUGHPUT_PENALTY,
    )
    parser.add_argument(
        "--energy-capacity",
        type=float,
        default=DEFAULT_STORAGE_PARAMETER_TEMPLATE.energy_capacity,
    )
    parser.add_argument(
        "--power-limit",
        type=float,
        default=DEFAULT_STORAGE_PARAMETER_TEMPLATE.power_limit,
    )
    parser.add_argument(
        "--charge-efficiency",
        type=float,
        default=DEFAULT_STORAGE_PARAMETER_TEMPLATE.charge_efficiency,
    )
    parser.add_argument(
        "--discharge-efficiency",
        type=float,
        default=DEFAULT_STORAGE_PARAMETER_TEMPLATE.discharge_efficiency,
    )
    parser.add_argument(
        "--initial-state-of-charge",
        type=float,
        default=DEFAULT_STORAGE_PARAMETER_TEMPLATE.initial_state_of_charge,
    )
    parser.add_argument(
        "--terminal-state-of-charge",
        type=float,
        default=DEFAULT_STORAGE_PARAMETER_TEMPLATE.terminal_state_of_charge,
    )
    parser.add_argument("--random-mask-seed", type=int, default=DEFAULT_RANDOM_MASK_SEED)
    parser.add_argument("--training-verbose", action="store_true")
    parser.add_argument(
        "--skip-sage-ablation",
        action="store_true",
        help="Do not compute the SAGE-guided feature-hiding add-on.",
    )
    parser.add_argument(
        "--compute-ante-infodva",
        action="store_true",
        help=(
            "Compute ante-InfoDVA metrics, implemented as the full-prediction "
            "ex-ante decision-value attribution family."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_workers <= 0:
        raise ValueError("max_workers must be strictly positive.")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = make_model_manifest()
    manifest.to_csv(outdir / "model_manifest.csv", index=False)
    manifest = _filter_manifest_by_model_family(manifest, args.model_family)

    selected_model_ids = set(args.model_id or [])
    if selected_model_ids:
        manifest_model_ids = set(manifest["model_id"])
        unknown_model_ids = sorted(selected_model_ids - manifest_model_ids)
        if unknown_model_ids:
            raise ValueError("Unknown model_id values: " + ", ".join(unknown_model_ids))
        manifest = manifest.loc[manifest["model_id"].isin(selected_model_ids)].reset_index(
            drop=True
        )

    split = build_fixed_caiso_guided_validation_split(
        args.dataset_path,
        validation_days=args.validation_days,
        test_days=args.test_days,
        background_days=args.background_days,
        validation_max_days=args.validation_max_days,
        test_max_days=args.test_max_days,
        train_months=args.train_months,
        validation_months=args.validation_months,
        test_rest=args.test_rest,
    )
    _write_json(
        outdir / "experiment_metadata.json",
        _build_experiment_metadata(args, split, len(manifest)),
    )

    worker_args = {
        "dataset_path": str(args.dataset_path),
        "outdir": str(outdir),
        "overwrite": bool(args.overwrite),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "validation_days": int(args.validation_days),
        "test_days": int(args.test_days),
        "train_months": args.train_months,
        "validation_months": args.validation_months,
        "test_rest": bool(args.test_rest),
        "validation_max_days": args.validation_max_days,
        "test_max_days": args.test_max_days,
        "background_days": int(args.background_days),
        "throughput_penalty": float(args.throughput_penalty),
        "energy_capacity": float(args.energy_capacity),
        "power_limit": float(args.power_limit),
        "charge_efficiency": float(args.charge_efficiency),
        "discharge_efficiency": float(args.discharge_efficiency),
        "initial_state_of_charge": float(args.initial_state_of_charge),
        "terminal_state_of_charge": float(args.terminal_state_of_charge),
        "random_mask_seed": int(args.random_mask_seed),
        "training_verbose": bool(args.training_verbose),
        "run_sage_ablation": not bool(args.skip_sage_ablation),
        "compute_ante_infodva": bool(args.compute_ante_infodva),
    }
    records = manifest.to_dict(orient="records")
    run_model_ids = {str(record["model_id"]) for record in records}
    results = _load_preserved_results(
        outdir / "model_results.csv",
        excluded_model_ids=run_model_ids,
    )
    if args.max_workers == 1:
        for index, record in enumerate(records, start=1):
            print(f"[{index}/{len(records)}] Running {record['model_id']}", flush=True)
            result = _run_single_model(record, worker_args)
            results.append(result)
            _write_results_csv(outdir / "model_results.csv", results)
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_record = {
                executor.submit(_run_single_model, record, worker_args): record
                for record in records
            }
            for future in as_completed(future_to_record):
                record = future_to_record[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive process boundary.
                    result = {
                        "model_id": record["model_id"],
                        "model_name": record["model_name"],
                        "status": "failed",
                        "error": repr(exc),
                    }
                results.append(result)
                completed += 1
                _write_results_csv(outdir / "model_results.csv", results)
                print(
                    f"[{completed}/{len(records)}] Finished {record['model_id']} "
                    f"with status={result.get('status')}",
                    flush=True,
                )

    print(f"Wrote guided validation sweep outputs to {outdir}", flush=True)


def _run_single_model(
    record: dict[str, Any],
    args: dict[str, Any],
) -> dict[str, Any]:
    model_id = str(record["model_id"])
    outdir = Path(args["outdir"])
    model_dir = outdir / "models" / model_id
    model_result_path = model_dir / "model_result.json"
    if model_result_path.exists() and not args["overwrite"]:
        result = _load_json(model_result_path)
        if result.get("status") != "failed":
            if args.get("run_sage_ablation", True) and not _result_has_sage_ablation(
                result
            ):
                return _append_sage_ablation_to_existing_model(record, args, result)
            if _result_has_lofo_ablation(result):
                result["status"] = "skipped_existing"
                return result
    if model_dir.exists() and args["overwrite"]:
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    try:
        split = build_fixed_caiso_guided_validation_split(
            args["dataset_path"],
            validation_days=int(args["validation_days"]),
            test_days=int(args["test_days"]),
            background_days=int(args["background_days"]),
            validation_max_days=args["validation_max_days"],
            test_max_days=args["test_max_days"],
            train_months=args["train_months"],
            validation_months=args["validation_months"],
            test_rest=bool(args["test_rest"]),
        )
        config = _build_config_for_record(
            record,
            dataset_path=Path(args["dataset_path"]),
            model_dir=model_dir,
            background_days=split.background_days,
            throughput_penalty=float(args["throughput_penalty"]),
            energy_capacity=float(args["energy_capacity"]),
            power_limit=float(args["power_limit"]),
            charge_efficiency=float(args["charge_efficiency"]),
            discharge_efficiency=float(args["discharge_efficiency"]),
            initial_state_of_charge=float(args["initial_state_of_charge"]),
            terminal_state_of_charge=float(args["terminal_state_of_charge"]),
            training_verbose=bool(args["training_verbose"]),
            compute_ante_infodva=bool(args.get("compute_ante_infodva", False)),
        )
        _write_json(model_dir / "model_config.json", _model_config_payload(record, config))

        training_artifacts = train_model(
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
            learning_rate=config.learning_rate,
            storage_parameters=config.storage_parameters,
            training_verbose=config.training_verbose,
            training_log_every=config.training_log_every,
            xgb_n_estimators=config.xgb_n_estimators,
            xgb_max_depth=config.xgb_max_depth,
            xgb_learning_rate=config.xgb_learning_rate,
            xgb_subsample=config.xgb_subsample,
            xgb_colsample_bytree=config.xgb_colsample_bytree,
            xgb_reg_lambda=config.xgb_reg_lambda,
            xgb_verbosity=config.xgb_verbosity,
        )

        validation_feature_mask_cache: dict[str, Any] = {}
        test_feature_mask_cache: dict[str, Any] = {}
        validation_baseline = _evaluate_split(
            config=config,
            training_artifacts=training_artifacts,
            split=split,
            explain_frame=split.validation_frame,
            outdir=model_dir / "validation_baseline",
            label="validation_baseline",
        )
        candidate_table = select_harmful_feature_candidates(
            validation_baseline.daily_shap,
            split.feature_columns,
            bootstrap_replicates=int(args["bootstrap_replicates"]),
            random_state=int(config.random_state or 0),
        )
        candidate_table.to_csv(model_dir / "candidate_features.csv", index=False)
        eligible_candidates = candidate_table.loc[candidate_table["eligible"]].head(
            DEFAULT_TOP_K_CANDIDATES
        )
        random_feature = select_random_ineligible_feature(
            candidate_table,
            random_mask_seed=int(args["random_mask_seed"]),
            model_id=model_id,
        )
        random_feature_row = (
            None
            if random_feature is None
            else candidate_table.loc[candidate_table["feature"] == random_feature].iloc[0]
        )
        random_validation_outputs = None
        random_test_outputs = None
        if random_feature is not None:
            random_validation_outputs = _evaluate_split(
                config=config,
                training_artifacts=training_artifacts,
                split=split,
                explain_frame=split.validation_frame,
                outdir=model_dir / "validation_random_ineligible_hide",
                label=f"validation_random_hide_{random_feature}",
                holdout_mean_impute_features=(random_feature,),
            )
            validation_feature_mask_cache[random_feature] = random_validation_outputs

        validation_candidate_summaries = []
        for _, candidate in eligible_candidates.iterrows():
            feature_name = str(candidate["feature"])
            candidate_rank = int(candidate["candidate_rank"])
            candidate_outputs = _evaluate_split(
                config=config,
                training_artifacts=training_artifacts,
                split=split,
                explain_frame=split.validation_frame,
                outdir=(
                    model_dir
                    / "validation_candidates"
                    / f"{candidate_rank:02d}_{_slugify(feature_name)}"
                ),
                label=f"validation_hide_{feature_name}",
                holdout_mean_impute_features=(feature_name,),
            )
            validation_feature_mask_cache[feature_name] = candidate_outputs
            validation_candidate_summaries.append(
                {
                    "feature": feature_name,
                    "candidate_rank": candidate_rank,
                    "replacement_value": candidate_outputs.run_metadata[
                        "holdout_feature_replacements"
                    ][feature_name],
                    **_prefixed_metrics("validation_candidate", candidate_outputs),
                    "validation_regret_improvement": _regret_improvement(
                        validation_baseline,
                        candidate_outputs,
                    ),
                }
            )

        test_baseline = _evaluate_split(
            config=config,
            training_artifacts=training_artifacts,
            split=split,
            explain_frame=split.test_frame,
            outdir=model_dir / "test_baseline",
            label="test_baseline",
        )
        if random_feature is not None:
            random_test_outputs = _evaluate_split(
                config=config,
                training_artifacts=training_artifacts,
                split=split,
                explain_frame=split.test_frame,
                outdir=model_dir / "test_random_ineligible_hide",
                label=f"test_random_hide_{random_feature}",
                holdout_mean_impute_features=(random_feature,),
            )
            test_feature_mask_cache[random_feature] = random_test_outputs

        selected_feature = None
        selected_validation_outputs = None
        selected_test_outputs = None
        if validation_candidate_summaries:
            selected_summary = max(
                validation_candidate_summaries,
                key=lambda row: (
                    row["validation_regret_improvement"],
                    -row["candidate_rank"],
                ),
            )
            selected_feature = str(selected_summary["feature"])
            selected_validation_outputs = _load_outputs_metrics_from_dir(
                model_dir
                / "validation_candidates"
                / f"{int(selected_summary['candidate_rank']):02d}_{_slugify(selected_feature)}"
            )
            selected_test_outputs = _evaluate_split(
                config=config,
                training_artifacts=training_artifacts,
                split=split,
                explain_frame=split.test_frame,
                outdir=model_dir / "test_selected_hide",
                label=f"test_hide_{selected_feature}",
                holdout_mean_impute_features=(selected_feature,),
            )
            test_feature_mask_cache[selected_feature] = selected_test_outputs

        selection_status = (
            "no_eligible_feature"
            if not validation_candidate_summaries
            else (
                "fewer_than_three_eligible_features"
                if len(validation_candidate_summaries) < DEFAULT_TOP_K_CANDIDATES
                else "selected"
            )
        )
        summary = {
            "model_id": model_id,
            "selection_status": selection_status,
            "eligible_feature_count": int(candidate_table["eligible"].sum()),
            "ineligible_feature_count": int((~candidate_table["eligible"]).sum()),
            "evaluated_candidate_count": len(validation_candidate_summaries),
            "selected_feature": selected_feature,
            "validation_candidates": validation_candidate_summaries,
            "validation_replacement_values": compute_background_feature_replacements(
                split.background_frame,
                [row["feature"] for row in validation_candidate_summaries],
            ),
            "random_mask_comparator": _build_random_comparator_summary(
                random_feature=random_feature,
                random_feature_row=random_feature_row,
                random_mask_seed=int(args["random_mask_seed"]),
                model_id=model_id,
                split=split,
                validation_baseline=validation_baseline,
                random_validation_outputs=random_validation_outputs,
                test_baseline=test_baseline,
                random_test_outputs=random_test_outputs,
            ),
        }
        if selected_feature is not None and selected_test_outputs is not None:
            summary["selected_test_replacement_value"] = selected_test_outputs.run_metadata[
                "holdout_feature_replacements"
            ][selected_feature]
            summary["test_regret_improvement"] = _regret_improvement(
                test_baseline,
                selected_test_outputs,
            )

        result = {
            "model_id": model_id,
            "model_name": record["model_name"],
            "status": "completed",
            "selection_status": selection_status,
            "eligible_feature_count": int(candidate_table["eligible"].sum()),
            "ineligible_feature_count": int((~candidate_table["eligible"]).sum()),
            "evaluated_candidate_count": len(validation_candidate_summaries),
            "selected_feature": selected_feature,
            "random_mask_seed": int(args["random_mask_seed"]),
            "random_feature": random_feature,
            "random_feature_ineligibility_reason": (
                None
                if random_feature_row is None
                else str(random_feature_row["ineligibility_reason"])
            ),
            "random_feature_ineligibility_reason_count": (
                None
                if random_feature_row is None
                else int(random_feature_row["ineligibility_reason_count"])
            ),
            "runtime_seconds": time.perf_counter() - started_at,
            **_flat_hyperparameters(record),
            **_prefixed_metrics("validation_baseline", validation_baseline),
            **_prefixed_metrics("test_baseline", test_baseline),
        }
        if random_validation_outputs is not None:
            result.update(
                _prefixed_metrics("validation_random_hidden", random_validation_outputs)
            )
            result.update(
                _prefixed_metric_deltas(
                    "validation_random",
                    validation_baseline,
                    random_validation_outputs,
                )
            )
            result["validation_random_regret_improvement"] = _regret_improvement(
                validation_baseline,
                random_validation_outputs,
            )
        if random_test_outputs is not None:
            result.update(_prefixed_metrics("test_random_hidden", random_test_outputs))
            result.update(
                _prefixed_metric_deltas(
                    "test_random",
                    test_baseline,
                    random_test_outputs,
                )
            )
            result["test_random_regret_improvement"] = _regret_improvement(
                test_baseline,
                random_test_outputs,
            )
        if selected_validation_outputs is not None:
            result.update(selected_validation_outputs)
            if selected_feature is not None:
                selected_validation_delta_outputs = _load_metrics_from_dir(
                    model_dir
                    / "validation_candidates"
                    / f"{int(selected_summary['candidate_rank']):02d}_{_slugify(selected_feature)}"
                )
                result.update(
                    _prefixed_metric_deltas(
                        "validation_guided",
                        validation_baseline,
                        selected_validation_delta_outputs,
                    )
                )
            result["validation_regret_improvement"] = max(
                row["validation_regret_improvement"]
                for row in validation_candidate_summaries
                if row["feature"] == selected_feature
            )
            if "validation_random_regret_improvement" in result:
                result["validation_guided_minus_random_regret_improvement"] = (
                    result["validation_regret_improvement"]
                    - result["validation_random_regret_improvement"]
                )
        if selected_test_outputs is not None:
            result.update(_prefixed_metrics("test_hidden", selected_test_outputs))
            result.update(
                _prefixed_metric_deltas(
                    "test_guided",
                    test_baseline,
                    selected_test_outputs,
                )
            )
            result["test_regret_improvement"] = _regret_improvement(
                test_baseline,
                selected_test_outputs,
            )
            if "test_random_regret_improvement" in result:
                result["test_guided_minus_random_regret_improvement"] = (
                    result["test_regret_improvement"]
                    - result["test_random_regret_improvement"]
                )
                result["test_guided_outperformed_random_regret"] = (
                    result["test_guided_minus_random_regret_improvement"] > 0.0
                )
        if args.get("run_sage_ablation", True):
            sage_payload = _run_sage_ablation(
                model_id=model_id,
                split=split,
                config=config,
                training_artifacts=training_artifacts,
                model_dir=model_dir,
                bootstrap_replicates=int(args["bootstrap_replicates"]),
                validation_baseline=validation_baseline,
                test_baseline=test_baseline,
                overwrite=bool(args["overwrite"]),
                existing_result=result,
                validation_feature_mask_cache=validation_feature_mask_cache,
                test_feature_mask_cache=test_feature_mask_cache,
            )
            summary["sage_ablation"] = sage_payload["summary"]
            result.update(sage_payload["result"])
            _add_sage_relative_comparisons(result)
        lofo_payload = _run_leave_one_feature_out_ablation(
            model_id=model_id,
            split=split,
            config=config,
            training_artifacts=training_artifacts,
            model_dir=model_dir,
            validation_baseline=validation_baseline,
            test_baseline=test_baseline,
            overwrite=bool(args["overwrite"]),
            existing_result=result,
            validation_feature_mask_cache=validation_feature_mask_cache,
            test_feature_mask_cache=test_feature_mask_cache,
        )
        summary["lofo_ablation"] = lofo_payload["summary"]
        result.update(lofo_payload["result"])
        _add_lofo_relative_comparisons(result)
        result["runtime_seconds"] = time.perf_counter() - started_at
        _write_json(model_dir / "selected_feature_summary.json", summary)
        _write_json(model_result_path, result)
        return result
    except Exception as exc:
        error_payload = {
            "model_id": model_id,
            "model_name": record.get("model_name"),
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - started_at,
        }
        _write_json(model_dir / "error.json", error_payload)
        _write_json(model_result_path, error_payload)
        return error_payload


def _append_sage_ablation_to_existing_model(
    record: dict[str, Any],
    args: dict[str, Any],
    existing_result: dict[str, Any],
) -> dict[str, Any]:
    model_id = str(record["model_id"])
    outdir = Path(args["outdir"])
    model_dir = outdir / "models" / model_id
    model_result_path = model_dir / "model_result.json"
    started_at = time.perf_counter()
    try:
        split = build_fixed_caiso_guided_validation_split(
            args["dataset_path"],
            validation_days=int(args["validation_days"]),
            test_days=int(args["test_days"]),
            background_days=int(args["background_days"]),
            validation_max_days=args["validation_max_days"],
            test_max_days=args["test_max_days"],
            train_months=args["train_months"],
            validation_months=args["validation_months"],
            test_rest=bool(args["test_rest"]),
        )
        config = _build_config_for_record(
            record,
            dataset_path=Path(args["dataset_path"]),
            model_dir=model_dir,
            background_days=split.background_days,
            throughput_penalty=float(args["throughput_penalty"]),
            energy_capacity=float(args["energy_capacity"]),
            power_limit=float(args["power_limit"]),
            charge_efficiency=float(args["charge_efficiency"]),
            discharge_efficiency=float(args["discharge_efficiency"]),
            initial_state_of_charge=float(args["initial_state_of_charge"]),
            terminal_state_of_charge=float(args["terminal_state_of_charge"]),
            training_verbose=bool(args["training_verbose"]),
            compute_ante_infodva=bool(args.get("compute_ante_infodva", False)),
        )
        model_config_path = model_dir / "model_config.json"
        if not model_config_path.exists():
            _write_json(model_config_path, _model_config_payload(record, config))

        training_artifacts = train_model(
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
            learning_rate=config.learning_rate,
            storage_parameters=config.storage_parameters,
            training_verbose=config.training_verbose,
            training_log_every=config.training_log_every,
            xgb_n_estimators=config.xgb_n_estimators,
            xgb_max_depth=config.xgb_max_depth,
            xgb_learning_rate=config.xgb_learning_rate,
            xgb_subsample=config.xgb_subsample,
            xgb_colsample_bytree=config.xgb_colsample_bytree,
            xgb_reg_lambda=config.xgb_reg_lambda,
            xgb_verbosity=config.xgb_verbosity,
        )
        sage_payload = _run_sage_ablation(
            model_id=model_id,
            split=split,
            config=config,
            training_artifacts=training_artifacts,
            model_dir=model_dir,
            bootstrap_replicates=int(args["bootstrap_replicates"]),
            validation_baseline=None,
            test_baseline=None,
            overwrite=bool(args["overwrite"]),
            existing_result=existing_result,
        )
        updated_result = dict(existing_result)
        updated_result.update(sage_payload["result"])
        updated_result["sage_append_status"] = "completed"
        updated_result["sage_append_runtime_seconds"] = time.perf_counter() - started_at
        _add_sage_relative_comparisons(updated_result)
        _write_json(model_result_path, updated_result)
        _merge_sage_summary_into_selected_summary(
            model_dir,
            sage_payload["summary"],
        )
        return updated_result
    except Exception as exc:
        error_payload = dict(existing_result)
        error_payload.update(
            {
                "sage_append_status": "failed",
                "sage_error": repr(exc),
                "sage_traceback": traceback.format_exc(),
                "sage_append_runtime_seconds": time.perf_counter() - started_at,
            }
        )
        _write_json(model_dir / "sage_error.json", error_payload)
        _write_json(model_result_path, error_payload)
        return error_payload


def _evaluate_split(
    *,
    config: CaisoShapCaseStudyConfig,
    training_artifacts: Any,
    split: CaisoGuidedValidationSplit,
    explain_frame: pd.DataFrame,
    outdir: Path,
    label: str,
    holdout_mean_impute_features: Sequence[str] = (),
) -> Any:
    outputs = run_caiso_shap_case_study_with_artifacts(
        config=config,
        training_artifacts=training_artifacts,
        dataset_path=split.dataset_path,
        date_column=split.date_column,
        feature_columns=split.feature_columns,
        target_columns=split.target_columns,
        train_frame=split.train_frame,
        background_frame=split.background_frame,
        explain_frame=explain_frame,
        holdout_mean_impute_features=holdout_mean_impute_features,
        holdout_days=split.validation_days + split.test_days,
        max_days=len(explain_frame),
        evaluation_label=label,
    )
    write_caiso_shap_case_study_outputs(outputs, outdir)
    return outputs


def _run_sage_ablation(
    *,
    model_id: str,
    split: CaisoGuidedValidationSplit,
    config: CaisoShapCaseStudyConfig,
    training_artifacts: Any,
    model_dir: Path,
    bootstrap_replicates: int,
    validation_baseline: Any | None,
    test_baseline: Any | None,
    overwrite: bool,
    existing_result: dict[str, Any] | None = None,
    validation_feature_mask_cache: dict[str, Any] | None = None,
    test_feature_mask_cache: dict[str, Any] | None = None,
    label_prefix: str = "",
) -> dict[str, Any]:
    started_at = time.perf_counter()
    daily_sage = compute_daily_sage_values(
        model=training_artifacts.model,
        feature_columns=split.feature_columns,
        target_columns=split.target_columns,
        background_frame=split.background_frame,
        explain_frame=split.validation_frame,
        date_column=split.date_column,
    )
    sage_candidates = select_sage_disruptive_feature_candidates(
        daily_sage,
        split.feature_columns,
        bootstrap_replicates=bootstrap_replicates,
        random_state=int(config.random_state or 0),
    )
    daily_sage.to_csv(model_dir / "validation_sage_daily.csv", index=False)
    sage_candidates.to_csv(model_dir / "sage_candidate_features.csv", index=False)

    disruptive_candidates = sage_candidates.loc[sage_candidates["disruptive"]].head(
        DEFAULT_TOP_K_CANDIDATES
    )
    validation_baseline_outputs = validation_baseline
    test_baseline_outputs = test_baseline
    if not disruptive_candidates.empty and validation_baseline_outputs is None:
        validation_baseline_outputs = _evaluate_split_or_load_metrics(
            config=config,
            training_artifacts=training_artifacts,
            split=split,
            explain_frame=split.validation_frame,
            outdir=model_dir / "validation_baseline",
            label="validation_baseline",
            overwrite=overwrite,
        )
    validation_candidate_summaries: list[dict[str, Any]] = []
    candidate_outputs_by_rank: dict[int, Any] = {}
    for _, candidate in disruptive_candidates.iterrows():
        feature_name = str(candidate["feature"])
        candidate_rank = int(candidate["candidate_rank"])
        candidate_outputs = _evaluate_feature_mask_split_with_cache(
            cache=validation_feature_mask_cache,
            config=config,
            training_artifacts=training_artifacts,
            split=split,
            explain_frame=split.validation_frame,
            outdir=(
                model_dir
                / "validation_sage_candidates"
                / f"{candidate_rank:02d}_{_slugify(feature_name)}"
            ),
            label=f"{label_prefix}validation_sage_hide_{feature_name}",
            feature_name=feature_name,
            overwrite=overwrite,
        )
        candidate_outputs_by_rank[candidate_rank] = candidate_outputs
        candidate_metrics = _coerce_metrics(candidate_outputs)
        validation_baseline_metrics = _coerce_metrics(validation_baseline_outputs)
        validation_sage_rmse_improvement = (
            float(validation_baseline_metrics["rmse"]) - float(candidate_metrics["rmse"])
        )
        validation_sage_regret_improvement = _regret_improvement(
            validation_baseline_outputs,
            candidate_outputs,
        )
        validation_candidate_summaries.append(
            {
                "feature": feature_name,
                "candidate_rank": candidate_rank,
                "mean_sage": float(candidate["mean_sage"]),
                "bootstrap_ci_lower": float(candidate["bootstrap_ci_lower"]),
                "bootstrap_ci_upper": float(candidate["bootstrap_ci_upper"]),
                "replacement_value": compute_background_feature_replacements(
                    split.background_frame,
                    (feature_name,),
                )[feature_name],
                **_prefixed_metrics("validation_sage_candidate", candidate_outputs),
                "validation_sage_rmse_improvement": validation_sage_rmse_improvement,
                "validation_sage_regret_improvement": validation_sage_regret_improvement,
            }
        )

    selected_candidate_summary = _select_sage_validation_candidate_summary(
        validation_candidate_summaries
    )
    selected_feature = (
        None
        if selected_candidate_summary is None
        else str(selected_candidate_summary["feature"])
    )
    selected_candidate_rank = (
        None
        if selected_candidate_summary is None
        else int(selected_candidate_summary["candidate_rank"])
    )
    selected_candidate_row = (
        None
        if selected_candidate_rank is None
        else disruptive_candidates.loc[
            disruptive_candidates["candidate_rank"] == selected_candidate_rank
        ].iloc[0]
    )
    selected_validation_outputs = (
        None
        if selected_candidate_rank is None
        else candidate_outputs_by_rank[selected_candidate_rank]
    )
    selected_test_outputs = None
    if selected_feature is not None:
        if test_baseline_outputs is None:
            test_baseline_outputs = _evaluate_split_or_load_metrics(
                config=config,
                training_artifacts=training_artifacts,
                split=split,
                explain_frame=split.test_frame,
                outdir=model_dir / "test_baseline",
                label="test_baseline",
                overwrite=overwrite,
            )
        selected_test_outputs = _evaluate_feature_mask_split_with_cache(
            cache=test_feature_mask_cache,
            config=config,
            training_artifacts=training_artifacts,
            split=split,
            explain_frame=split.test_frame,
            outdir=model_dir / "test_sage_selected_hide",
            label=f"{label_prefix}test_sage_hide_{selected_feature}",
            feature_name=selected_feature,
            overwrite=overwrite,
        )

    selection_status = (
        "no_disruptive_feature"
        if validation_candidate_summaries == []
        else (
            "fewer_than_three_disruptive_features"
            if len(validation_candidate_summaries) < DEFAULT_TOP_K_CANDIDATES
            else "selected"
        )
    )
    validation_mean_null_loss = float(daily_sage["sage_loss_null"].mean())
    validation_mean_full_loss = float(daily_sage["sage_loss_full"].mean())
    validation_sage_value_full = float(daily_sage["sage_value_full"].mean())
    validation_sage_efficiency_gap = float(
        daily_sage["sage_efficiency_gap"].abs().max()
    )
    summary: dict[str, Any] = {
        "status": selection_status,
        "model_id": model_id,
        "value_definition": "v_sage(S) = L(null) - L(S)",
        "loss_definition": "mean squared error averaged over the target hours",
        "ci_level": DEFAULT_CI_LEVEL,
        "bootstrap_replicates": bootstrap_replicates,
        "disruptive_definition": "bootstrap_ci_upper < 0",
        "selection_rule": (
            "evaluate up to three disruptive validation SAGE candidates, hide each "
            "candidate on validation, and select the largest validation RMSE improvement"
        ),
        "disruptive_feature_count": int(sage_candidates["disruptive"].sum()),
        "evaluated_candidate_count": len(validation_candidate_summaries),
        "selected_feature": selected_feature,
        "validation_mean_null_loss": validation_mean_null_loss,
        "validation_mean_full_loss": validation_mean_full_loss,
        "validation_sage_value_full": validation_sage_value_full,
        "validation_sage_efficiency_gap_max_abs_daily": validation_sage_efficiency_gap,
        "validation_candidates": validation_candidate_summaries,
    }
    if selected_candidate_row is not None:
        summary["selected_candidate"] = {
            "feature": selected_feature,
            "candidate_rank": int(selected_candidate_row["candidate_rank"]),
            "mean_sage": float(selected_candidate_row["mean_sage"]),
            "bootstrap_ci_lower": float(selected_candidate_row["bootstrap_ci_lower"]),
            "bootstrap_ci_upper": float(selected_candidate_row["bootstrap_ci_upper"]),
            "validation_sage_rmse_improvement": (
                None
                if selected_candidate_summary is None
                else float(selected_candidate_summary["validation_sage_rmse_improvement"])
            ),
            "validation_sage_regret_improvement": (
                None
                if selected_candidate_summary is None
                else float(
                    selected_candidate_summary["validation_sage_regret_improvement"]
                )
            ),
        }
    if selected_test_outputs is not None and selected_feature is not None:
        summary["selected_test_replacement_value"] = (
            compute_background_feature_replacements(
                split.background_frame,
                (selected_feature,),
            )[selected_feature]
        )
        summary["test_sage_regret_improvement"] = _regret_improvement(
            test_baseline_outputs,
            selected_test_outputs,
        )
    _write_json(model_dir / "sage_selected_feature_summary.json", summary)

    result: dict[str, Any] = {
        SAGE_RESULT_STATUS_KEY: selection_status,
        "sage_disruptive_feature_count": int(sage_candidates["disruptive"].sum()),
        "sage_evaluated_candidate_count": len(validation_candidate_summaries),
        "sage_selected_feature": selected_feature,
        "sage_validation_mean_null_loss": validation_mean_null_loss,
        "sage_validation_mean_full_loss": validation_mean_full_loss,
        "sage_validation_value_full": validation_sage_value_full,
        "sage_validation_efficiency_gap_max_abs_daily": validation_sage_efficiency_gap,
        "sage_runtime_seconds": time.perf_counter() - started_at,
    }
    if selected_candidate_row is not None:
        result.update(
            {
                "sage_selected_candidate_rank": int(
                    selected_candidate_row["candidate_rank"]
                ),
                "sage_selected_mean_value": float(selected_candidate_row["mean_sage"]),
                "sage_selected_bootstrap_ci_lower": float(
                    selected_candidate_row["bootstrap_ci_lower"]
                ),
                "sage_selected_bootstrap_ci_upper": float(
                    selected_candidate_row["bootstrap_ci_upper"]
                ),
            }
        )
        if selected_candidate_summary is not None:
            result["sage_selected_validation_rmse_improvement"] = float(
                selected_candidate_summary["validation_sage_rmse_improvement"]
            )
            result["sage_selected_validation_regret_improvement"] = float(
                selected_candidate_summary["validation_sage_regret_improvement"]
            )
    if selected_validation_outputs is not None:
        result.update(
            _prefixed_metrics("validation_sage_hidden", selected_validation_outputs)
        )
        result.update(
            _prefixed_metric_deltas(
                "validation_sage",
                validation_baseline_outputs,
                selected_validation_outputs,
            )
        )
        result["validation_sage_regret_improvement"] = _regret_improvement(
            validation_baseline_outputs,
            selected_validation_outputs,
        )
    if selected_test_outputs is not None:
        result.update(_prefixed_metrics("test_sage_hidden", selected_test_outputs))
        result.update(
            _prefixed_metric_deltas(
                "test_sage",
                test_baseline_outputs,
                selected_test_outputs,
            )
        )
        result["test_sage_regret_improvement"] = _regret_improvement(
            test_baseline_outputs,
            selected_test_outputs,
        )
    if existing_result:
        result = {**result}
        comparison_payload = dict(existing_result)
        comparison_payload.update(result)
        _add_sage_relative_comparisons(comparison_payload)
        for key, value in comparison_payload.items():
            if key.startswith("validation_sage_minus_") or key.startswith(
                "test_sage_minus_"
            ):
                result[key] = value
    return {"summary": summary, "result": result}


def _select_sage_validation_candidate_summary(
    validation_candidate_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    if not validation_candidate_summaries:
        return None
    return max(
        validation_candidate_summaries,
        key=lambda row: (
            row["validation_sage_rmse_improvement"],
            row["validation_sage_regret_improvement"],
            -row["candidate_rank"],
        ),
    )


def _run_leave_one_feature_out_ablation(
    *,
    model_id: str,
    split: CaisoGuidedValidationSplit,
    config: CaisoShapCaseStudyConfig,
    training_artifacts: Any,
    model_dir: Path,
    validation_baseline: Any,
    test_baseline: Any,
    overwrite: bool,
    existing_result: dict[str, Any] | None = None,
    validation_feature_mask_cache: dict[str, Any] | None = None,
    test_feature_mask_cache: dict[str, Any] | None = None,
    label_prefix: str = "",
) -> dict[str, Any]:
    started_at = time.perf_counter()
    validation_baseline_metrics = _coerce_metrics(validation_baseline)
    validation_baseline_total_regret = _total_regret(validation_baseline_metrics)
    validation_candidate_summaries: list[dict[str, Any]] = []
    candidate_outputs_by_feature: dict[str, Any] = {}
    for feature_idx, feature_name in enumerate(split.feature_columns, start=1):
        candidate_outputs = _evaluate_feature_mask_split_with_cache(
            cache=validation_feature_mask_cache,
            config=config,
            training_artifacts=training_artifacts,
            split=split,
            explain_frame=split.validation_frame,
            outdir=model_dir / "validation_lofo_candidates" / _slugify(feature_name),
            label=f"{label_prefix}validation_lofo_hide_{feature_name}",
            feature_name=feature_name,
            overwrite=overwrite,
        )
        candidate_outputs_by_feature[feature_name] = candidate_outputs
        candidate_metrics = _coerce_metrics(candidate_outputs)
        candidate_total_regret = _total_regret(candidate_metrics)
        total_regret_delta = candidate_total_regret - validation_baseline_total_regret
        validation_candidate_summaries.append(
            {
                "feature": feature_name,
                "candidate_rank": feature_idx,
                "replacement_value": compute_background_feature_replacements(
                    split.background_frame,
                    (feature_name,),
                )[feature_name],
                **_prefixed_metrics("validation_lofo_candidate", candidate_outputs),
                "validation_lofo_total_regret": candidate_total_regret,
                "validation_lofo_total_regret_delta": total_regret_delta,
                "validation_lofo_regret_improvement": _regret_improvement(
                    validation_baseline,
                    candidate_outputs,
                ),
                "validation_lofo_rmse_improvement": (
                    float(validation_baseline_metrics["rmse"])
                    - float(candidate_metrics["rmse"])
                ),
            }
        )

    selected_candidate_summary = _select_lofo_validation_candidate_summary(
        validation_candidate_summaries
    )
    selected_feature = (
        None
        if selected_candidate_summary is None
        else str(selected_candidate_summary["feature"])
    )
    selected_validation_outputs = (
        None
        if selected_feature is None
        else candidate_outputs_by_feature[selected_feature]
    )
    selected_test_outputs = None
    if selected_feature is not None:
        selected_test_outputs = _evaluate_feature_mask_split_with_cache(
            cache=test_feature_mask_cache,
            config=config,
            training_artifacts=training_artifacts,
            split=split,
            explain_frame=split.test_frame,
            outdir=model_dir / "test_lofo_selected_hide",
            label=f"{label_prefix}test_lofo_hide_{selected_feature}",
            feature_name=selected_feature,
            overwrite=overwrite,
        )

    selection_status = (
        "no_feature"
        if selected_candidate_summary is None
        else (
            "selected"
            if len(validation_candidate_summaries) == len(split.feature_columns)
            else "partial_feature_set"
        )
    )
    summary: dict[str, Any] = {
        "status": selection_status,
        "model_id": model_id,
        "selection_rule": (
            "mask every feature individually on validation and select argmin of "
            "total_regret(masked) - total_regret(full)"
        ),
        "evaluated_candidate_count": len(validation_candidate_summaries),
        "selected_feature": selected_feature,
        "validation_baseline_total_regret": validation_baseline_total_regret,
        "validation_candidates": validation_candidate_summaries,
    }
    if selected_candidate_summary is not None:
        summary["selected_candidate"] = {
            "feature": selected_feature,
            "validation_lofo_total_regret_delta": float(
                selected_candidate_summary["validation_lofo_total_regret_delta"]
            ),
            "validation_lofo_regret_improvement": float(
                selected_candidate_summary["validation_lofo_regret_improvement"]
            ),
            "validation_lofo_rmse_improvement": float(
                selected_candidate_summary["validation_lofo_rmse_improvement"]
            ),
        }
    if selected_test_outputs is not None and selected_feature is not None:
        summary["selected_test_replacement_value"] = (
            compute_background_feature_replacements(
                split.background_frame,
                (selected_feature,),
            )[selected_feature]
        )
        summary["test_lofo_regret_improvement"] = _regret_improvement(
            test_baseline,
            selected_test_outputs,
        )
    _write_json(model_dir / "lofo_selected_feature_summary.json", summary)

    result: dict[str, Any] = {
        LOFO_RESULT_STATUS_KEY: selection_status,
        "lofo_evaluated_candidate_count": len(validation_candidate_summaries),
        "lofo_selected_feature": selected_feature,
        "lofo_validation_baseline_total_regret": validation_baseline_total_regret,
        "lofo_runtime_seconds": time.perf_counter() - started_at,
    }
    if selected_candidate_summary is not None:
        result["lofo_selected_validation_total_regret_delta"] = float(
            selected_candidate_summary["validation_lofo_total_regret_delta"]
        )
        result["lofo_selected_validation_regret_improvement"] = float(
            selected_candidate_summary["validation_lofo_regret_improvement"]
        )
        result["lofo_selected_validation_rmse_improvement"] = float(
            selected_candidate_summary["validation_lofo_rmse_improvement"]
        )
    if selected_validation_outputs is not None:
        result.update(
            _prefixed_metrics("validation_lofo_hidden", selected_validation_outputs)
        )
        result.update(
            _prefixed_metric_deltas(
                "validation_lofo",
                validation_baseline,
                selected_validation_outputs,
            )
        )
        result["validation_lofo_regret_improvement"] = _regret_improvement(
            validation_baseline,
            selected_validation_outputs,
        )
    if selected_test_outputs is not None:
        result.update(_prefixed_metrics("test_lofo_hidden", selected_test_outputs))
        result.update(
            _prefixed_metric_deltas(
                "test_lofo",
                test_baseline,
                selected_test_outputs,
            )
        )
        result["test_lofo_regret_improvement"] = _regret_improvement(
            test_baseline,
            selected_test_outputs,
        )
    if existing_result:
        comparison_payload = dict(existing_result)
        comparison_payload.update(result)
        _add_lofo_relative_comparisons(comparison_payload)
        for key, value in comparison_payload.items():
            if key.startswith("validation_lofo_minus_") or key.startswith(
                "test_lofo_minus_"
            ):
                result[key] = value
    return {"summary": summary, "result": result}


def _select_lofo_validation_candidate_summary(
    validation_candidate_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    if not validation_candidate_summaries:
        return None
    return min(
        validation_candidate_summaries,
        key=lambda row: (
            row["validation_lofo_total_regret_delta"],
            -row["validation_lofo_rmse_improvement"],
            row["feature"],
        ),
    )


def _evaluate_feature_mask_split_with_cache(
    *,
    cache: dict[str, Any] | None,
    config: CaisoShapCaseStudyConfig,
    training_artifacts: Any,
    split: CaisoGuidedValidationSplit,
    explain_frame: pd.DataFrame,
    outdir: Path,
    label: str,
    feature_name: str,
    overwrite: bool,
) -> Any:
    if cache is not None and feature_name in cache:
        return cache[feature_name]
    outputs = _evaluate_split_or_load_metrics(
        config=config,
        training_artifacts=training_artifacts,
        split=split,
        explain_frame=explain_frame,
        outdir=outdir,
        label=label,
        holdout_mean_impute_features=(feature_name,),
        overwrite=overwrite,
    )
    if cache is not None:
        cache[feature_name] = outputs
    return outputs


def _total_regret(metrics: dict[str, Any]) -> float:
    return float(metrics["mean_actual_daily_regret"]) * float(metrics["days"])


def _evaluate_split_or_load_metrics(
    *,
    config: CaisoShapCaseStudyConfig,
    training_artifacts: Any,
    split: CaisoGuidedValidationSplit,
    explain_frame: pd.DataFrame,
    outdir: Path,
    label: str,
    holdout_mean_impute_features: Sequence[str] = (),
    overwrite: bool = False,
) -> Any:
    if not overwrite and _metrics_outputs_exist(outdir):
        return _load_metrics_from_dir(outdir)
    return _evaluate_split(
        config=config,
        training_artifacts=training_artifacts,
        split=split,
        explain_frame=explain_frame,
        outdir=outdir,
        label=label,
        holdout_mean_impute_features=holdout_mean_impute_features,
    )


def _metrics_outputs_exist(outdir: Path) -> bool:
    return all(
        (outdir / filename).exists()
        for filename in (
            "prediction_metrics.json",
            "evaluation_metrics.json",
            "daily_shap.csv",
        )
    )


def _result_has_sage_ablation(result: dict[str, Any]) -> bool:
    return SAGE_RESULT_STATUS_KEY in result


def _result_has_lofo_ablation(result: dict[str, Any]) -> bool:
    return LOFO_RESULT_STATUS_KEY in result


def _merge_sage_summary_into_selected_summary(
    model_dir: Path,
    sage_summary: dict[str, Any],
) -> None:
    summary_path = model_dir / "selected_feature_summary.json"
    summary = _load_json(summary_path) if summary_path.exists() else {}
    summary["sage_ablation"] = sage_summary
    _write_json(summary_path, summary)


def _add_sage_relative_comparisons(result: dict[str, Any]) -> None:
    if _is_finite_number(result.get("validation_sage_regret_improvement")):
        if _is_finite_number(result.get("validation_regret_improvement")):
            result["validation_sage_minus_decision_guided_regret_improvement"] = (
                float(result["validation_sage_regret_improvement"])
                - float(result["validation_regret_improvement"])
            )
        if _is_finite_number(result.get("validation_random_regret_improvement")):
            result["validation_sage_minus_random_regret_improvement"] = (
                float(result["validation_sage_regret_improvement"])
                - float(result["validation_random_regret_improvement"])
            )
    if _is_finite_number(result.get("test_sage_regret_improvement")):
        if _is_finite_number(result.get("test_regret_improvement")):
            result["test_sage_minus_decision_guided_regret_improvement"] = (
                float(result["test_sage_regret_improvement"])
                - float(result["test_regret_improvement"])
            )
        if _is_finite_number(result.get("test_random_regret_improvement")):
            result["test_sage_minus_random_regret_improvement"] = (
                float(result["test_sage_regret_improvement"])
                - float(result["test_random_regret_improvement"])
            )


def _add_lofo_relative_comparisons(result: dict[str, Any]) -> None:
    if _is_finite_number(result.get("validation_lofo_regret_improvement")):
        if _is_finite_number(result.get("validation_regret_improvement")):
            result["validation_lofo_minus_decision_guided_regret_improvement"] = (
                float(result["validation_lofo_regret_improvement"])
                - float(result["validation_regret_improvement"])
            )
        if _is_finite_number(result.get("validation_sage_regret_improvement")):
            result["validation_lofo_minus_sage_regret_improvement"] = (
                float(result["validation_lofo_regret_improvement"])
                - float(result["validation_sage_regret_improvement"])
            )
        if _is_finite_number(result.get("validation_random_regret_improvement")):
            result["validation_lofo_minus_random_regret_improvement"] = (
                float(result["validation_lofo_regret_improvement"])
                - float(result["validation_random_regret_improvement"])
            )
    if _is_finite_number(result.get("test_lofo_regret_improvement")):
        if _is_finite_number(result.get("test_regret_improvement")):
            result["test_lofo_minus_decision_guided_regret_improvement"] = (
                float(result["test_lofo_regret_improvement"])
                - float(result["test_regret_improvement"])
            )
        if _is_finite_number(result.get("test_sage_regret_improvement")):
            result["test_lofo_minus_sage_regret_improvement"] = (
                float(result["test_lofo_regret_improvement"])
                - float(result["test_sage_regret_improvement"])
            )
        if _is_finite_number(result.get("test_random_regret_improvement")):
            result["test_lofo_minus_random_regret_improvement"] = (
                float(result["test_lofo_regret_improvement"])
                - float(result["test_random_regret_improvement"])
            )


def _build_random_comparator_summary(
    *,
    random_feature: str | None,
    random_feature_row: pd.Series | None,
    random_mask_seed: int,
    model_id: str,
    split: CaisoGuidedValidationSplit,
    validation_baseline: Any,
    random_validation_outputs: Any | None,
    test_baseline: Any,
    random_test_outputs: Any | None,
) -> dict[str, Any]:
    if random_feature is None:
        return {
            "status": "no_ineligible_feature",
            "random_mask_seed": random_mask_seed,
            "model_seed_offset": _stable_model_seed_offset(model_id),
            "feature": None,
        }

    summary: dict[str, Any] = {
        "status": "evaluated",
        "random_mask_seed": random_mask_seed,
        "model_seed_offset": _stable_model_seed_offset(model_id),
        "feature": random_feature,
        "ineligibility_reason": (
            None
            if random_feature_row is None
            else str(random_feature_row["ineligibility_reason"])
        ),
        "ineligibility_reason_count": (
            None
            if random_feature_row is None
            else int(random_feature_row["ineligibility_reason_count"])
        ),
        "replacement_value": compute_background_feature_replacements(
            split.background_frame,
            (random_feature,),
        )[random_feature],
    }
    if random_validation_outputs is not None:
        summary["validation_metrics"] = _extract_metrics(random_validation_outputs)
        summary["validation_regret_improvement"] = _regret_improvement(
            validation_baseline,
            random_validation_outputs,
        )
    if random_test_outputs is not None:
        summary["test_metrics"] = _extract_metrics(random_test_outputs)
        summary["test_regret_improvement"] = _regret_improvement(
            test_baseline,
            random_test_outputs,
        )
    return summary


def _build_config_for_record(
    record: dict[str, Any],
    *,
    dataset_path: Path,
    model_dir: Path,
    background_days: int,
    throughput_penalty: float,
    energy_capacity: float | None = None,
    power_limit: float | None = None,
    charge_efficiency: float | None = None,
    discharge_efficiency: float | None = None,
    initial_state_of_charge: float | None = None,
    terminal_state_of_charge: float | None = None,
    training_verbose: bool = False,
    compute_ante_infodva: bool = False,
) -> CaisoShapCaseStudyConfig:
    storage_replacements: dict[str, float | None] = {
        "throughput_penalty": throughput_penalty,
    }
    if energy_capacity is not None:
        storage_replacements["energy_capacity"] = energy_capacity
    if power_limit is not None:
        storage_replacements["power_limit"] = power_limit
    if charge_efficiency is not None:
        storage_replacements["charge_efficiency"] = charge_efficiency
    if discharge_efficiency is not None:
        storage_replacements["discharge_efficiency"] = discharge_efficiency
    if initial_state_of_charge is not None:
        storage_replacements["initial_state_of_charge"] = initial_state_of_charge
    if terminal_state_of_charge is not None:
        storage_replacements["terminal_state_of_charge"] = terminal_state_of_charge
    storage_parameters = dataclasses.replace(
        build_default_storage_parameters(),
        **storage_replacements,
    )
    model_name = str(record["model_name"])
    common = {
        "dataset_path": dataset_path,
        "holdout_days": DEFAULT_VALIDATION_DAYS + DEFAULT_TEST_DAYS,
        "outdir": model_dir,
        "model_name": model_name,
        "n_jobs": 1,
        "background_days": background_days,
        "storage_parameters": storage_parameters,
        "interaction_order": None,
        "training_verbose": training_verbose,
        "compute_ead_decision_shap": compute_ante_infodva,
    }
    if model_name == "xgb":
        return CaisoShapCaseStudyConfig(
            **common,
            random_state=0,
            xgb_n_estimators=int(record["n_estimators"]),
            xgb_max_depth=int(record["max_depth"]),
            xgb_learning_rate=float(record["learning_rate"]),
            xgb_subsample=float(record["subsample"]),
            xgb_colsample_bytree=float(record["colsample_bytree"]),
            xgb_reg_lambda=float(record["reg_lambda"]),
        )
    raise ValueError(f"Unsupported model_name in manifest: {model_name}")


def _prefixed_metrics(prefix: str, outputs: Any) -> dict[str, Any]:
    metrics = _coerce_metrics(outputs)
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _prefixed_metric_deltas(
    prefix: str,
    baseline_outputs: Any,
    candidate_outputs: Any,
) -> dict[str, Any]:
    baseline_metrics = _coerce_metrics(baseline_outputs)
    candidate_metrics = _coerce_metrics(candidate_outputs)
    deltas: dict[str, Any] = {}
    for key, candidate_value in candidate_metrics.items():
        baseline_value = baseline_metrics.get(key)
        if _is_finite_number(candidate_value) and _is_finite_number(baseline_value):
            deltas[f"{prefix}_delta_{key}"] = float(candidate_value) - float(
                baseline_value
            )
    return deltas


def _coerce_metrics(outputs_or_metrics: Any) -> dict[str, Any]:
    if isinstance(outputs_or_metrics, dict):
        return dict(outputs_or_metrics)
    return _extract_metrics(outputs_or_metrics)


def _extract_metrics(outputs: Any) -> dict[str, Any]:
    holdout = dict(outputs.prediction_metrics["holdout"])
    daily_shap = outputs.daily_shap
    holdout["mean_oracle_value"] = float(daily_shap["oracle_obj"].mean())
    holdout["mean_realized_decision_value"] = float(
        daily_shap["decision_full_value"].mean()
    )
    holdout["mean_distance_from_oracle"] = float(
        (
            daily_shap["oracle_obj"]
            - daily_shap["decision_full_value"]
        ).mean()
    )
    _add_ante_infodva_metrics(holdout, daily_shap, outputs.summary_shap)
    for family_name, family_metrics in outputs.evaluation_metrics.items():
        for metric_name, metric_summary in family_metrics.items():
            metric_mean = metric_summary.get("mean")
            holdout[f"{family_name}_{metric_name}_mean"] = metric_mean
            if family_name == "ead_decision":
                holdout[f"ante_infodva_{metric_name}_mean"] = metric_mean
    return holdout


def _regret_improvement(
    baseline_outputs: Any,
    candidate_outputs: Any,
) -> float:
    baseline_metrics = _coerce_metrics(baseline_outputs)
    candidate_metrics = _coerce_metrics(candidate_outputs)
    baseline_regret = float(baseline_metrics["mean_actual_daily_regret"])
    candidate_regret = float(candidate_metrics["mean_actual_daily_regret"])
    return baseline_regret - candidate_regret


def _load_outputs_metrics_from_dir(outdir: Path) -> dict[str, Any]:
    metrics = _load_metrics_from_dir(outdir)
    return {f"validation_hidden_{key}": value for key, value in metrics.items()}


def _load_metrics_from_dir(outdir: Path) -> dict[str, Any]:
    prediction_metrics = _load_json(outdir / "prediction_metrics.json")
    evaluation_metrics = _load_json(outdir / "evaluation_metrics.json")
    daily_shap = pd.read_csv(outdir / "daily_shap.csv")
    metrics = dict(prediction_metrics["holdout"])
    metrics["mean_oracle_value"] = float(daily_shap["oracle_obj"].mean())
    metrics["mean_realized_decision_value"] = float(
        daily_shap["decision_full_value"].mean()
    )
    metrics["mean_distance_from_oracle"] = float(
        (daily_shap["oracle_obj"] - daily_shap["decision_full_value"]).mean()
    )
    summary_shap_path = outdir / "summary_shap.csv"
    summary_shap = pd.read_csv(summary_shap_path) if summary_shap_path.exists() else None
    _add_ante_infodva_metrics(metrics, daily_shap, summary_shap)
    for family_name, family_metrics in evaluation_metrics.items():
        for metric_name, metric_summary in family_metrics.items():
            metric_mean = metric_summary.get("mean")
            metrics[f"{family_name}_{metric_name}_mean"] = metric_mean
            if family_name == "ead_decision":
                metrics[f"ante_infodva_{metric_name}_mean"] = metric_mean
    return metrics


def _add_ante_infodva_metrics(
    metrics: dict[str, Any],
    daily_shap: pd.DataFrame,
    summary_shap: pd.DataFrame | None = None,
) -> None:
    """Add explicit ante-InfoDVA aliases for the existing EAD decision SHAP output."""

    if "ead_decision_value_gain" not in daily_shap.columns:
        return

    metrics["mean_ante_infodva_value_gain"] = float(
        daily_shap["ead_decision_value_gain"].mean()
    )
    if "ead_decision_baseline_value" in daily_shap.columns:
        metrics["mean_ante_infodva_baseline_value"] = float(
            daily_shap["ead_decision_baseline_value"].mean()
        )
    if "ead_decision_full_value" in daily_shap.columns:
        metrics["mean_ante_infodva_full_value"] = float(
            daily_shap["ead_decision_full_value"].mean()
        )
    if "ead_decision_characteristic_full_value" in daily_shap.columns:
        metrics["mean_ante_infodva_characteristic_full_value"] = float(
            daily_shap["ead_decision_characteristic_full_value"].mean()
        )

    ead_columns = [
        column
        for column in daily_shap.columns
        if column.startswith("ead_decision_shap_")
    ]
    if ead_columns:
        ead_values = daily_shap.loc[:, ead_columns].to_numpy(dtype=float)
        metrics["mean_abs_ante_infodva_shap"] = float(np.mean(np.abs(ead_values)))
        metrics["mean_signed_ante_infodva_shap"] = float(np.mean(ead_values))

    if summary_shap is not None and "ead_decision_mean_abs_shap" in summary_shap:
        metrics["max_feature_mean_abs_ante_infodva_shap"] = float(
            summary_shap["ead_decision_mean_abs_shap"].max()
        )


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(
        float(value)
    )


def _flat_hyperparameters(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"model_id", "model_name"}
        and not _is_missing(value)
    }


def _is_missing(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except TypeError:
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _count_ineligibility_reasons(value: Any) -> int:
    if not isinstance(value, str):
        return 0
    stripped = value.strip()
    if not stripped:
        return 0
    return len([reason for reason in stripped.split(";") if reason])


def _stable_model_seed_offset(model_id: str) -> int:
    offset = 0
    for char in model_id:
        offset = (offset * 131 + ord(char)) % 1_000_000_007
    return offset


def _model_config_payload(
    record: dict[str, Any],
    config: CaisoShapCaseStudyConfig,
) -> dict[str, Any]:
    return {
        "manifest_record": _flat_hyperparameters(record),
        "model_id": record["model_id"],
        "model_name": record["model_name"],
        "config": dataclasses.asdict(config),
    }


def _build_experiment_metadata(
    args: argparse.Namespace,
    split: CaisoGuidedValidationSplit,
    model_count: int,
) -> dict[str, Any]:
    return {
        "experiment": "caiso_decision_shap_guided_validation",
        "model_count": model_count,
        "model_family": args.model_family,
        "dataset_path": str(split.dataset_path),
        "date_column": split.date_column,
        "feature_columns": list(split.feature_columns),
        "target_columns": list(split.target_columns),
        "train_rows": int(len(split.train_frame)),
        "validation_rows": int(len(split.validation_frame)),
        "test_rows": int(len(split.test_frame)),
        "background_rows": int(len(split.background_frame)),
        "train_date_start": split.train_frame[split.date_column].iloc[0],
        "train_date_end": split.train_frame[split.date_column].iloc[-1],
        "validation_date_start": split.validation_frame[split.date_column].iloc[0],
        "validation_date_end": split.validation_frame[split.date_column].iloc[-1],
        "test_date_start": split.test_frame[split.date_column].iloc[0],
        "test_date_end": split.test_frame[split.date_column].iloc[-1],
        "background_date_start": split.background_frame[split.date_column].iloc[0],
        "background_date_end": split.background_frame[split.date_column].iloc[-1],
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_ci_level": DEFAULT_CI_LEVEL,
        "min_mean_abs_fraction": DEFAULT_MIN_MEAN_ABS_FRACTION,
        "top_k_candidates": DEFAULT_TOP_K_CANDIDATES,
        "train_months": args.train_months,
        "validation_months": args.validation_months,
        "validation_days": int(args.validation_days),
        "test_days": int(args.test_days),
        "test_rest": bool(args.test_rest),
        "random_mask_seed": int(args.random_mask_seed),
        "random_mask_selection": (
            "uniform_over_ineligible_features_with_max_failed_eligibility_criteria"
        ),
        "ante_infodva_enabled": bool(args.compute_ante_infodva),
        "ante_infodva_definition": (
            "ante-InfoDVA uses the full model prediction as the valuation vector; "
            "implemented by the existing EAD decision SHAP family."
        ),
        "sage_ablation_enabled": not bool(args.skip_sage_ablation),
        "sage_value_definition": "v_sage(S) = L(null) - L(S)",
        "sage_loss_definition": (
            "daily mean squared error averaged over the target hours, then averaged "
            "over validation days"
        ),
        "sage_disruptive_definition": (
            f"{DEFAULT_CI_LEVEL:.0%} bootstrap upper confidence bound is less than zero"
        ),
        "sage_selection_rule": (
            "evaluate up to the three disruptive features with the lowest validation "
            "mean SAGE values, hide each on validation, then select the feature with "
            "the largest validation RMSE improvement for test hiding"
        ),
        "lofo_selection_rule": (
            "mask every feature individually on validation and select argmin of "
            "total_regret(masked) - total_regret(full)"
        ),
        "throughput_penalty": float(args.throughput_penalty),
        "storage_parameters": {
            "energy_capacity": float(args.energy_capacity),
            "power_limit": float(args.power_limit),
            "charge_efficiency": float(args.charge_efficiency),
            "discharge_efficiency": float(args.discharge_efficiency),
            "throughput_penalty": float(args.throughput_penalty),
            "initial_state_of_charge": float(args.initial_state_of_charge),
            "terminal_state_of_charge": float(args.terminal_state_of_charge),
        },
        "validation_max_days": args.validation_max_days,
        "test_max_days": args.test_max_days,
    }


def _write_results_csv(path: Path, results: Sequence[dict[str, Any]]) -> None:
    pd.DataFrame(results).sort_values("model_id").to_csv(path, index=False)


def _load_preserved_results(
    path: Path,
    *,
    excluded_model_ids: set[str],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    existing = pd.read_csv(path)
    if "model_id" not in existing.columns:
        return []
    preserved = existing.loc[
        ~existing["model_id"].astype(str).isin(excluded_model_ids)
    ]
    return preserved.to_dict(orient="records")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


if __name__ == "__main__":
    main()
