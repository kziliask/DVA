from __future__ import annotations

import argparse
import dataclasses
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from dva.analysis.run_caiso_decision_shap_guided_validation import (
    DEFAULT_BACKGROUND_DAYS,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_CI_LEVEL,
    DEFAULT_OUTDIR,
    DEFAULT_RANDOM_MASK_SEED,
    DEFAULT_STORAGE_PARAMETER_TEMPLATE,
    DEFAULT_THROUGHPUT_PENALTY,
    LOFO_RESULT_STATUS_KEY,
    DEFAULT_TOP_K_CANDIDATES,
    SAGE_RESULT_STATUS_KEY,
    CaisoGuidedValidationSplit,
    _add_sage_relative_comparisons,
    _add_lofo_relative_comparisons,
    _build_config_for_record,
    _build_random_comparator_summary,
    _filter_manifest_by_model_family,
    _flat_hyperparameters,
    _load_json,
    _load_metrics_from_dir,
    _load_outputs_metrics_from_dir,
    _model_config_payload,
    _prefixed_metric_deltas,
    _prefixed_metrics,
    _regret_improvement,
    _run_leave_one_feature_out_ablation,
    _run_sage_ablation,
    _slugify,
    _write_json,
    compute_background_feature_replacements,
    make_model_manifest,
    select_harmful_feature_candidates,
    select_random_ineligible_feature,
)
from dva.analysis.caiso_shap import (
    select_recent_background_frame,
    run_caiso_shap_case_study_with_artifacts,
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


DEFAULT_ROLLING_OUTDIR = DEFAULT_OUTDIR.with_name(
    "caiso_decision_shap_guided_validation_rolling_windows"
)
DEFAULT_FOLD_COUNT = 4
DEFAULT_TRAIN_MONTHS = 24
DEFAULT_VALIDATION_MONTHS = 3
DEFAULT_TEST_MONTHS = 3
DEFAULT_STEP_MONTHS = 3


@dataclass(frozen=True, slots=True)
class CaisoGuidedValidationRollingFold:
    fold_id: str
    fold_index: int
    season: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    split: CaisoGuidedValidationSplit


def build_calendar_rolling_caiso_guided_validation_folds(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    *,
    fold_count: int = DEFAULT_FOLD_COUNT,
    train_months: int = DEFAULT_TRAIN_MONTHS,
    validation_months: int = DEFAULT_VALIDATION_MONTHS,
    test_months: int = DEFAULT_TEST_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
    start_date: str | None = None,
    background_days: int = DEFAULT_BACKGROUND_DAYS,
    validation_max_days: int | None = None,
    test_max_days: int | None = None,
    date_column: str = DEFAULT_DATE_COLUMN,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS,
) -> tuple[CaisoGuidedValidationRollingFold, ...]:
    if fold_count <= 0:
        raise ValueError("fold_count must be strictly positive.")
    if train_months <= 0:
        raise ValueError("train_months must be strictly positive.")
    if validation_months <= 0:
        raise ValueError("validation_months must be strictly positive.")
    if test_months <= 0:
        raise ValueError("test_months must be strictly positive.")
    if step_months <= 0:
        raise ValueError("step_months must be strictly positive.")
    if background_days <= 0:
        raise ValueError("background_days must be strictly positive.")

    frame = load_default_training_frame(
        dataset_path,
        date_column=date_column,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )
    date_values = pd.to_datetime(frame[date_column])
    first_date = pd.Timestamp(date_values.min()).normalize()
    last_available_end = pd.Timestamp(date_values.max()).normalize() + pd.Timedelta(
        days=1
    )
    resolved_start = (
        pd.Timestamp(start_date).normalize()
        if start_date is not None
        else _latest_aligned_start_date(
            first_date=first_date,
            last_available_end=last_available_end,
            fold_count=fold_count,
            train_months=train_months,
            validation_months=validation_months,
            test_months=test_months,
            step_months=step_months,
        )
    )

    folds: list[CaisoGuidedValidationRollingFold] = []
    for fold_offset in range(fold_count):
        train_start_ts = resolved_start + pd.DateOffset(
            months=fold_offset * step_months
        )
        validation_start_ts = train_start_ts + pd.DateOffset(months=train_months)
        test_start_ts = validation_start_ts + pd.DateOffset(months=validation_months)
        test_end_ts = test_start_ts + pd.DateOffset(months=test_months)
        if test_end_ts > last_available_end:
            raise ValueError(
                "Requested rolling folds extend beyond the available dataset. "
                f"Fold {fold_offset + 1} ends at {_date_string(test_end_ts)} but "
                f"the dataset ends before {_date_string(last_available_end)}."
            )

        train_frame = _slice_frame_by_date_window(
            frame,
            date_values,
            train_start_ts,
            validation_start_ts,
        )
        validation_frame = _slice_frame_by_date_window(
            frame,
            date_values,
            validation_start_ts,
            test_start_ts,
        )
        test_frame = _slice_frame_by_date_window(
            frame,
            date_values,
            test_start_ts,
            test_end_ts,
        )
        if validation_max_days is not None:
            if validation_max_days <= 0:
                raise ValueError(
                    "validation_max_days must be strictly positive when provided."
                )
            validation_frame = validation_frame.iloc[:validation_max_days].reset_index(
                drop=True
            )
        if test_max_days is not None:
            if test_max_days <= 0:
                raise ValueError(
                    "test_max_days must be strictly positive when provided."
                )
            test_frame = test_frame.iloc[:test_max_days].reset_index(drop=True)
        _validate_nonempty_fold_frames(
            fold_offset + 1,
            train_frame,
            validation_frame,
            test_frame,
        )

        background_frame = select_recent_background_frame(
            train_frame,
            date_column,
            background_days,
        )
        split = CaisoGuidedValidationSplit(
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
        season = _season_for_month(int(test_start_ts.month))
        fold_index = fold_offset + 1
        fold_id = f"fold_{fold_index:02d}_{season}_{_date_string(test_start_ts)}"
        folds.append(
            CaisoGuidedValidationRollingFold(
                fold_id=fold_id,
                fold_index=fold_index,
                season=season,
                train_start=str(train_frame[date_column].iloc[0]),
                train_end=str(train_frame[date_column].iloc[-1]),
                validation_start=str(validation_frame[date_column].iloc[0]),
                validation_end=str(validation_frame[date_column].iloc[-1]),
                test_start=str(test_frame[date_column].iloc[0]),
                test_end=str(test_frame[date_column].iloc[-1]),
                split=split,
            )
        )
    return tuple(folds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CAISO GDSI/SAGE guided validation over rolling seasonal "
            "calendar windows."
        ),
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_ROLLING_OUTDIR)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--model-id", action="append", default=None)
    parser.add_argument(
        "--model-family",
        choices=("all", "xgb"),
        default="xgb",
        help=(
            "Restrict the manifest to the XGBoost model family."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
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
    parser.add_argument("--fold-count", type=int, default=DEFAULT_FOLD_COUNT)
    parser.add_argument("--train-months", type=int, default=DEFAULT_TRAIN_MONTHS)
    parser.add_argument(
        "--validation-months",
        type=int,
        default=DEFAULT_VALIDATION_MONTHS,
    )
    parser.add_argument("--test-months", type=int, default=DEFAULT_TEST_MONTHS)
    parser.add_argument("--step-months", type=int, default=DEFAULT_STEP_MONTHS)
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help=(
            "Optional first fold train-start date. By default, use the latest "
            "dataset-aligned start that leaves all folds complete."
        ),
    )
    parser.add_argument("--validation-max-days", type=int, default=None)
    parser.add_argument("--test-max-days", type=int, default=None)
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

    folds = build_calendar_rolling_caiso_guided_validation_folds(
        args.dataset_path,
        fold_count=args.fold_count,
        train_months=args.train_months,
        validation_months=args.validation_months,
        test_months=args.test_months,
        step_months=args.step_months,
        start_date=args.start_date,
        background_days=args.background_days,
        validation_max_days=args.validation_max_days,
        test_max_days=args.test_max_days,
    )
    fold_manifest = _build_fold_manifest_frame(folds)
    fold_manifest.to_csv(outdir / "fold_manifest.csv", index=False)
    _write_json(
        outdir / "experiment_metadata.json",
        _build_rolling_experiment_metadata(args, folds, len(manifest)),
    )

    worker_args = {
        "dataset_path": str(args.dataset_path),
        "outdir": str(outdir),
        "overwrite": bool(args.overwrite),
        "bootstrap_replicates": int(args.bootstrap_replicates),
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
        "fold_count": int(args.fold_count),
        "train_months": int(args.train_months),
        "validation_months": int(args.validation_months),
        "test_months": int(args.test_months),
        "step_months": int(args.step_months),
        "start_date": args.start_date,
        "validation_max_days": args.validation_max_days,
        "test_max_days": args.test_max_days,
    }
    tasks = [
        {
            "record": record,
            "fold_index": fold.fold_index,
            "fold_id": fold.fold_id,
        }
        for fold in folds
        for record in manifest.to_dict(orient="records")
    ]
    run_ids = {
        _fold_model_run_id(str(task["fold_id"]), str(task["record"]["model_id"]))
        for task in tasks
    }
    results = _load_preserved_rolling_results(
        outdir / "model_results.csv",
        excluded_run_ids=run_ids,
    )
    if args.max_workers == 1:
        for index, task in enumerate(tasks, start=1):
            print(
                f"[{index}/{len(tasks)}] Running {task['fold_id']} "
                f"{task['record']['model_id']}",
                flush=True,
            )
            result = _run_single_fold_model(task, worker_args)
            results.append(result)
            _write_rolling_results_csv(outdir / "model_results.csv", results)
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(_run_single_fold_model, task, worker_args): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                record = task["record"]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive process boundary.
                    result = {
                        "fold_id": task["fold_id"],
                        "fold_index": task["fold_index"],
                        "model_id": record["model_id"],
                        "model_name": record["model_name"],
                        "fold_model_id": _fold_model_run_id(
                            str(task["fold_id"]),
                            str(record["model_id"]),
                        ),
                        "status": "failed",
                        "error": repr(exc),
                    }
                results.append(result)
                completed += 1
                _write_rolling_results_csv(outdir / "model_results.csv", results)
                print(
                    f"[{completed}/{len(tasks)}] Finished {task['fold_id']} "
                    f"{record['model_id']} with status={result.get('status')}",
                    flush=True,
                )

    print(f"Wrote rolling guided validation outputs to {outdir}", flush=True)


def _run_single_fold_model(
    task: dict[str, Any],
    args: dict[str, Any],
) -> dict[str, Any]:
    record = dict(task["record"])
    model_id = str(record["model_id"])
    fold_id = str(task["fold_id"])
    fold_index = int(task["fold_index"])
    fold_model_id = _fold_model_run_id(fold_id, model_id)
    outdir = Path(args["outdir"])
    model_dir = outdir / "folds" / fold_id / "models" / model_id
    model_result_path = model_dir / "model_result.json"
    if model_result_path.exists() and not args["overwrite"]:
        result = _load_json(model_result_path)
        if result.get("status") != "failed" and (
            not args.get("run_sage_ablation", True)
            or SAGE_RESULT_STATUS_KEY in result
        ) and LOFO_RESULT_STATUS_KEY in result:
            result["status"] = "skipped_existing"
            return result
    if model_dir.exists() and args["overwrite"]:
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    try:
        folds = build_calendar_rolling_caiso_guided_validation_folds(
            args["dataset_path"],
            fold_count=int(args["fold_count"]),
            train_months=int(args["train_months"]),
            validation_months=int(args["validation_months"]),
            test_months=int(args["test_months"]),
            step_months=int(args["step_months"]),
            start_date=args["start_date"],
            background_days=int(args["background_days"]),
            validation_max_days=args["validation_max_days"],
            test_max_days=args["test_max_days"],
        )
        fold = folds[fold_index - 1]
        split = fold.split
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
        )
        config = dataclasses.replace(
            config,
            holdout_days=split.validation_days + split.test_days,
            outdir=model_dir,
        )
        _write_json(
            model_dir / "model_config.json",
            {
                **_model_config_payload(record, config),
                "fold": _fold_payload(fold),
                "fold_model_id": fold_model_id,
            },
        )

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
        validation_baseline = _evaluate_fold_split(
            config=config,
            training_artifacts=training_artifacts,
            split=split,
            explain_frame=split.validation_frame,
            outdir=model_dir / "validation_baseline",
            label=f"{fold_id}_validation_baseline",
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
            model_id=fold_model_id,
        )
        random_feature_row = (
            None
            if random_feature is None
            else candidate_table.loc[candidate_table["feature"] == random_feature].iloc[0]
        )
        random_validation_outputs = None
        random_test_outputs = None
        if random_feature is not None:
            random_validation_outputs = _evaluate_fold_split(
                config=config,
                training_artifacts=training_artifacts,
                split=split,
                explain_frame=split.validation_frame,
                outdir=model_dir / "validation_random_ineligible_hide",
                label=f"{fold_id}_validation_random_hide_{random_feature}",
                holdout_mean_impute_features=(random_feature,),
            )
            validation_feature_mask_cache[random_feature] = random_validation_outputs

        validation_candidate_summaries = []
        for _, candidate in eligible_candidates.iterrows():
            feature_name = str(candidate["feature"])
            candidate_rank = int(candidate["candidate_rank"])
            candidate_outputs = _evaluate_fold_split(
                config=config,
                training_artifacts=training_artifacts,
                split=split,
                explain_frame=split.validation_frame,
                outdir=(
                    model_dir
                    / "validation_candidates"
                    / f"{candidate_rank:02d}_{_slugify(feature_name)}"
                ),
                label=f"{fold_id}_validation_hide_{feature_name}",
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

        test_baseline = _evaluate_fold_split(
            config=config,
            training_artifacts=training_artifacts,
            split=split,
            explain_frame=split.test_frame,
            outdir=model_dir / "test_baseline",
            label=f"{fold_id}_test_baseline",
        )
        if random_feature is not None:
            random_test_outputs = _evaluate_fold_split(
                config=config,
                training_artifacts=training_artifacts,
                split=split,
                explain_frame=split.test_frame,
                outdir=model_dir / "test_random_ineligible_hide",
                label=f"{fold_id}_test_random_hide_{random_feature}",
                holdout_mean_impute_features=(random_feature,),
            )
            test_feature_mask_cache[random_feature] = random_test_outputs

        selected_feature = None
        selected_validation_outputs = None
        selected_test_outputs = None
        selected_summary = None
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
            selected_test_outputs = _evaluate_fold_split(
                config=config,
                training_artifacts=training_artifacts,
                split=split,
                explain_frame=split.test_frame,
                outdir=model_dir / "test_selected_hide",
                label=f"{fold_id}_test_hide_{selected_feature}",
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
            "fold": _fold_payload(fold),
            "fold_model_id": fold_model_id,
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
                model_id=fold_model_id,
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
            **_fold_result_payload(fold),
            "fold_model_id": fold_model_id,
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
            if selected_feature is not None and selected_summary is not None:
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
                model_id=fold_model_id,
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
                label_prefix=f"{fold_id}_",
            )
            summary["sage_ablation"] = sage_payload["summary"]
            result.update(sage_payload["result"])
            _add_sage_relative_comparisons(result)
        lofo_payload = _run_leave_one_feature_out_ablation(
            model_id=fold_model_id,
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
            label_prefix=f"{fold_id}_",
        )
        summary["lofo_ablation"] = lofo_payload["summary"]
        result.update(lofo_payload["result"])
        _add_lofo_relative_comparisons(result)
        result.update(_fold_result_payload(fold))
        result["fold_model_id"] = fold_model_id
        result["model_id"] = model_id
        result["runtime_seconds"] = time.perf_counter() - started_at
        _write_json(model_dir / "selected_feature_summary.json", summary)
        _write_json(model_result_path, result)
        return result
    except Exception as exc:
        error_payload = {
            **{
                "fold_id": fold_id,
                "fold_index": fold_index,
                "fold_model_id": fold_model_id,
                "model_id": model_id,
                "model_name": record.get("model_name"),
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "runtime_seconds": time.perf_counter() - started_at,
            }
        }
        _write_json(model_dir / "error.json", error_payload)
        _write_json(model_result_path, error_payload)
        return error_payload


def _evaluate_fold_split(
    *,
    config: Any,
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


def _latest_aligned_start_date(
    *,
    first_date: pd.Timestamp,
    last_available_end: pd.Timestamp,
    fold_count: int,
    train_months: int,
    validation_months: int,
    test_months: int,
    step_months: int,
) -> pd.Timestamp:
    candidate = first_date
    last_valid = candidate
    while True:
        final_end = (
            candidate
            + pd.DateOffset(months=(fold_count - 1) * step_months)
            + pd.DateOffset(months=train_months + validation_months + test_months)
        )
        if final_end > last_available_end:
            break
        last_valid = candidate
        candidate = candidate + pd.DateOffset(months=step_months)
    return last_valid


def _slice_frame_by_date_window(
    frame: pd.DataFrame,
    date_values: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    mask = (date_values >= start) & (date_values < end)
    return frame.loc[mask].reset_index(drop=True)


def _validate_nonempty_fold_frames(
    fold_index: int,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> None:
    if train_frame.empty:
        raise ValueError(f"Fold {fold_index} has an empty training window.")
    if validation_frame.empty:
        raise ValueError(f"Fold {fold_index} has an empty validation window.")
    if test_frame.empty:
        raise ValueError(f"Fold {fold_index} has an empty test window.")


def _build_fold_manifest_frame(
    folds: Sequence[CaisoGuidedValidationRollingFold],
) -> pd.DataFrame:
    return pd.DataFrame([_fold_payload(fold) for fold in folds])


def _fold_payload(fold: CaisoGuidedValidationRollingFold) -> dict[str, Any]:
    split = fold.split
    return {
        "fold_id": fold.fold_id,
        "fold_index": fold.fold_index,
        "season": fold.season,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "validation_start": fold.validation_start,
        "validation_end": fold.validation_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "train_rows": int(len(split.train_frame)),
        "validation_rows": int(len(split.validation_frame)),
        "test_rows": int(len(split.test_frame)),
        "background_rows": int(len(split.background_frame)),
        "background_start": str(split.background_frame[split.date_column].iloc[0]),
        "background_end": str(split.background_frame[split.date_column].iloc[-1]),
    }


def _fold_result_payload(fold: CaisoGuidedValidationRollingFold) -> dict[str, Any]:
    return {
        "fold_id": fold.fold_id,
        "fold_index": fold.fold_index,
        "season": fold.season,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "validation_start": fold.validation_start,
        "validation_end": fold.validation_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
    }


def _build_rolling_experiment_metadata(
    args: argparse.Namespace,
    folds: Sequence[CaisoGuidedValidationRollingFold],
    model_count: int,
) -> dict[str, Any]:
    first_split = folds[0].split
    return {
        "experiment": "caiso_decision_shap_guided_validation_rolling_windows",
        "dataset_path": str(first_split.dataset_path),
        "date_column": first_split.date_column,
        "feature_columns": list(first_split.feature_columns),
        "target_columns": list(first_split.target_columns),
        "model_family": args.model_family,
        "model_count_per_fold": model_count,
        "fold_count": len(folds),
        "total_model_runs": model_count * len(folds),
        "folds": [_fold_payload(fold) for fold in folds],
        "train_months": int(args.train_months),
        "validation_months": int(args.validation_months),
        "test_months": int(args.test_months),
        "step_months": int(args.step_months),
        "start_date": args.start_date,
        "resolved_start_date": folds[0].train_start,
        "background_days": int(args.background_days),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_ci_level": DEFAULT_CI_LEVEL,
        "top_k_candidates": DEFAULT_TOP_K_CANDIDATES,
        "random_mask_seed": int(args.random_mask_seed),
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
        "sage_ablation_enabled": not bool(args.skip_sage_ablation),
        "sage_value_definition": "v_sage(S) = L(null) - L(S)",
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
    }


def _load_preserved_rolling_results(
    path: Path,
    *,
    excluded_run_ids: set[str],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    existing = pd.read_csv(path)
    if "fold_model_id" not in existing.columns:
        return []
    preserved = existing.loc[
        ~existing["fold_model_id"].astype(str).isin(excluded_run_ids)
    ]
    return preserved.to_dict(orient="records")


def _write_rolling_results_csv(
    path: Path,
    results: Sequence[dict[str, Any]],
) -> None:
    frame = pd.DataFrame(results)
    sort_columns = [
        column
        for column in ("fold_index", "model_id", "fold_model_id")
        if column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns)
    frame.to_csv(path, index=False)


def _fold_model_run_id(fold_id: str, model_id: str) -> str:
    return f"{fold_id}__{model_id}"


def _season_for_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "fall"


def _date_string(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


if __name__ == "__main__":
    main()
