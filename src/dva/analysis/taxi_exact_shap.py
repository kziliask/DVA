from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import matplotlib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmcrameri  # noqa: F401 - registers cmc.* colormaps with Matplotlib.
from matplotlib.colors import Normalize

from dva.analysis.caiso_shap import compute_exact_shapley_values
from dva.analysis.evaluation_metrics import (
    build_metric_summary,
    compute_decision_deletion_auc,
    compute_decision_insertion_auc,
    compute_exact_decision_infidelity,
)
from dva.model.orienteering import (
    DEFAULT_ORTOOLS_LOCAL_SEARCH_METAHEURISTIC,
    DEFAULT_ORTOOLS_TIME_LIMIT_S,
    DEFAULT_MAX_DISTANCE_BUDGET,
    DEFAULT_START_ZONE_ID,
    DEFAULT_ZONE_DISTANCE_MATRIX_PATH,
    OrienteeringSolveMethod,
    load_zone_distance_matrix,
    solve_orienteering,
)
from dva.model.taxi_features import (
    DEFAULT_TAXI_TRAINING_FEATURE_SET,
    TAXI_CATEGORICAL_FEATURES,
    TAXI_TARGET_COLUMN,
    TAXI_TIMESTAMP_COLUMN,
    TAXI_ZONE_ID_COLUMN,
    resolve_taxi_training_features,
)


DEFAULT_TAXI_FEATURES_PATH = Path(
    "data/nyc_data/processed/nyc_taxi_zone_hour_features_2025_manhattan.csv"
)
DEFAULT_MANHATTAN_DISTANCE_MATRIX_PATH = Path(
    "data/nyc_data/zone_centroid_distance_matrix_km_manhattan.parquet"
)
DEFAULT_OUTPUT_DIR = Path("results/taxi_rf_exact_shap_poc")
DEFAULT_HOLDOUT_HOURS = 24
DEFAULT_MAX_EXPLAIN_HOURS = 1
DEFAULT_BACKGROUND_ROWS = 8
DEFAULT_COALITION_BATCH_SIZE = 128
DEFAULT_PROGRESS_EVERY_COALITIONS = 4096
DEFAULT_MAX_ZONES = 25
DEFAULT_RF_ESTIMATORS = 100
TAXI_ZONE_SELECTION_COLUMN = "pickup_count"
BEESWARM_CMAP = plt.get_cmap("cmc.berlin")
FEATURE_VALUE_CLIP_PERCENTILES = (5, 95)
MISSING_FEATURE_COLOR = "#7f7f7f"


@dataclass(frozen=True, slots=True)
class TaxiExactShapConfig:
    features_path: Path = DEFAULT_TAXI_FEATURES_PATH
    distance_matrix_path: Path | None = None
    outdir: Path = DEFAULT_OUTPUT_DIR
    feature_set: str = DEFAULT_TAXI_TRAINING_FEATURE_SET
    holdout_hours: int = DEFAULT_HOLDOUT_HOURS
    max_hours: int | None = DEFAULT_MAX_EXPLAIN_HOURS
    max_zones: int | None = DEFAULT_MAX_ZONES
    background_rows: int = DEFAULT_BACKGROUND_ROWS
    coalition_batch_size: int = DEFAULT_COALITION_BATCH_SIZE
    progress_every_coalitions: int = DEFAULT_PROGRESS_EVERY_COALITIONS
    random_state: int = 0
    n_estimators: int = DEFAULT_RF_ESTIMATORS
    max_depth: int | None = None
    min_samples_leaf: int = 1
    n_jobs: int = 1
    rf_verbose: int = 0
    train_sample_rows: int | None = None
    max_distance_budget: float = DEFAULT_MAX_DISTANCE_BUDGET
    start_zone_id: int = DEFAULT_START_ZONE_ID
    end_zone_id: int | None = None
    orienteering_method: OrienteeringSolveMethod = "heuristic"
    orienteering_threads: int = 1
    save_coalition_values: bool = False


@dataclass(frozen=True, slots=True)
class TaxiExactShapOutputs:
    hourly_shap: pd.DataFrame
    summary_shap: pd.DataFrame
    full_routes: pd.DataFrame
    prediction_metrics: dict[str, Any]
    evaluation_metrics: dict[str, Any]
    run_metadata: dict[str, Any]
    coalition_values: pd.DataFrame | None = None


class ExactBackgroundCoalitionPredictor:
    def __init__(
        self,
        model: Pipeline,
        feature_names: Sequence[str],
        background_frame: pd.DataFrame,
        *,
        coalition_batch_size: int,
    ) -> None:
        if coalition_batch_size <= 0:
            raise ValueError("coalition_batch_size must be strictly positive.")

        self.model = model
        self.feature_names = tuple(feature_names)
        self.categorical_feature_names = tuple(
            feature_name
            for feature_name in self.feature_names
            if feature_name in TAXI_CATEGORICAL_FEATURES
        )
        self.numeric_feature_names = tuple(
            feature_name
            for feature_name in self.feature_names
            if feature_name not in self.categorical_feature_names
        )
        self.feature_count = len(self.feature_names)
        self.coalition_count = 1 << self.feature_count
        self.background_frame = background_frame.loc[:, list(self.feature_names)].copy()
        self.background_values = self.background_frame.to_numpy(dtype=object, copy=True)
        self.background_count = int(len(self.background_frame))
        self.coalition_batch_size = coalition_batch_size
        self._included_indices_by_mask = tuple(
            tuple(
                feature_idx
                for feature_idx in range(self.feature_count)
                if coalition_mask & (1 << feature_idx)
            )
            for coalition_mask in range(self.coalition_count)
        )

    def predict_all_coalitions(
        self,
        observations: pd.DataFrame,
        *,
        progress_label: str | None = None,
        progress_every_coalitions: int = DEFAULT_PROGRESS_EVERY_COALITIONS,
    ) -> np.ndarray:
        observation_frame = observations.loc[:, list(self.feature_names)].reset_index(drop=True)
        observation_values = observation_frame.to_numpy(dtype=object, copy=True)
        zone_count = int(len(observation_frame))
        rows_per_mask = self.background_count * zone_count
        if rows_per_mask <= 0:
            raise ValueError("Need at least one observation row and one background row.")

        background_template = np.repeat(self.background_values, zone_count, axis=0)
        zone_row_indices = np.tile(np.arange(zone_count), self.background_count)
        observation_rows_for_background = observation_values[zone_row_indices]
        predictions_by_mask = np.empty((self.coalition_count, zone_count), dtype=float)

        for batch_start in range(
            0,
            self.coalition_count,
            self.coalition_batch_size,
        ):
            batch_end = min(
                self.coalition_count,
                batch_start + self.coalition_batch_size,
            )
            batch_masks = range(batch_start, batch_end)
            batch_values = np.tile(background_template, (len(batch_masks), 1))
            for local_idx, coalition_mask in enumerate(batch_masks):
                included_indices = self._included_indices_by_mask[coalition_mask]
                if not included_indices:
                    continue
                row_slice = slice(
                    local_idx * rows_per_mask,
                    (local_idx + 1) * rows_per_mask,
                )
                batch_values[row_slice, included_indices] = (
                    observation_rows_for_background[:, included_indices]
                )

            batch_frame = pd.DataFrame(batch_values, columns=self.feature_names)
            for feature_name in self.categorical_feature_names:
                batch_frame[feature_name] = batch_frame[feature_name].astype(str)
            for feature_name in self.numeric_feature_names:
                batch_frame[feature_name] = pd.to_numeric(batch_frame[feature_name])

            batch_predictions = np.asarray(
                self.model.predict(batch_frame),
                dtype=float,
            ).reshape(len(batch_masks), self.background_count, zone_count)
            predictions_by_mask[batch_start:batch_end] = batch_predictions.mean(axis=1)
            if (
                progress_label is not None
                and progress_every_coalitions > 0
                and (
                    batch_end == self.coalition_count
                    or batch_end % progress_every_coalitions == 0
                )
            ):
                model_rows = batch_end * self.background_count * zone_count
                print(
                    f"{progress_label}: predicted {batch_end:,}/{self.coalition_count:,} "
                    f"coalitions ({model_rows:,} RF rows)",
                    flush=True,
                )

        return np.maximum(predictions_by_mask, 0.0)


def run_taxi_rf_exact_shap(config: TaxiExactShapConfig) -> TaxiExactShapOutputs:
    _validate_config(config)
    feature_columns = resolve_taxi_training_features(config.feature_set)
    categorical_features = tuple(
        feature_name
        for feature_name in feature_columns
        if feature_name in TAXI_CATEGORICAL_FEATURES
    )
    numeric_features = tuple(
        feature_name
        for feature_name in feature_columns
        if feature_name not in categorical_features
    )
    frame = _load_taxi_frame(config.features_path, feature_columns)
    train_frame, explain_pool = _build_time_split(
        frame,
        holdout_hours=config.holdout_hours,
    )
    if config.train_sample_rows is not None and config.train_sample_rows < len(train_frame):
        train_frame = (
            train_frame.sample(
                n=config.train_sample_rows,
                random_state=config.random_state,
            )
            .sort_values([TAXI_TIMESTAMP_COLUMN, TAXI_ZONE_ID_COLUMN])
            .reset_index(drop=True)
        )

    explain_hours = tuple(
        pd.Index(explain_pool[TAXI_TIMESTAMP_COLUMN].drop_duplicates()).sort_values()
    )
    if config.max_hours is not None:
        explain_hours = explain_hours[: config.max_hours]
    if not explain_hours:
        raise ValueError("No explanation hours remain after applying max_hours.")

    print(
        "Training RandomForestRegressor "
        f"on {len(train_frame):,} rows with {len(feature_columns)} raw features.",
        flush=True,
    )
    training_started_at = time.perf_counter()
    model = _fit_random_forest_pipeline(
        train_frame=train_frame,
        feature_columns=feature_columns,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        config=config,
    )
    print(
        f"Finished RandomForestRegressor training in {time.perf_counter() - training_started_at:.2f}s.",
        flush=True,
    )
    print(
        "Scoring holdout predictions "
        f"on {len(explain_pool):,} rows; explaining {len(explain_hours):,} hour(s).",
        flush=True,
    )
    prediction_metrics = _build_prediction_metrics(
        model=model,
        holdout_frame=explain_pool,
        feature_columns=feature_columns,
    )
    background_frame = _sample_background_frame(
        train_frame=train_frame,
        feature_columns=feature_columns,
        rows=config.background_rows,
        random_state=config.random_state,
    )
    coalition_predictor = ExactBackgroundCoalitionPredictor(
        model=model,
        feature_names=feature_columns,
        background_frame=background_frame,
        coalition_batch_size=config.coalition_batch_size,
    )
    distance_matrix_path = _resolve_distance_matrix_path(config)
    distance_matrix = load_zone_distance_matrix(distance_matrix_path)

    started_at = time.perf_counter()
    hourly_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    coalition_rows: list[dict[str, Any]] = []
    metric_values: dict[str, dict[str, float | None]] = {
        "predictive_decision_deletion_auc": {},
        "predictive_decision_insertion_auc": {},
        "predictive_decision_infidelity": {},
        "decision_decision_deletion_auc": {},
        "decision_decision_insertion_auc": {},
        "decision_decision_infidelity": {},
    }

    for hour_idx, timestamp_hour in enumerate(explain_hours, start=1):
        hour_started_at = time.perf_counter()
        hour_frame = _select_hour_frame(
            explain_pool,
            timestamp_hour=timestamp_hour,
            max_zones=config.max_zones,
        )
        timestamp_key = str(timestamp_hour)
        coalition_count = coalition_predictor.coalition_count
        rf_rows_per_hour = coalition_count * config.background_rows * len(hour_frame)
        print(
            f"[{hour_idx}/{len(explain_hours)}] starting {timestamp_key}: "
            f"{len(hour_frame)} zones, {coalition_count:,} coalitions, "
            f"{rf_rows_per_hour:,} RF prediction rows, "
            f"{coalition_count:,} {config.orienteering_method} route solves",
            flush=True,
        )
        if config.orienteering_method == "exact":
            print(
                "Exact orienteering runs a Gurobi optimization for every coalition; "
                "use only tiny max-zones/max-hours settings unless you are intentionally "
                "waiting a long time.",
                flush=True,
            )
        coalition_score_matrix = coalition_predictor.predict_all_coalitions(
            hour_frame.loc[:, list(feature_columns)],
            progress_label=f"[{hour_idx}/{len(explain_hours)}] predictive SHAP {timestamp_key}",
            progress_every_coalitions=config.progress_every_coalitions,
        )
        zone_ids = tuple(int(zone_id) for zone_id in hour_frame[TAXI_ZONE_ID_COLUMN])
        true_scores = {
            int(row.zone_id): float(row.target_pickup_count_next_hour)
            for row in hour_frame[
                [TAXI_ZONE_ID_COLUMN, TAXI_TARGET_COLUMN]
            ].itertuples(index=False, name="TaxiScore")
        }
        predictive_values = coalition_score_matrix.sum(axis=1)
        decision_values, full_route, baseline_route = _solve_decision_values(
            coalition_score_matrix=coalition_score_matrix,
            zone_ids=zone_ids,
            true_scores=true_scores,
            config=config,
            distance_matrix=distance_matrix,
            progress_label=f"[{hour_idx}/{len(explain_hours)}] decision SHAP {timestamp_key}",
        )
        oracle_route = solve_orienteering(
            zone_scores=true_scores,
            max_distance_budget=config.max_distance_budget,
            distance_matrix=distance_matrix,
            start_zone_id=config.start_zone_id,
            end_zone_id=config.end_zone_id,
            method=config.orienteering_method,
            solver_params=_build_orienteering_solver_params(config),
        )

        predictive_shap = compute_exact_shapley_values(
            predictive_values,
            feature_count=len(feature_columns),
        )
        decision_game_values = decision_values - decision_values[0]
        decision_shap = compute_exact_shapley_values(
            decision_game_values,
            feature_count=len(feature_columns),
        )
        predictive_auc = compute_decision_insertion_auc(
            predictive_shap,
            decision_game_values,
            feature_columns,
        )
        predictive_deletion_auc = compute_decision_deletion_auc(
            predictive_shap,
            decision_game_values,
            feature_columns,
        )
        predictive_infidelity = compute_exact_decision_infidelity(
            predictive_shap,
            decision_game_values,
            feature_columns,
        )
        decision_auc = compute_decision_insertion_auc(
            decision_shap,
            decision_game_values,
            feature_columns,
        )
        decision_deletion_auc = compute_decision_deletion_auc(
            decision_shap,
            decision_game_values,
            feature_columns,
        )
        decision_infidelity = compute_exact_decision_infidelity(
            decision_shap,
            decision_game_values,
            feature_columns,
        )
        metric_values["predictive_decision_deletion_auc"][timestamp_key] = (
            predictive_deletion_auc
        )
        metric_values["predictive_decision_insertion_auc"][timestamp_key] = predictive_auc
        metric_values["predictive_decision_infidelity"][timestamp_key] = predictive_infidelity
        metric_values["decision_decision_deletion_auc"][timestamp_key] = (
            decision_deletion_auc
        )
        metric_values["decision_decision_insertion_auc"][timestamp_key] = decision_auc
        metric_values["decision_decision_infidelity"][timestamp_key] = decision_infidelity

        row = {
            "timestamp_hour": timestamp_key,
            "zone_count": len(zone_ids),
            "coalition_count": coalition_predictor.coalition_count,
            "background_rows": coalition_predictor.background_count,
            "predictive_baseline_total": float(predictive_values[0]),
            "predictive_full_total": float(predictive_values[-1]),
            "predictive_total_gain": float(predictive_values[-1] - predictive_values[0]),
            "decision_baseline_value": float(decision_values[0]),
            "decision_full_value": float(decision_values[-1]),
            "decision_value_gain": float(decision_game_values[-1]),
            "oracle_heuristic_value": _realized_route_value(oracle_route.visited_zone_ids, true_scores),
            "full_route_zone_ids": json.dumps(list(full_route.route_zone_ids)),
            "full_route_total_distance": float(full_route.total_distance),
            "baseline_route_zone_ids": json.dumps(list(baseline_route.route_zone_ids)),
            "predictive_decision_deletion_auc": predictive_deletion_auc,
            "predictive_decision_insertion_auc": predictive_auc,
            "predictive_decision_infidelity": predictive_infidelity,
            "decision_decision_deletion_auc": decision_deletion_auc,
            "decision_decision_insertion_auc": decision_auc,
            "decision_decision_infidelity": decision_infidelity,
            "hour_runtime_seconds": time.perf_counter() - hour_started_at,
        }
        for feature_name, shap_value in zip(
            feature_columns,
            predictive_shap,
            strict=True,
        ):
            row[f"predictive_shap_{feature_name}"] = float(shap_value)
        for feature_name, shap_value in zip(
            feature_columns,
            decision_shap,
            strict=True,
        ):
            row[f"decision_shap_{feature_name}"] = float(shap_value)
        hourly_rows.append(row)

        route_rows.extend(
            [
                _build_route_row(
                    timestamp_key,
                    "baseline_model",
                    baseline_route,
                    true_scores,
                ),
                _build_route_row(timestamp_key, "full_model", full_route, true_scores),
                _build_route_row(timestamp_key, "oracle_truth", oracle_route, true_scores),
            ]
        )
        if config.save_coalition_values:
            coalition_rows.extend(
                _build_coalition_rows(
                    timestamp_key,
                    predictive_values,
                    decision_values,
                )
            )
        print(
            f"[{hour_idx}/{len(explain_hours)}] explained {timestamp_key} "
            f"with {len(zone_ids)} zones in {time.perf_counter() - hour_started_at:.2f}s",
            flush=True,
        )

    hourly_shap = pd.DataFrame(hourly_rows)
    summary_shap = _build_summary_shap_frame(hourly_shap, feature_columns)
    full_routes = pd.DataFrame(route_rows)
    evaluation_metrics = {
        metric_name: build_metric_summary(values_by_hour)
        for metric_name, values_by_hour in metric_values.items()
    }
    run_metadata = {
        "features_path": str(config.features_path),
        "distance_matrix_path": str(distance_matrix_path),
        "feature_set": config.feature_set,
        "feature_columns": list(feature_columns),
        "categorical_features": list(categorical_features),
        "numeric_features": list(numeric_features),
        "target_column": TAXI_TARGET_COLUMN,
        "timestamp_column": TAXI_TIMESTAMP_COLUMN,
        "zone_id_column": TAXI_ZONE_ID_COLUMN,
        "model_name": "RandomForestRegressor",
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "min_samples_leaf": config.min_samples_leaf,
        "n_jobs": config.n_jobs,
        "rf_verbose": config.rf_verbose,
        "random_state": config.random_state,
        "train_rows": int(len(train_frame)),
        "holdout_rows": int(len(explain_pool)),
        "holdout_hours": int(config.holdout_hours),
        "explained_hours": [str(hour) for hour in explain_hours],
        "max_hours": config.max_hours,
        "max_zones": config.max_zones,
        "background_rows": config.background_rows,
        "coalition_count": 1 << len(feature_columns),
        "coalition_batch_size": config.coalition_batch_size,
        "progress_every_coalitions": config.progress_every_coalitions,
        "shap_method": "exact_coalition_enumeration_empirical_background",
        "uses_tree_shap": False,
        "orienteering_method": config.orienteering_method,
        "orienteering_threads": config.orienteering_threads,
        "orienteering_solver_params": _build_orienteering_solver_params(config),
        "max_distance_budget": config.max_distance_budget,
        "start_zone_id": config.start_zone_id,
        "end_zone_id": config.end_zone_id,
        "runtime_seconds": time.perf_counter() - started_at,
    }

    coalition_values = pd.DataFrame(coalition_rows) if config.save_coalition_values else None
    return TaxiExactShapOutputs(
        hourly_shap=hourly_shap,
        summary_shap=summary_shap,
        full_routes=full_routes,
        prediction_metrics=prediction_metrics,
        evaluation_metrics=evaluation_metrics,
        run_metadata=run_metadata,
        coalition_values=coalition_values,
    )


def write_taxi_exact_shap_outputs(
    outputs: TaxiExactShapOutputs,
    outdir: Path | str,
    *,
    write_plots: bool = True,
) -> None:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    outputs.hourly_shap.to_csv(outdir_path / "hourly_shap.csv", index=False)
    outputs.summary_shap.to_csv(outdir_path / "summary_shap.csv", index=False)
    outputs.full_routes.to_csv(outdir_path / "full_routes.csv", index=False)
    if outputs.coalition_values is not None:
        outputs.coalition_values.to_csv(outdir_path / "coalition_values.csv", index=False)
    with (outdir_path / "prediction_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(outputs.prediction_metrics, handle, indent=2, sort_keys=True)
    with (outdir_path / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(outputs.evaluation_metrics, handle, indent=2, sort_keys=True)
    with (outdir_path / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(outputs.run_metadata, handle, indent=2, sort_keys=True)
    if write_plots:
        write_taxi_exact_shap_plots(outputs, outdir_path / "plots")


def load_taxi_exact_shap_outputs(results_dir: Path | str) -> TaxiExactShapOutputs:
    results_dir_path = Path(results_dir)
    hourly_shap = pd.read_csv(results_dir_path / "hourly_shap.csv")
    summary_shap = pd.read_csv(results_dir_path / "summary_shap.csv")
    full_routes = pd.read_csv(results_dir_path / "full_routes.csv")
    coalition_values_path = results_dir_path / "coalition_values.csv"
    coalition_values = (
        pd.read_csv(coalition_values_path)
        if coalition_values_path.exists()
        else None
    )
    with (results_dir_path / "prediction_metrics.json").open(
        encoding="utf-8",
    ) as handle:
        prediction_metrics = json.load(handle)
    with (results_dir_path / "evaluation_metrics.json").open(
        encoding="utf-8",
    ) as handle:
        evaluation_metrics = json.load(handle)
    with (results_dir_path / "run_metadata.json").open(encoding="utf-8") as handle:
        run_metadata = json.load(handle)
    return TaxiExactShapOutputs(
        hourly_shap=hourly_shap,
        summary_shap=summary_shap,
        full_routes=full_routes,
        prediction_metrics=prediction_metrics,
        evaluation_metrics=evaluation_metrics,
        run_metadata=run_metadata,
        coalition_values=coalition_values,
    )


def write_taxi_exact_shap_plots(
    outputs: TaxiExactShapOutputs,
    plots_dir: Path | str,
) -> None:
    plots_dir_path = Path(plots_dir)
    plots_dir_path.mkdir(parents=True, exist_ok=True)
    feature_names = _predictive_shap_feature_order(outputs.summary_shap)
    if outputs.hourly_shap.empty or not feature_names:
        return
    feature_value_frame, feature_value_bounds = _load_hourly_feature_value_context(
        outputs.hourly_shap,
        feature_names,
        outputs.run_metadata,
    )

    _plot_importance_comparison(
        outputs.summary_shap,
        feature_names,
        plots_dir_path / "normalized_importance_comparison.png",
    )
    _plot_signed_mean_comparison(
        outputs.summary_shap,
        feature_names,
        plots_dir_path / "signed_mean_shap_comparison.png",
    )
    _plot_method_scatter(
        outputs.summary_shap,
        plots_dir_path / "featurewise_method_correlation.png",
    )
    _plot_shap_strip(
        outputs.hourly_shap,
        feature_names,
        prefix="predictive",
        output_path=plots_dir_path / "predictive_beeswarm.png",
        feature_value_frame=feature_value_frame,
        feature_value_bounds=feature_value_bounds,
    )
    _plot_shap_strip(
        outputs.hourly_shap,
        feature_names,
        prefix="decision",
        output_path=plots_dir_path / "decision_beeswarm.png",
        feature_value_frame=feature_value_frame,
        feature_value_bounds=feature_value_bounds,
    )
    _plot_gain_over_time(
        outputs.hourly_shap,
        plots_dir_path / "gain_over_time.png",
    )


def _plot_importance_comparison(
    summary_shap: pd.DataFrame,
    feature_names: Sequence[str],
    output_path: Path,
) -> None:
    frame = _ordered_summary_shap_frame(summary_shap, feature_names)
    predictive = frame["predictive_mean_abs_shap"].to_numpy(dtype=float)
    decision = frame["decision_mean_abs_shap"].to_numpy(dtype=float)
    predictive_total = float(predictive.sum())
    decision_total = float(decision.sum())
    predictive_share = predictive / predictive_total if predictive_total > 0 else predictive
    decision_share = decision / decision_total if decision_total > 0 else decision

    y_positions = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(10, _feature_plot_height(len(frame))))
    ax.barh(y_positions - 0.18, predictive_share, height=0.36, label="Predictive")
    ax.barh(y_positions + 0.18, decision_share, height=0.36, label="Decision")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(frame["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Share of mean absolute SHAP")
    ax.set_title("Normalized SHAP importance")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_signed_mean_comparison(
    summary_shap: pd.DataFrame,
    feature_names: Sequence[str],
    output_path: Path,
) -> None:
    frame = _ordered_summary_shap_frame(summary_shap, feature_names)
    y_positions = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(10, _feature_plot_height(len(frame))))
    ax.barh(
        y_positions - 0.18,
        frame["predictive_mean_signed_shap"],
        height=0.36,
        label="Predictive",
    )
    ax.barh(
        y_positions + 0.18,
        frame["decision_mean_signed_shap"],
        height=0.36,
        label="Decision",
    )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(frame["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean signed SHAP")
    ax.set_title("Average directional contribution")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_method_scatter(
    summary_shap: pd.DataFrame,
    output_path: Path,
) -> None:
    predictive = summary_shap["predictive_mean_abs_shap"].to_numpy(dtype=float)
    decision = summary_shap["decision_mean_abs_shap"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(predictive, decision, s=36)
    for feature, predictive_value, decision_value in zip(
        summary_shap["feature"],
        predictive,
        decision,
        strict=True,
    ):
        ax.annotate(
            str(feature),
            (float(predictive_value), float(decision_value)),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    lower = float(min(np.min(predictive), np.min(decision), 0.0))
    upper = float(max(np.max(predictive), np.max(decision), 1.0))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Predictive mean absolute SHAP")
    ax.set_ylabel("Decision mean absolute SHAP")
    ax.set_title("Featurewise predictive vs decision attribution")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _predictive_shap_feature_order(summary_shap: pd.DataFrame) -> tuple[str, ...]:
    if "feature" not in summary_shap.columns:
        raise KeyError("summary_shap must contain a feature column.")

    frame = summary_shap.loc[:, ["feature"]].copy()
    frame["_input_order"] = np.arange(len(frame))
    if "predictive_rank" in summary_shap.columns:
        frame["sort_key"] = pd.to_numeric(
            summary_shap["predictive_rank"],
            errors="raise",
        )
        ascending = [True, True]
    else:
        frame["sort_key"] = pd.to_numeric(
            summary_shap["predictive_mean_abs_shap"],
            errors="raise",
        )
        ascending = [False, True]

    return tuple(
        frame.sort_values(["sort_key", "_input_order"], ascending=ascending)[
            "feature"
        ].astype(str)
    )


def _ordered_summary_shap_frame(
    summary_shap: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    missing_features = sorted(
        set(feature_names) - set(summary_shap["feature"].astype(str))
    )
    if missing_features:
        raise KeyError(
            "summary_shap is missing requested features: "
            + ", ".join(missing_features)
        )

    frame = summary_shap.copy()
    frame["feature"] = frame["feature"].astype(str)
    return frame.set_index("feature").loc[list(feature_names)].reset_index()


def _load_hourly_feature_value_context(
    hourly_shap: pd.DataFrame,
    feature_names: Sequence[str],
    run_metadata: dict[str, Any],
) -> tuple[pd.DataFrame | None, dict[str, tuple[float, float]]]:
    features_path_value = run_metadata.get("features_path")
    if not isinstance(features_path_value, str) or not features_path_value:
        return None, {}

    features_path = Path(features_path_value)
    if not features_path.exists():
        return None, {}

    timestamp_column = str(run_metadata.get("timestamp_column", TAXI_TIMESTAMP_COLUMN))
    max_zones_value = run_metadata.get("max_zones")
    max_zones = None if max_zones_value is None else int(max_zones_value)
    required_columns = {timestamp_column, *feature_names}
    if max_zones is not None:
        required_columns.add(TAXI_ZONE_SELECTION_COLUMN)

    feature_frame = pd.read_csv(
        features_path,
        usecols=lambda column_name: column_name in required_columns,
        parse_dates=[timestamp_column],
    )
    missing_columns = sorted({timestamp_column, *feature_names} - set(feature_frame.columns))
    if missing_columns:
        return None, {}

    all_hourly_values = _build_hourly_feature_value_summaries(
        feature_frame,
        feature_names,
        timestamp_column=timestamp_column,
        max_zones=max_zones,
    )
    requested_hours = pd.DataFrame(
        {
            "timestamp_hour": pd.to_datetime(
                hourly_shap["timestamp_hour"],
                errors="raise",
            )
        }
    )
    feature_value_frame = requested_hours.merge(
        all_hourly_values,
        on="timestamp_hour",
        how="left",
        validate="many_to_one",
    )
    feature_value_bounds = _build_feature_value_bounds(all_hourly_values, feature_names)
    return feature_value_frame, feature_value_bounds


def _build_hourly_feature_value_summaries(
    feature_frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    timestamp_column: str,
    max_zones: int | None,
) -> pd.DataFrame:
    summary_rows: list[dict[str, float | pd.Timestamp]] = []
    for timestamp_hour, hour_frame in feature_frame.groupby(timestamp_column, sort=True):
        hour_summary_frame = _select_feature_summary_hour_frame(hour_frame, max_zones)
        row: dict[str, float | pd.Timestamp] = {"timestamp_hour": cast(pd.Timestamp, timestamp_hour)}
        for feature_name in feature_names:
            numeric_values = pd.to_numeric(
                hour_summary_frame[feature_name],
                errors="coerce",
            )
            row[feature_name] = (
                float(numeric_values.mean())
                if numeric_values.notna().any()
                else float("nan")
            )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows, columns=["timestamp_hour", *feature_names])


def _select_feature_summary_hour_frame(
    hour_frame: pd.DataFrame,
    max_zones: int | None,
) -> pd.DataFrame:
    if (
        max_zones is None
        or max_zones >= len(hour_frame)
        or TAXI_ZONE_SELECTION_COLUMN not in hour_frame
    ):
        return hour_frame

    selection_frame = hour_frame.copy()
    selection_frame["_selection_pickup_count"] = pd.to_numeric(
        selection_frame[TAXI_ZONE_SELECTION_COLUMN],
        errors="coerce",
    )
    if selection_frame["_selection_pickup_count"].notna().any():
        selection_frame = selection_frame.nlargest(
            max_zones,
            "_selection_pickup_count",
        )
    return selection_frame.drop(columns=["_selection_pickup_count"])


def _build_feature_value_bounds(
    feature_value_frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for feature_name in feature_names:
        values = pd.to_numeric(
            feature_value_frame[feature_name],
            errors="coerce",
        ).to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue
        vmin, vmax = np.nanpercentile(
            finite_values,
            FEATURE_VALUE_CLIP_PERCENTILES,
        )
        if np.isclose(vmin, vmax):
            vmin = np.nanmin(finite_values)
            vmax = np.nanmax(finite_values)
        bounds[feature_name] = (float(vmin), float(vmax))
    return bounds


def _normalize_feature_values(
    values: Sequence[float] | np.ndarray,
    bounds: tuple[float, float] | None,
) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    normalized = np.full(values_array.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(values_array)
    if not finite_mask.any():
        return normalized

    if bounds is None:
        finite_values = values_array[finite_mask]
        vmin, vmax = np.nanpercentile(finite_values, FEATURE_VALUE_CLIP_PERCENTILES)
        if np.isclose(vmin, vmax):
            vmin = np.nanmin(finite_values)
            vmax = np.nanmax(finite_values)
    else:
        vmin, vmax = bounds

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return normalized
    if np.isclose(vmin, vmax):
        normalized[finite_mask] = 0.5
        return normalized

    clipped = np.clip(values_array[finite_mask], vmin, vmax)
    normalized[finite_mask] = (clipped - vmin) / (vmax - vmin)
    return normalized


def _plot_shap_strip(
    hourly_shap: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    prefix: str,
    output_path: Path,
    feature_value_frame: pd.DataFrame | None = None,
    feature_value_bounds: dict[str, tuple[float, float]] | None = None,
) -> None:
    columns = [f"{prefix}_shap_{feature_name}" for feature_name in feature_names]
    missing_columns = sorted(set(columns) - set(hourly_shap.columns))
    if missing_columns:
        raise KeyError("Missing SHAP columns for plotting: " + ", ".join(missing_columns))

    order = tuple(feature_names)
    fig, ax = plt.subplots(figsize=(10, _feature_plot_height(len(order))))
    color_norm = Normalize(vmin=0.0, vmax=1.0)
    plotted_feature_values = False
    for y_position, feature_name in enumerate(order):
        values = hourly_shap[f"{prefix}_shap_{feature_name}"].to_numpy(dtype=float)
        jitter = (
            np.linspace(-0.16, 0.16, num=len(values))
            if len(values) > 1
            else np.zeros(len(values))
        )
        color_values = np.full(values.shape, np.nan, dtype=float)
        if (
            feature_value_frame is not None
            and len(feature_value_frame) == len(values)
            and feature_name in feature_value_frame.columns
        ):
            raw_color_values = feature_value_frame[feature_name].to_numpy(dtype=float)
            color_values = _normalize_feature_values(
                raw_color_values,
                (
                    feature_value_bounds or {}
                ).get(feature_name),
            )
        finite_color_mask = np.isfinite(color_values)
        if finite_color_mask.any():
            plotted_feature_values = True
            ax.scatter(
                values[finite_color_mask],
                y_position + jitter[finite_color_mask],
                c=color_values[finite_color_mask],
                cmap=BEESWARM_CMAP,
                norm=color_norm,
                s=24,
                alpha=0.78,
                edgecolors="none",
            )
        if (~finite_color_mask).any():
            ax.scatter(
                values[~finite_color_mask],
                y_position + jitter[~finite_color_mask],
                color=MISSING_FEATURE_COLOR,
                s=24,
                alpha=0.78,
                edgecolors="none",
            )
    if plotted_feature_values:
        scalar_mappable = plt.cm.ScalarMappable(cmap=BEESWARM_CMAP, norm=color_norm)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02)
        colorbar.set_ticks([0.0, 1.0])
        colorbar.set_ticklabels(["Low", "High"])
        colorbar.set_label("Feature value")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel(f"{prefix.title()} SHAP")
    ax.set_title(f"{prefix.title()} SHAP distribution")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_gain_over_time(
    hourly_shap: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = hourly_shap.copy()
    frame["timestamp_hour"] = pd.to_datetime(frame["timestamp_hour"])
    frame = frame.sort_values("timestamp_hour")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        frame["timestamp_hour"],
        frame["predictive_total_gain"],
        marker="o",
        label="Predictive total gain",
    )
    ax.plot(
        frame["timestamp_hour"],
        frame["decision_value_gain"],
        marker="o",
        label="Decision value gain",
    )
    ax.set_xlabel("Hour")
    ax.set_ylabel("Gain")
    ax.set_title("Predictive and decision gain over explained hours")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _feature_plot_height(feature_count: int) -> float:
    return max(4.0, 0.36 * feature_count + 1.4)


def _validate_config(config: TaxiExactShapConfig) -> None:
    if config.holdout_hours <= 0:
        raise ValueError("holdout_hours must be strictly positive.")
    if config.max_hours is not None and config.max_hours <= 0:
        raise ValueError("max_hours must be strictly positive when provided.")
    if config.max_zones is not None and config.max_zones <= 0:
        raise ValueError("max_zones must be strictly positive when provided.")
    if config.background_rows <= 0:
        raise ValueError("background_rows must be strictly positive.")
    if config.progress_every_coalitions < 0:
        raise ValueError("progress_every_coalitions must be non-negative.")
    if config.n_estimators <= 0:
        raise ValueError("n_estimators must be strictly positive.")
    if config.min_samples_leaf <= 0:
        raise ValueError("min_samples_leaf must be strictly positive.")
    if config.train_sample_rows is not None and config.train_sample_rows <= 0:
        raise ValueError("train_sample_rows must be strictly positive when provided.")
    if config.orienteering_method not in {"exact", "heuristic", "ortools"}:
        raise ValueError("orienteering_method must be one of: exact, heuristic, ortools.")
    if config.orienteering_threads <= 0:
        raise ValueError("orienteering_threads must be strictly positive.")


def _load_taxi_frame(
    features_path: Path,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    required_columns = list(
        dict.fromkeys(
            [
                TAXI_TIMESTAMP_COLUMN,
                TAXI_ZONE_ID_COLUMN,
                *feature_columns,
                TAXI_ZONE_SELECTION_COLUMN,
                TAXI_TARGET_COLUMN,
            ]
        )
    )
    frame = pd.read_csv(
        features_path,
        usecols=required_columns,
        parse_dates=[TAXI_TIMESTAMP_COLUMN],
    )
    frame = frame.dropna(subset=required_columns).copy()
    frame[TAXI_ZONE_ID_COLUMN] = frame[TAXI_ZONE_ID_COLUMN].astype(int)
    for feature_name in feature_columns:
        if feature_name in TAXI_CATEGORICAL_FEATURES:
            frame[feature_name] = frame[feature_name].astype(str)
        else:
            frame[feature_name] = pd.to_numeric(frame[feature_name])
    frame[TAXI_ZONE_SELECTION_COLUMN] = pd.to_numeric(
        frame[TAXI_ZONE_SELECTION_COLUMN]
    )
    frame[TAXI_TARGET_COLUMN] = pd.to_numeric(frame[TAXI_TARGET_COLUMN])
    return frame.sort_values([TAXI_TIMESTAMP_COLUMN, TAXI_ZONE_ID_COLUMN]).reset_index(
        drop=True
    )


def _build_time_split(
    frame: pd.DataFrame,
    *,
    holdout_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_hours = pd.Index(frame[TAXI_TIMESTAMP_COLUMN].drop_duplicates()).sort_values()
    if holdout_hours >= len(unique_hours):
        raise ValueError(
            f"holdout_hours={holdout_hours} must be smaller than the "
            f"{len(unique_hours)} available hours."
        )
    holdout_hour_index = unique_hours[-holdout_hours:]
    is_holdout = frame[TAXI_TIMESTAMP_COLUMN].isin(holdout_hour_index)
    train_frame = frame.loc[~is_holdout].reset_index(drop=True)
    holdout_frame = frame.loc[is_holdout].reset_index(drop=True)
    return train_frame, holdout_frame


def _fit_random_forest_pipeline(
    *,
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
    config: TaxiExactShapConfig,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_features),
            ),
            ("numeric", "passthrough", list(numeric_features)),
        ],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbose=config.rf_verbose,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(
        train_frame.loc[:, list(feature_columns)],
        train_frame.loc[:, TAXI_TARGET_COLUMN],
    )
    return pipeline


def _build_prediction_metrics(
    *,
    model: Pipeline,
    holdout_frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    y_true = holdout_frame[TAXI_TARGET_COLUMN].to_numpy(dtype=float)
    y_pred = np.asarray(
        model.predict(holdout_frame.loc[:, list(feature_columns)]),
        dtype=float,
    )
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "holdout": {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "rows": int(len(holdout_frame)),
            "hours": int(holdout_frame[TAXI_TIMESTAMP_COLUMN].nunique()),
            "predictions": int(y_true.size),
        }
    }


def _sample_background_frame(
    *,
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    rows: int,
    random_state: int,
) -> pd.DataFrame:
    if rows > len(train_frame):
        raise ValueError(
            f"Requested {rows} background rows but only {len(train_frame)} training rows are available."
        )
    return (
        train_frame.loc[:, list(feature_columns)]
        .sample(n=rows, random_state=random_state)
        .reset_index(drop=True)
    )


def _resolve_distance_matrix_path(config: TaxiExactShapConfig) -> Path:
    if config.distance_matrix_path is not None:
        return config.distance_matrix_path
    if "manhattan" in config.features_path.stem.lower():
        return DEFAULT_MANHATTAN_DISTANCE_MATRIX_PATH
    return DEFAULT_ZONE_DISTANCE_MATRIX_PATH


def _select_hour_frame(
    explain_pool: pd.DataFrame,
    *,
    timestamp_hour: pd.Timestamp,
    max_zones: int | None,
) -> pd.DataFrame:
    hour_frame = explain_pool.loc[
        explain_pool[TAXI_TIMESTAMP_COLUMN] == timestamp_hour
    ].copy()
    if hour_frame.empty:
        raise ValueError(f"No rows found for explanation hour {timestamp_hour}.")
    if max_zones is not None and max_zones < len(hour_frame):
        hour_frame = hour_frame.nlargest(max_zones, TAXI_ZONE_SELECTION_COLUMN)
    return hour_frame.sort_values(TAXI_ZONE_ID_COLUMN).reset_index(drop=True)


def _solve_decision_values(
    *,
    coalition_score_matrix: np.ndarray,
    zone_ids: Sequence[int],
    true_scores: dict[int, float],
    config: TaxiExactShapConfig,
    distance_matrix: pd.DataFrame,
    progress_label: str | None = None,
) -> tuple[np.ndarray, Any, Any]:
    decision_values = np.empty(coalition_score_matrix.shape[0], dtype=float)
    full_route = None
    baseline_route = None
    full_mask = coalition_score_matrix.shape[0] - 1
    for coalition_mask, predicted_scores in enumerate(coalition_score_matrix):
        zone_scores = {
            zone_id: float(score)
            for zone_id, score in zip(zone_ids, predicted_scores, strict=True)
            if score > 0.0
        }
        route = solve_orienteering(
            zone_scores=zone_scores,
            max_distance_budget=config.max_distance_budget,
            distance_matrix=distance_matrix,
            start_zone_id=config.start_zone_id,
            end_zone_id=config.end_zone_id,
            method=config.orienteering_method,
            solver_params=_build_orienteering_solver_params(config),
        )
        decision_values[coalition_mask] = _realized_route_value(
            route.visited_zone_ids,
            true_scores,
        )
        if coalition_mask == 0:
            baseline_route = route
        if coalition_mask == full_mask:
            full_route = route
        completed_count = coalition_mask + 1
        if (
            progress_label is not None
            and config.progress_every_coalitions > 0
            and (
                completed_count == len(coalition_score_matrix)
                or completed_count % config.progress_every_coalitions == 0
            )
        ):
            print(
                f"{progress_label}: solved {completed_count:,}/{len(coalition_score_matrix):,} "
                f"{config.orienteering_method} routes",
                flush=True,
            )
    if full_route is None or baseline_route is None:
        raise RuntimeError("Expected baseline and full-coalition routes.")
    return decision_values, full_route, baseline_route


def _build_orienteering_solver_params(
    config: TaxiExactShapConfig,
) -> dict[str, float | int | str] | None:
    if config.orienteering_method == "exact":
        return {"Threads": config.orienteering_threads}
    if config.orienteering_method == "ortools":
        return {
            "time_limit_s": DEFAULT_ORTOOLS_TIME_LIMIT_S,
            "local_search_metaheuristic": DEFAULT_ORTOOLS_LOCAL_SEARCH_METAHEURISTIC,
        }
    return None


def _realized_route_value(
    visited_zone_ids: Sequence[int],
    true_scores: dict[int, float],
) -> float:
    return float(sum(true_scores.get(int(zone_id), 0.0) for zone_id in visited_zone_ids))


def _build_route_row(
    timestamp_hour: str,
    route_type: str,
    route: Any,
    true_scores: dict[int, float],
) -> dict[str, Any]:
    return {
        "timestamp_hour": timestamp_hour,
        "route_type": route_type,
        "method": route.method,
        "route_zone_ids": json.dumps(list(route.route_zone_ids)),
        "visited_zone_ids": json.dumps(list(route.visited_zone_ids)),
        "visited_zone_count": len(route.visited_zone_ids),
        "model_collected_score": float(route.collected_score),
        "realized_pickup_value": _realized_route_value(
            route.visited_zone_ids,
            true_scores,
        ),
        "total_distance": float(route.total_distance),
        "remaining_budget": float(route.remaining_budget),
        "optimal": bool(route.optimal),
    }


def _build_coalition_rows(
    timestamp_hour: str,
    predictive_values: np.ndarray,
    decision_values: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "timestamp_hour": timestamp_hour,
            "coalition_mask": coalition_mask,
            "subset_size": int(coalition_mask.bit_count()),
            "predictive_value": float(predictive_values[coalition_mask]),
            "decision_value": float(decision_values[coalition_mask]),
            "decision_characteristic_value": float(
                decision_values[coalition_mask] - decision_values[0]
            ),
        }
        for coalition_mask in range(len(predictive_values))
    ]


def _build_summary_shap_frame(
    hourly_shap: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    predictive_abs_means: dict[str, float] = {}
    decision_abs_means: dict[str, float] = {}
    for feature_name in feature_names:
        predictive_column = hourly_shap[f"predictive_shap_{feature_name}"]
        decision_column = hourly_shap[f"decision_shap_{feature_name}"]
        predictive_abs_means[feature_name] = float(predictive_column.abs().mean())
        decision_abs_means[feature_name] = float(decision_column.abs().mean())

    predictive_ranks = _descending_rank_map(predictive_abs_means)
    decision_ranks = _descending_rank_map(decision_abs_means)
    for feature_name in feature_names:
        predictive_column = hourly_shap[f"predictive_shap_{feature_name}"]
        decision_column = hourly_shap[f"decision_shap_{feature_name}"]
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


def _descending_rank_map(values_by_feature: dict[str, float]) -> dict[str, int]:
    ranks = (
        pd.Series(values_by_feature, dtype=float)
        .rank(method="dense", ascending=False)
        .to_dict()
    )
    return {feature_name: int(rank) for feature_name, rank in ranks.items()}
