from __future__ import annotations

import argparse
import ast
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy import stats

from dva.analysis.ems_exact_shap import (
    EMS_TIMESTAMP_COLUMN,
    _load_zone_order,
    _target_columns,
    build_coverage_matrix,
)
from dva.case_studies.ems.models import EMS_XGB_MODEL_IDS


DEFAULT_RESULTS_ROOT = Path("results/ems/experiment_a_infodva")
DEFAULT_OUTDIR = Path("results/ems/diagnostics")
DEFAULT_DISTANCE_MATRIX_PATH = Path(
    "data/ems_data/processed/ems_zip_centroid_distance_matrix_km.parquet"
)
DEFAULT_Y_PATH = Path("data/ems_data/processed/ems_zip_hour_features_2025_manhattan_wide_y.csv")
DEFAULT_ZONE_ORDER_PATH = Path("data/ems_data/processed/ems_zip_wide_zone_order.csv")
DEFAULT_REGIMES = ((1.0, 3), (1.0, 5), (1.0, 8), (2.0, 5))
RECONSTRUCTION_REGIMES = (
    (1.0, 3),
    (1.0, 5),
    (1.0, 8),
    (2.0, 3),
    (2.0, 5),
    (2.0, 8),
    (3.0, 3),
    (3.0, 5),
    (3.0, 8),
)


@dataclass(frozen=True)
class PredictionBundle:
    model_id: str
    zip_codes: tuple[str, ...]
    target_columns: tuple[str, ...]
    y_true: np.ndarray
    yhat_full: np.ndarray
    timestamps: tuple[str, ...]
    reconstruction_rank_min: int
    reconstruction_max_abs_residual: float
    reconstruction_mean_abs_residual: float
    reconstruction_total_max_abs_diff: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose why EMS ante/pre InfoDVA is lower than post Decision-DVA "
            "using saved exact-SHAP EMS outputs."
        )
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--distance-matrix-path",
        type=Path,
        default=DEFAULT_DISTANCE_MATRIX_PATH,
    )
    parser.add_argument("--y-path", type=Path, default=DEFAULT_Y_PATH)
    parser.add_argument("--zone-order-path", type=Path, default=DEFAULT_ZONE_ORDER_PATH)
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        help="Optional model id to include. Repeatable. Defaults to all 25 EMS XGB models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_ids = tuple(args.model_ids or EMS_XGB_MODEL_IDS)
    args.outdir.mkdir(parents=True, exist_ok=True)

    prediction_cache: dict[str, PredictionBundle] = {}
    model_regime_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    for model_idx, model_id in enumerate(model_ids, start=1):
        print(f"[{model_idx}/{len(model_ids)}] reconstructing saved full predictions for {model_id}", flush=True)
        prediction_cache[model_id] = reconstruct_full_prediction_bundle(
            model_id=model_id,
            results_root=args.results_root,
            distance_matrix_path=args.distance_matrix_path,
            y_path=args.y_path,
            zone_order_path=args.zone_order_path,
        )
        for tau, p in DEFAULT_REGIMES:
            run_dir = run_dir_for(args.results_root, model_id, tau, p)
            print(f"  reading diagnostics for tau={tau:g}, p={p}", flush=True)
            model_regime_rows.append(
                summarize_model_regime(
                    model_id=model_id,
                    tau=tau,
                    p=p,
                    run_dir=run_dir,
                    predictions=prediction_cache[model_id],
                    distance_matrix_path=args.distance_matrix_path,
                )
            )
            edge_rows.append(
                summarize_model_regime_edges(
                    model_id=model_id,
                    tau=tau,
                    p=p,
                    run_dir=run_dir,
                    predictions=prediction_cache[model_id],
                    distance_matrix_path=args.distance_matrix_path,
                )
            )

    model_regime = pd.DataFrame(model_regime_rows)
    edge_summary = pd.DataFrame(edge_rows)
    regime_summary = summarize_regimes(model_regime, edge_summary)

    model_regime_path = args.outdir / "ems_pre_post_gap_diagnostics_by_model_regime.csv"
    edge_summary_path = args.outdir / "ems_pre_post_gap_edge_diagnostics_by_model_regime.csv"
    regime_summary_path = args.outdir / "ems_pre_post_gap_diagnostics_by_regime.csv"
    model_regime.to_csv(model_regime_path, index=False)
    edge_summary.to_csv(edge_summary_path, index=False)
    regime_summary.to_csv(regime_summary_path, index=False)

    print("\nRegime summary")
    print(regime_summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nWrote {regime_summary_path}")
    print(f"Wrote {model_regime_path}")
    print(f"Wrote {edge_summary_path}")


def run_dir_for(results_root: Path, model_id: str, tau: float, p: int) -> Path:
    tau_label = f"{tau:g}".replace(".", "p")
    path = (
        results_root
        / model_id
        / "models"
        / model_id
        / "runs"
        / f"ems_{model_id}_exact_radius_{tau_label}km_budget_{p}"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing EMS run directory: {path}")
    return path


def reconstruct_full_prediction_bundle(
    *,
    model_id: str,
    results_root: Path,
    distance_matrix_path: Path,
    y_path: Path,
    zone_order_path: Path,
) -> PredictionBundle:
    y_frame = pd.read_csv(y_path)
    zone_order = _load_zone_order(zone_order_path, tuple(_target_columns(y_frame)))
    target_columns = tuple(zone_order["target_column"].astype(str))
    zip_codes = tuple(zone_order["zip_code"].astype(str))

    reference_run_dir = run_dir_for(results_root, model_id, DEFAULT_REGIMES[0][0], DEFAULT_REGIMES[0][1])
    reference_hourly = pd.read_csv(reference_run_dir / "hourly_shap.csv")
    timestamps = tuple(str(value) for value in reference_hourly[EMS_TIMESTAMP_COLUMN])
    y_frame[EMS_TIMESTAMP_COLUMN] = y_frame[EMS_TIMESTAMP_COLUMN].astype(str)
    y_by_timestamp = y_frame.set_index(EMS_TIMESTAMP_COLUMN).loc[
        list(timestamps),
        list(target_columns),
    ]
    y_true = y_by_timestamp.to_numpy(dtype=float, copy=True)

    distance_matrix = pd.read_parquet(distance_matrix_path)
    coverage_by_tau = {
        tau: build_coverage_matrix(
            distance_matrix,
            zip_codes,
            coverage_radius_km=tau,
        )
        for tau in sorted({tau for tau, _ in RECONSTRUCTION_REGIMES})
    }
    coalition_by_regime = {
        (tau, p): pd.read_csv(
            run_dir_for(results_root, model_id, tau, p) / "coalition_values.csv"
        ).assign(**{EMS_TIMESTAMP_COLUMN: lambda frame: frame[EMS_TIMESTAMP_COLUMN].astype(str)})
        for tau, p in RECONSTRUCTION_REGIMES
    }

    yhat_rows: list[np.ndarray] = []
    ranks: list[int] = []
    max_abs_residuals: list[float] = []
    mean_abs_residuals: list[float] = []
    total_abs_diffs: list[float] = []
    for timestamp in timestamps:
        yhat, diagnostics = reconstruct_hour_yhat(
            timestamp=timestamp,
            zip_count=len(zip_codes),
            coverage_by_tau=coverage_by_tau,
            coalition_by_regime=coalition_by_regime,
        )
        yhat_rows.append(yhat)
        ranks.append(diagnostics["rank"])
        max_abs_residuals.append(diagnostics["max_abs_residual"])
        mean_abs_residuals.append(diagnostics["mean_abs_residual"])
        total_abs_diffs.append(diagnostics["total_abs_diff"])

    return PredictionBundle(
        model_id=model_id,
        zip_codes=zip_codes,
        target_columns=target_columns,
        y_true=y_true,
        yhat_full=np.vstack(yhat_rows),
        timestamps=timestamps,
        reconstruction_rank_min=int(min(ranks)),
        reconstruction_max_abs_residual=float(max(max_abs_residuals)),
        reconstruction_mean_abs_residual=float(np.mean(mean_abs_residuals)),
        reconstruction_total_max_abs_diff=float(max(total_abs_diffs)),
    )


def reconstruct_hour_yhat(
    *,
    timestamp: str,
    zip_count: int,
    coverage_by_tau: dict[float, np.ndarray],
    coalition_by_regime: dict[tuple[float, int], pd.DataFrame],
) -> tuple[np.ndarray, dict[str, float | int]]:
    equation_rows: list[np.ndarray] = []
    targets: list[float] = []
    full_totals: list[float] = []
    full_mask_values: list[int] = []

    for (tau, _p), coalition in coalition_by_regime.items():
        group = coalition.loc[coalition[EMS_TIMESTAMP_COLUMN].eq(timestamp)].sort_values(
            "coalition_mask"
        )
        if group.empty:
            raise ValueError(f"Missing coalition rows for {timestamp}")
        full_mask = int(group["coalition_mask"].max())
        full_mask_values.append(full_mask)
        full_total = float(
            group.loc[group["coalition_mask"].eq(full_mask), "predictive_value"].iloc[0]
        )
        full_totals.append(full_total)
        coverage_matrix = coverage_by_tau[tau]
        for row in group.itertuples(index=False):
            selected_indices = tuple(
                int(index)
                for index in parse_json_list(
                    getattr(row, "decision_selected_facility_indices")
                )
            )
            equation_rows.append(coverage_vector(coverage_matrix, selected_indices).astype(float))
            targets.append(float(getattr(row, "ante_decision_value")) * full_total)

    if len(set(full_mask_values)) != 1:
        raise ValueError(f"Inconsistent full coalition masks for {timestamp}: {full_mask_values}")
    full_total_array = np.asarray(full_totals, dtype=float)
    full_total = float(np.mean(full_total_array))
    equation_rows.append(np.ones(zip_count, dtype=float))
    targets.append(full_total)
    matrix = np.vstack(equation_rows)
    target = np.asarray(targets, dtype=float)

    fit = lsq_linear(matrix, target, bounds=(0.0, np.inf), tol=1e-12, lsmr_tol="auto")
    yhat = np.maximum(np.asarray(fit.x, dtype=float), 0.0)
    residual = matrix @ yhat - target
    return yhat, {
        "rank": int(np.linalg.matrix_rank(matrix)),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "mean_abs_residual": float(np.mean(np.abs(residual))),
        "total_abs_diff": float(abs(yhat.sum() - full_total)),
    }


def summarize_model_regime(
    *,
    model_id: str,
    tau: float,
    p: int,
    run_dir: Path,
    predictions: PredictionBundle,
    distance_matrix_path: Path,
) -> dict[str, Any]:
    hourly = pd.read_csv(run_dir / "hourly_shap.csv")
    hourly[EMS_TIMESTAMP_COLUMN] = hourly[EMS_TIMESTAMP_COLUMN].astype(str)
    assert_timestamps_match(hourly[EMS_TIMESTAMP_COLUMN], predictions.timestamps, run_dir)

    smoothness = smoothness_metrics(predictions.y_true, predictions.yhat_full)
    coverage_alignment = full_null_alignment(
        hourly=hourly,
        predictions=predictions,
        distance_matrix_path=distance_matrix_path,
        tau=tau,
    )
    return {
        "model_id": model_id,
        "tau": tau,
        "p": p,
        "hours": int(len(hourly)),
        "hhi_y_mean": smoothness["hhi_y_mean"],
        "hhi_yhat_mean": smoothness["hhi_yhat_mean"],
        "hhi_ratio_y_over_yhat": safe_div(
            smoothness["hhi_y_mean"],
            smoothness["hhi_yhat_mean"],
        ),
        "top5_y_mean": smoothness["top5_y_mean"],
        "top5_yhat_mean": smoothness["top5_yhat_mean"],
        "top5_gap_y_minus_yhat": smoothness["top5_y_mean"] - smoothness["top5_yhat_mean"],
        "gini_y_mean": smoothness["gini_y_mean"],
        "gini_yhat_mean": smoothness["gini_yhat_mean"],
        "entropy_y_mean": smoothness["entropy_y_mean"],
        "entropy_yhat_mean": smoothness["entropy_yhat_mean"],
        "pre_null_mean": float(hourly["ante_decision_baseline_value"].mean()),
        "pre_full_mean": float(hourly["ante_decision_full_value"].mean()),
        "pre_gap_mean": float(hourly["ante_decision_value_gain"].mean()),
        "pre_gap_abs_mean": float(hourly["ante_decision_value_gain"].abs().mean()),
        "pre_gap_share_of_full": safe_div(
            float(hourly["ante_decision_value_gain"].mean()),
            float(hourly["ante_decision_full_value"].mean()),
        ),
        "post_null_mean": float(hourly["decision_baseline_value"].mean()),
        "post_full_mean": float(hourly["decision_full_value"].mean()),
        "post_gap_mean": float(hourly["decision_value_gain"].mean()),
        "post_gap_abs_mean": float(hourly["decision_value_gain"].abs().mean()),
        "post_gap_share_of_full": safe_div(
            float(hourly["decision_value_gain"].mean()),
            float(hourly["decision_full_value"].mean()),
        ),
        "post_minus_pre_gap_mean": float(
            (hourly["decision_value_gain"] - hourly["ante_decision_value_gain"]).mean()
        ),
        "post_to_pre_gap_ratio": safe_div(
            float(hourly["decision_value_gain"].mean()),
            float(hourly["ante_decision_value_gain"].mean()),
        ),
        "full_null_residual_alignment_mean": coverage_alignment["alignment_mean"],
        "full_null_residual_alignment_positive_share": coverage_alignment[
            "alignment_positive_share"
        ],
        "full_null_coverage_changed_share": coverage_alignment[
            "coverage_changed_share"
        ],
        "full_null_selected_changed_share": selected_change_share(
            hourly["baseline_selected_zip_codes"],
            hourly["full_selected_zip_codes"],
        ),
        "yhat_reconstruction_rank_min": predictions.reconstruction_rank_min,
        "yhat_reconstruction_max_abs_residual": (
            predictions.reconstruction_max_abs_residual
        ),
        "yhat_reconstruction_mean_abs_residual": (
            predictions.reconstruction_mean_abs_residual
        ),
        "yhat_reconstruction_total_max_abs_diff": (
            predictions.reconstruction_total_max_abs_diff
        ),
    }


def summarize_model_regime_edges(
    *,
    model_id: str,
    tau: float,
    p: int,
    run_dir: Path,
    predictions: PredictionBundle,
    distance_matrix_path: Path,
) -> dict[str, Any]:
    coalition = pd.read_csv(run_dir / "coalition_values.csv")
    coalition[EMS_TIMESTAMP_COLUMN] = coalition[EMS_TIMESTAMP_COLUMN].astype(str)
    player_count = int(round(math.log2(coalition["coalition_mask"].nunique())))
    expected_edges_per_hour = player_count * (1 << (player_count - 1))

    distance_matrix = pd.read_parquet(distance_matrix_path)
    coverage_matrix = build_coverage_matrix(
        distance_matrix,
        predictions.zip_codes,
        coverage_radius_km=tau,
    )
    rows: list[dict[str, float]] = []

    for hour_idx, timestamp in enumerate(predictions.timestamps):
        group = coalition.loc[coalition[EMS_TIMESTAMP_COLUMN].eq(timestamp)].sort_values(
            "coalition_mask"
        )
        if len(group) != (1 << player_count):
            raise ValueError(f"{run_dir} has incomplete coalitions for {timestamp}")
        selected = [
            frozenset(parse_json_list(value)) for value in group["decision_selected_facility_indices"]
        ]
        coverage = [coverage_vector(coverage_matrix, indices) for indices in selected]
        pre_values = group["ante_decision_value"].to_numpy(dtype=float, copy=True)
        post_values = group["decision_value"].to_numpy(dtype=float, copy=True)
        y_share = normalized_share(predictions.y_true[hour_idx])
        yhat_share = normalized_share(predictions.yhat_full[hour_idx])
        residual = y_share - yhat_share

        for mask in range(1 << player_count):
            for player_idx in range(player_count):
                bit = 1 << player_idx
                if mask & bit:
                    continue
                next_mask = mask | bit
                set_left = selected[mask]
                set_right = selected[next_mask]
                decision_changed = set_left != set_right
                jaccard_distance = set_jaccard_distance(set_left, set_right)
                pre_delta = float(pre_values[next_mask] - pre_values[mask])
                post_delta = float(post_values[next_mask] - post_values[mask])
                coverage_delta = coverage[next_mask].astype(float) - coverage[mask].astype(float)
                alignment = float(np.dot(residual, coverage_delta))
                rows.append(
                    {
                        "decision_changed": float(decision_changed),
                        "jaccard_distance": jaccard_distance,
                        "hamming_distance": float(len(set_left.symmetric_difference(set_right))),
                        "abs_pre_delta": abs(pre_delta),
                        "abs_post_delta": abs(post_delta),
                        "pre_delta": pre_delta,
                        "post_delta": post_delta,
                        "post_minus_pre_delta": post_delta - pre_delta,
                        "residual_alignment": alignment,
                    }
                )

    edge_frame = pd.DataFrame(rows)
    changed = edge_frame.loc[edge_frame["decision_changed"].eq(1.0)]
    if len(edge_frame) != len(predictions.timestamps) * expected_edges_per_hour:
        raise ValueError(f"Unexpected edge count for {run_dir}: {len(edge_frame)}")

    return {
        "model_id": model_id,
        "tau": tau,
        "p": p,
        "edges": int(len(edge_frame)),
        "decision_change_rate": float(edge_frame["decision_changed"].mean()),
        "jaccard_distance_mean_all_edges": float(edge_frame["jaccard_distance"].mean()),
        "jaccard_distance_mean_changed_edges": float(changed["jaccard_distance"].mean()),
        "hamming_distance_mean_changed_edges": float(changed["hamming_distance"].mean()),
        "edge_abs_pre_delta_mean_changed": float(changed["abs_pre_delta"].mean()),
        "edge_abs_post_delta_mean_changed": float(changed["abs_post_delta"].mean()),
        "edge_abs_post_to_pre_ratio_changed": safe_div(
            float(changed["abs_post_delta"].mean()),
            float(changed["abs_pre_delta"].mean()),
        ),
        "edge_pre_delta_mean_changed": float(changed["pre_delta"].mean()),
        "edge_post_delta_mean_changed": float(changed["post_delta"].mean()),
        "edge_residual_alignment_mean_changed": float(
            changed["residual_alignment"].mean()
        ),
        "edge_residual_alignment_positive_share_changed": float(
            changed["residual_alignment"].gt(0.0).mean()
        ),
        "edge_residual_alignment_post_delta_spearman_changed": spearman(
            changed["residual_alignment"],
            changed["post_delta"],
        ),
        "edge_residual_alignment_post_minus_pre_spearman_changed": spearman(
            changed["residual_alignment"],
            changed["post_minus_pre_delta"],
        ),
    }


def summarize_regimes(model_regime: pd.DataFrame, edge_summary: pd.DataFrame) -> pd.DataFrame:
    left = (
        model_regime.groupby(["tau", "p"], as_index=False)
        .agg(
            models=("model_id", "nunique"),
            hours=("hours", "sum"),
            hhi_y_mean=("hhi_y_mean", "mean"),
            hhi_yhat_mean=("hhi_yhat_mean", "mean"),
            hhi_ratio_y_over_yhat=("hhi_ratio_y_over_yhat", "mean"),
            top5_y_mean=("top5_y_mean", "mean"),
            top5_yhat_mean=("top5_yhat_mean", "mean"),
            top5_gap_y_minus_yhat=("top5_gap_y_minus_yhat", "mean"),
            gini_y_mean=("gini_y_mean", "mean"),
            gini_yhat_mean=("gini_yhat_mean", "mean"),
            entropy_y_mean=("entropy_y_mean", "mean"),
            entropy_yhat_mean=("entropy_yhat_mean", "mean"),
            pre_null_mean=("pre_null_mean", "mean"),
            pre_full_mean=("pre_full_mean", "mean"),
            pre_gap_mean=("pre_gap_mean", "mean"),
            pre_gap_abs_mean=("pre_gap_abs_mean", "mean"),
            pre_gap_share_of_full=("pre_gap_share_of_full", "mean"),
            post_null_mean=("post_null_mean", "mean"),
            post_full_mean=("post_full_mean", "mean"),
            post_gap_mean=("post_gap_mean", "mean"),
            post_gap_abs_mean=("post_gap_abs_mean", "mean"),
            post_gap_share_of_full=("post_gap_share_of_full", "mean"),
            post_minus_pre_gap_mean=("post_minus_pre_gap_mean", "mean"),
            post_to_pre_gap_ratio=("post_to_pre_gap_ratio", "mean"),
            full_null_residual_alignment_mean=(
                "full_null_residual_alignment_mean",
                "mean",
            ),
            full_null_residual_alignment_positive_share=(
                "full_null_residual_alignment_positive_share",
                "mean",
            ),
            full_null_coverage_changed_share=("full_null_coverage_changed_share", "mean"),
            full_null_selected_changed_share=("full_null_selected_changed_share", "mean"),
            yhat_reconstruction_rank_min=("yhat_reconstruction_rank_min", "min"),
            yhat_reconstruction_max_abs_residual=(
                "yhat_reconstruction_max_abs_residual",
                "max",
            ),
            yhat_reconstruction_mean_abs_residual=(
                "yhat_reconstruction_mean_abs_residual",
                "mean",
            ),
            yhat_reconstruction_total_max_abs_diff=(
                "yhat_reconstruction_total_max_abs_diff",
                "max",
            ),
        )
    )
    right = (
        edge_summary.groupby(["tau", "p"], as_index=False)
        .agg(
            edges=("edges", "sum"),
            decision_change_rate=("decision_change_rate", "mean"),
            jaccard_distance_mean_all_edges=("jaccard_distance_mean_all_edges", "mean"),
            jaccard_distance_mean_changed_edges=(
                "jaccard_distance_mean_changed_edges",
                "mean",
            ),
            hamming_distance_mean_changed_edges=(
                "hamming_distance_mean_changed_edges",
                "mean",
            ),
            edge_abs_pre_delta_mean_changed=("edge_abs_pre_delta_mean_changed", "mean"),
            edge_abs_post_delta_mean_changed=("edge_abs_post_delta_mean_changed", "mean"),
            edge_abs_post_to_pre_ratio_changed=(
                "edge_abs_post_to_pre_ratio_changed",
                "mean",
            ),
            edge_pre_delta_mean_changed=("edge_pre_delta_mean_changed", "mean"),
            edge_post_delta_mean_changed=("edge_post_delta_mean_changed", "mean"),
            edge_residual_alignment_mean_changed=(
                "edge_residual_alignment_mean_changed",
                "mean",
            ),
            edge_residual_alignment_positive_share_changed=(
                "edge_residual_alignment_positive_share_changed",
                "mean",
            ),
            edge_residual_alignment_post_delta_spearman_changed=(
                "edge_residual_alignment_post_delta_spearman_changed",
                "mean",
            ),
            edge_residual_alignment_post_minus_pre_spearman_changed=(
                "edge_residual_alignment_post_minus_pre_spearman_changed",
                "mean",
            ),
        )
    )
    return left.merge(right, on=["tau", "p"], how="inner").sort_values(["tau", "p"])


def smoothness_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_share = np.apply_along_axis(normalized_share, 1, y_true)
    yhat_share = np.apply_along_axis(normalized_share, 1, y_pred)
    return {
        "hhi_y_mean": float(np.mean(np.sum(y_share**2, axis=1))),
        "hhi_yhat_mean": float(np.mean(np.sum(yhat_share**2, axis=1))),
        "top5_y_mean": float(np.mean(np.sum(-np.sort(-y_share, axis=1)[:, :5], axis=1))),
        "top5_yhat_mean": float(
            np.mean(np.sum(-np.sort(-yhat_share, axis=1)[:, :5], axis=1))
        ),
        "gini_y_mean": float(np.mean([gini(row) for row in y_true])),
        "gini_yhat_mean": float(np.mean([gini(row) for row in y_pred])),
        "entropy_y_mean": float(np.mean([entropy(row) for row in y_share])),
        "entropy_yhat_mean": float(np.mean([entropy(row) for row in yhat_share])),
    }


def full_null_alignment(
    *,
    hourly: pd.DataFrame,
    predictions: PredictionBundle,
    distance_matrix_path: Path,
    tau: float,
) -> dict[str, float]:
    distance_matrix = pd.read_parquet(distance_matrix_path)
    coverage_matrix = build_coverage_matrix(
        distance_matrix,
        predictions.zip_codes,
        coverage_radius_km=tau,
    )
    alignments: list[float] = []
    coverage_changed: list[bool] = []
    for idx, row in hourly.reset_index(drop=True).iterrows():
        baseline_indices = zip_codes_to_indices(
            parse_json_list(row["baseline_selected_zip_codes"]),
            predictions.zip_codes,
        )
        full_indices = zip_codes_to_indices(
            parse_json_list(row["full_selected_zip_codes"]),
            predictions.zip_codes,
        )
        baseline_coverage = coverage_vector(coverage_matrix, baseline_indices)
        full_coverage = coverage_vector(coverage_matrix, full_indices)
        coverage_delta = full_coverage.astype(float) - baseline_coverage.astype(float)
        residual = normalized_share(predictions.y_true[idx]) - normalized_share(
            predictions.yhat_full[idx]
        )
        alignments.append(float(np.dot(residual, coverage_delta)))
        coverage_changed.append(bool(np.any(coverage_delta != 0.0)))
    alignments_array = np.asarray(alignments, dtype=float)
    return {
        "alignment_mean": float(np.mean(alignments_array)),
        "alignment_positive_share": float(np.mean(alignments_array > 0.0)),
        "coverage_changed_share": float(np.mean(coverage_changed)),
    }


def assert_timestamps_match(
    hourly_timestamps: Iterable[str],
    prediction_timestamps: Sequence[str],
    run_dir: Path,
) -> None:
    left = tuple(str(value) for value in hourly_timestamps)
    right = tuple(str(value) for value in prediction_timestamps)
    if left != right:
        raise ValueError(f"Timestamp mismatch between saved outputs and predictions: {run_dir}")


def normalized_share(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(array.sum())
    if total <= 0.0:
        return np.zeros_like(array, dtype=float)
    return array / total


def gini(values: Sequence[float] | np.ndarray) -> float:
    array = np.sort(np.maximum(np.asarray(values, dtype=float), 0.0))
    total = float(array.sum())
    if total <= 0.0:
        return 0.0
    n = len(array)
    weights = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.dot(weights, array)) / (n * total) - (n + 1) / n)


def entropy(shares: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(shares, dtype=float)
    positive = array[array > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def selected_change_share(left: pd.Series, right: pd.Series) -> float:
    changes = [
        set(parse_json_list(left_value)) != set(parse_json_list(right_value))
        for left_value, right_value in zip(left, right, strict=True)
    ]
    return float(np.mean(changes))


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list-like JSON value, got {value!r}")
    return parsed


def zip_codes_to_indices(zip_codes: Sequence[Any], ordered_zip_codes: Sequence[str]) -> tuple[int, ...]:
    index_by_zip = {zip_code: idx for idx, zip_code in enumerate(ordered_zip_codes)}
    return tuple(index_by_zip[str(zip_code)] for zip_code in zip_codes)


def coverage_vector(
    coverage_matrix: np.ndarray,
    selected_indices: Sequence[int] | frozenset[int],
) -> np.ndarray:
    selected = tuple(int(idx) for idx in selected_indices)
    if not selected:
        return np.zeros(coverage_matrix.shape[0], dtype=bool)
    return np.asarray(coverage_matrix[:, selected].any(axis=1), dtype=bool)


def set_jaccard_distance(left: frozenset[int], right: frozenset[int]) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)


def safe_div(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return float("nan")
    return float(numerator / denominator)


def spearman(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2:
        return float("nan")
    result = stats.spearmanr(left.to_numpy(dtype=float), right.to_numpy(dtype=float))
    return float(result.statistic)


if __name__ == "__main__":
    main()
