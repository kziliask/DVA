from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cmcrameri  # noqa: F401 - registers cmc.* colormaps with Matplotlib.
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import Voronoi
from shapely.geometry import MultiPoint, Polygon


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RESULT_ROOT = Path("results/ems/experiment_a_infodva")
DEFAULT_GEOJSON_PATH = Path("data/ems_data/modzcta_boundaries.geojson")
DEFAULT_GEOGRAPHY_CSV_PATH = Path("data/ems_data/processed/ems_zip_geography_manhattan.csv")
DEFAULT_TARGET_PATH = Path("data/ems_data/processed/ems_zip_hour_features_2025_manhattan_wide_y.csv")
DEFAULT_DISTANCE_MATRIX_PATH = Path("data/ems_data/processed/ems_zip_centroid_distance_matrix_km.parquet")
DEFAULT_OUTDIR = Path("data/plots/ems_zip_zone_regime_map")
DEFAULT_COVERAGE_RADIUS_KM = 1.0
DEFAULT_FACILITY_BUDGET = 8
DEFAULT_COVERAGE_SOLVER = "exact"
DEFAULT_PANEL_A_METRIC = "full_minus_oracle_covered"
DEFAULT_OUTPUT_PREFIX = "ems_zip_zone_decision_active_regime"
NYC_AREA_CRS = "EPSG:2263"

FEATURE_LABELS = {
    "hour": "Hour",
    "day_of_week": "Day of week",
    "temp_c": "Temperature",
    "precip_mm": "Precipitation",
    "citywide_ems_incidents_lag_1": "Citywide EMS lag 1",
    "ems_incidents_lag_1": "ZIP lag 1 EMS",
    "neighbor_ems_incidents_lag_1_mean": "Neighbor lag 1 EMS",
    "zone_hour_baseline": "Zone-hour baseline",
}
PREFERRED_FEATURE_ORDER = (
    "zone_hour_baseline",
    "ems_incidents_lag_1",
    "neighbor_ems_incidents_lag_1_mean",
    "citywide_ems_incidents_lag_1",
    "hour",
    "temp_c",
    "day_of_week",
    "precip_mm",
)

BASE_EDGE_COLOR = "#383838"
BACKGROUND_ZONE_COLOR = "#f4f1eb"
MISSING_ZONE_COLOR = "#eeeeee"
ORACLE_COLOR = "#2a7f62"
BASELINE_COLOR = "#2468a8"
FULL_COLOR = "#c74f32"
NEUTRALIZED_COLOR = "#6f4aa8"
BOTH_COVERED_COLOR = "#9e9e9e"
ANNOTATION_COLOR = "#202020"


@dataclass(frozen=True, slots=True)
class RegimeRun:
    model_id: str
    setting_id: str
    results_dir: Path
    coverage_radius_km: float
    facility_budget: int
    coverage_solver: str


@dataclass(frozen=True, slots=True)
class RepresentativeCase:
    run: RegimeRun
    timestamp_hour: pd.Timestamp
    feature: str
    feature_dva_value: float
    score: float
    actual_total_demand: float
    baseline_selected_zip_codes: tuple[str, ...]
    full_selected_zip_codes: tuple[str, ...]
    oracle_selected_zip_codes: tuple[str, ...]
    neutralized_selected_zip_codes: tuple[str, ...]
    baseline_covered_zip_codes: tuple[str, ...]
    full_covered_zip_codes: tuple[str, ...]
    oracle_covered_zip_codes: tuple[str, ...]
    neutralized_covered_zip_codes: tuple[str, ...]
    baseline_covered_demand: float
    full_covered_demand: float
    oracle_covered_demand: float
    neutralized_covered_demand: float
    geography_source: str


@dataclass(frozen=True, slots=True)
class RegimeMapOutputs:
    png_path: Path
    pdf_path: Path
    panel_a_csv: Path
    zone_summary_csv: Path
    staging_frequency_csv: Path
    panel_value_summary_csv: Path
    representative_context_csv: Path
    metadata_json: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a three-panel Manhattan EMS ZIP-zone map for one "
            "decision-active staging regime."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Root containing xgb_*/manifest.csv and InfoDVA run outputs.",
    )
    parser.add_argument(
        "--geojson",
        type=Path,
        default=DEFAULT_GEOJSON_PATH,
        help=(
            "ZIP polygon GeoJSON. Supports either processed files with zip_code or "
            "raw NYC MODZCTA files with modzcta/label/zcta. If missing, an "
            "approximate centroid Voronoi ZIP-zone map is drawn from --geography-csv."
        ),
    )
    parser.add_argument(
        "--geography-csv",
        type=Path,
        default=DEFAULT_GEOGRAPHY_CSV_PATH,
        help="Processed EMS ZIP geography CSV with centroid_lon/centroid_lat columns.",
    )
    parser.add_argument("--target-path", type=Path, default=DEFAULT_TARGET_PATH)
    parser.add_argument(
        "--distance-matrix",
        type=Path,
        default=DEFAULT_DISTANCE_MATRIX_PATH,
        help="ZIP centroid distance matrix in km.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--coverage-radius-km",
        type=float,
        default=DEFAULT_COVERAGE_RADIUS_KM,
        help="Coverage radius tau for the representative regime.",
    )
    parser.add_argument(
        "--facility-budget",
        type=int,
        default=DEFAULT_FACILITY_BUDGET,
        help="Number of staging units p for the representative regime.",
    )
    parser.add_argument(
        "--coverage-solver",
        default=DEFAULT_COVERAGE_SOLVER,
        help="Coverage solver label/name to select from manifest rows.",
    )
    parser.add_argument(
        "--model-id",
        default="auto",
        help="Model id to plot, e.g. xgb_001. Use auto to choose a decision-active run.",
    )
    parser.add_argument(
        "--timestamp-hour",
        default=None,
        help="Timestamp hour to plot. When omitted, a decision-active hour is selected.",
    )
    parser.add_argument(
        "--feature",
        default="auto",
        help=(
            "Feature to neutralize in panel C. Use auto to select the highest-impact "
            "feature for the representative hour."
        ),
    )
    parser.add_argument(
        "--feature-sign",
        choices=("auto", "positive", "negative"),
        default="auto",
        help="Restrict auto feature selection to positive or negative post-DVA values.",
    )
    parser.add_argument(
        "--panel-a-metric",
        choices=(
            "full_minus_oracle_covered",
            "oracle_minus_full_uncovered",
            "oracle_minus_baseline_uncovered",
            "oracle_uncovered_demand",
            "realized_demand",
        ),
        default=DEFAULT_PANEL_A_METRIC,
        help="Zone quantity used for panel A.",
    )
    parser.add_argument(
        "--aggregation-scope",
        "--panel-a-scope",
        dest="aggregation_scope",
        choices=("selected-run", "all-runs"),
        default="all-runs",
        help=(
            "Aggregate continuous polygon and staging summaries over the selected "
            "representative run or all matching runs."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for generated PNG/PDF/CSV/JSON files.",
    )
    parser.add_argument(
        "--include-zip-labels",
        action="store_true",
        help="Annotate ZIP codes on each panel.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = write_ems_zip_zone_regime_map(
        result_root=args.result_root,
        geojson_path=args.geojson,
        geography_csv_path=args.geography_csv,
        target_path=args.target_path,
        distance_matrix_path=args.distance_matrix,
        outdir=args.outdir,
        coverage_radius_km=args.coverage_radius_km,
        facility_budget=args.facility_budget,
        coverage_solver=args.coverage_solver,
        model_id=args.model_id,
        timestamp_hour=args.timestamp_hour,
        feature=args.feature,
        feature_sign=args.feature_sign,
        panel_a_metric=args.panel_a_metric,
        aggregation_scope=args.aggregation_scope,
        output_prefix=args.output_prefix,
        include_zip_labels=args.include_zip_labels,
    )
    print(f"Wrote EMS ZIP-zone regime map to {outputs.png_path}")
    print(f"Wrote EMS ZIP-zone regime map PDF to {outputs.pdf_path}")
    print(f"Wrote panel A zone summary to {outputs.panel_a_csv}")
    print(f"Wrote aggregate zone summary to {outputs.zone_summary_csv}")
    print(f"Wrote staging frequency summary to {outputs.staging_frequency_csv}")
    print(f"Wrote panel value summary to {outputs.panel_value_summary_csv}")
    print(f"Wrote representative context to {outputs.representative_context_csv}")
    print(f"Wrote representative-case metadata to {outputs.metadata_json}")


def write_ems_zip_zone_regime_map(
    *,
    result_root: Path,
    geojson_path: Path,
    geography_csv_path: Path,
    target_path: Path,
    distance_matrix_path: Path,
    outdir: Path,
    coverage_radius_km: float = DEFAULT_COVERAGE_RADIUS_KM,
    facility_budget: int = DEFAULT_FACILITY_BUDGET,
    coverage_solver: str = DEFAULT_COVERAGE_SOLVER,
    model_id: str = "auto",
    timestamp_hour: str | None = None,
    feature: str = "auto",
    feature_sign: str = "auto",
    panel_a_metric: str = DEFAULT_PANEL_A_METRIC,
    aggregation_scope: str = "all-runs",
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
    include_zip_labels: bool = False,
) -> RegimeMapOutputs:
    outdir.mkdir(parents=True, exist_ok=True)

    runs = load_matching_runs(
        result_root,
        coverage_radius_km=coverage_radius_km,
        facility_budget=facility_budget,
        coverage_solver=coverage_solver,
        model_id=model_id,
    )
    distance_matrix = load_distance_matrix(distance_matrix_path)
    targets = load_target_frame(target_path)

    geography, geography_source = load_zip_zone_geography(
        geojson_path=geojson_path,
        geography_csv_path=geography_csv_path,
    )
    zip_codes = tuple(geography["zip_code"].astype(str))

    aggregate_feature = choose_aggregate_feature(
        runs,
        feature=feature,
        feature_sign=feature_sign,
    )
    representative = choose_representative_case(
        runs,
        targets=targets,
        distance_matrix=distance_matrix,
        timestamp_hour=timestamp_hour,
        feature=aggregate_feature,
        feature_sign=feature_sign,
    )
    representative = replace(representative, geography_source=geography_source)

    aggregate_runs = (
        (representative.run,) if aggregation_scope == "selected-run" else tuple(runs)
    )
    zone_summary, staging_frequency = compute_aggregate_zone_summary(
        aggregate_runs,
        targets=targets,
        distance_matrix=distance_matrix,
        zip_codes=zip_codes,
        feature=aggregate_feature,
        metric=panel_a_metric,
    )
    panel_a = zone_summary.loc[
        :, ["zip_code", "panel_a_sum", "observation_count", "panel_a_mean", "panel_a_metric"]
    ].copy()
    panel_a_csv = outdir / f"{output_prefix}_panel_a_zone_values.csv"
    panel_a.to_csv(panel_a_csv, index=False)
    zone_summary_csv = outdir / f"{output_prefix}_aggregate_zone_summary.csv"
    staging_frequency_csv = outdir / f"{output_prefix}_staging_frequency_summary.csv"
    panel_value_summary_csv = outdir / f"{output_prefix}_panel_value_summary.csv"
    representative_context_csv = outdir / f"{output_prefix}_representative_context.csv"
    panel_value_summary = build_panel_value_summary(
        zone_summary,
        panel_a_metric=panel_a_metric,
    )
    zone_summary.to_csv(zone_summary_csv, index=False)
    staging_frequency.to_csv(staging_frequency_csv, index=False)
    panel_value_summary.to_csv(panel_value_summary_csv, index=False)
    representative_context_frame(representative).to_csv(
        representative_context_csv,
        index=False,
    )

    output_stem = (
        f"{output_prefix}_tau{_number_label(coverage_radius_km)}"
        f"_p{facility_budget}_{representative.run.model_id}"
        f"_{aggregate_feature}_aggregate_{aggregation_scope.replace('-', '_')}"
    )
    png_path = outdir / f"{output_stem}.png"
    pdf_path = outdir / f"{output_stem}.pdf"
    plot_regime_map(
        geography=geography,
        zone_summary=zone_summary,
        staging_frequency=staging_frequency,
        representative=representative,
        panel_a_metric=panel_a_metric,
        output_paths=(png_path, pdf_path),
        include_zip_labels=include_zip_labels,
    )

    metadata_json = outdir / f"{output_stem}.json"
    metadata_json.write_text(
        json.dumps(
            {
                "coverage_radius_km": coverage_radius_km,
                "facility_budget": facility_budget,
                "coverage_solver": coverage_solver,
                "aggregation_scope": aggregation_scope,
                "aggregate_run_count": len(aggregate_runs),
                "feature": aggregate_feature,
                "panel_a_metric": panel_a_metric,
                "panel_value_summary": panel_value_summary.to_dict(orient="records"),
                "representative_case": representative_to_metadata(representative),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return RegimeMapOutputs(
        png_path=png_path,
        pdf_path=pdf_path,
        panel_a_csv=panel_a_csv,
        zone_summary_csv=zone_summary_csv,
        staging_frequency_csv=staging_frequency_csv,
        panel_value_summary_csv=panel_value_summary_csv,
        representative_context_csv=representative_context_csv,
        metadata_json=metadata_json,
    )


def load_matching_runs(
    result_root: Path,
    *,
    coverage_radius_km: float,
    facility_budget: int,
    coverage_solver: str,
    model_id: str,
) -> tuple[RegimeRun, ...]:
    manifest_paths = sorted(result_root.glob("xgb_*/manifest.csv"))
    if not manifest_paths and (result_root / "manifest.csv").exists():
        manifest_paths = [result_root / "manifest.csv"]
    if not manifest_paths:
        raise FileNotFoundError(
            f"No EMS InfoDVA manifest.csv files found under {result_root}."
        )

    runs: list[RegimeRun] = []
    for manifest_path in manifest_paths:
        manifest = pd.read_csv(manifest_path)
        required_columns = {
            "setting_id",
            "model_id",
            "results_dir",
            "coverage_solver",
            "coverage_radius_km",
            "facility_budget",
        }
        missing = required_columns - set(manifest.columns)
        if missing:
            raise KeyError(
                f"{manifest_path} is missing columns: {', '.join(sorted(missing))}"
            )
        frame = manifest.loc[
            np.isclose(manifest["coverage_radius_km"].astype(float), coverage_radius_km)
            & manifest["facility_budget"].astype(int).eq(facility_budget)
        ].copy()
        if model_id != "auto":
            frame = frame.loc[frame["model_id"].astype(str).eq(model_id)].copy()
        solver_columns = ["coverage_solver"]
        if "coverage_solver_label" in frame.columns:
            solver_columns.append("coverage_solver_label")
        solver_match = pd.Series(False, index=frame.index)
        for solver_column in solver_columns:
            solver_match |= frame[solver_column].astype(str).eq(coverage_solver)
        frame = frame.loc[solver_match].copy()

        for row in frame.to_dict(orient="records"):
            results_dir = _resolve_manifest_path(manifest_path, Path(str(row["results_dir"])))
            runs.append(
                RegimeRun(
                    model_id=str(row["model_id"]),
                    setting_id=str(row["setting_id"]),
                    results_dir=results_dir,
                    coverage_radius_km=float(row["coverage_radius_km"]),
                    facility_budget=int(row["facility_budget"]),
                    coverage_solver=str(row["coverage_solver"]),
                )
            )

    if not runs:
        raise FileNotFoundError(
            "No EMS InfoDVA run matched "
            f"tau={coverage_radius_km:g}, p={facility_budget}, "
            f"solver={coverage_solver!r}, model_id={model_id!r}."
        )
    return tuple(sorted(runs, key=lambda run: (run.model_id, run.setting_id)))


def choose_aggregate_feature(
    runs: Sequence[RegimeRun],
    *,
    feature: str,
    feature_sign: str,
) -> str:
    if feature != "auto":
        return feature

    totals: dict[str, dict[str, float]] = {}
    for run in runs:
        hourly = load_hourly_shap(run.results_dir)
        for column in hourly.columns:
            column_name = str(column)
            if not column_name.startswith("decision_shap_"):
                continue
            feature_name = column_name.removeprefix("decision_shap_")
            values = pd.to_numeric(hourly[column_name], errors="raise").dropna()
            if values.empty:
                continue
            summary = totals.setdefault(
                feature_name,
                {"signed_sum": 0.0, "abs_sum": 0.0, "count": 0.0},
            )
            summary["signed_sum"] += float(values.sum())
            summary["abs_sum"] += float(values.abs().sum())
            summary["count"] += float(values.shape[0])

    if not totals:
        raise ValueError("No decision_shap_* columns found for aggregate feature choice.")

    rows = []
    for feature_name, summary in totals.items():
        count = summary["count"]
        rows.append(
            {
                "feature": feature_name,
                "mean_signed": summary["signed_sum"] / count,
                "mean_abs": summary["abs_sum"] / count,
            }
        )
    frame = pd.DataFrame(rows)
    if feature_sign == "positive":
        candidates = frame.loc[frame["mean_signed"].gt(0)].copy()
        sort_columns = ["mean_signed", "mean_abs"]
        ascending = [False, False]
    elif feature_sign == "negative":
        candidates = frame.loc[frame["mean_signed"].lt(0)].copy()
        sort_columns = ["mean_signed", "mean_abs"]
        ascending = [True, False]
    else:
        candidates = frame.copy()
        sort_columns = ["mean_abs"]
        ascending = [False]
    if candidates.empty:
        candidates = frame.copy()
        sort_columns = ["mean_abs"]
        ascending = [False]

    preferred_rank = {
        feature_name: idx for idx, feature_name in enumerate(PREFERRED_FEATURE_ORDER)
    }
    candidates["preferred_rank"] = candidates["feature"].map(
        lambda name: preferred_rank.get(str(name), len(preferred_rank))
    )
    candidates = candidates.sort_values(
        [*sort_columns, "preferred_rank", "feature"],
        ascending=[*ascending, True, True],
    )
    return str(candidates.iloc[0]["feature"])


def choose_representative_case(
    runs: Sequence[RegimeRun],
    *,
    targets: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    timestamp_hour: str | None,
    feature: str,
    feature_sign: str,
) -> RepresentativeCase:
    requested_timestamp = pd.Timestamp(timestamp_hour) if timestamp_hour else None
    best: RepresentativeCase | None = None
    for run in runs:
        hourly = load_hourly_shap(run.results_dir)
        if requested_timestamp is not None:
            hourly = hourly.loc[hourly["timestamp_hour"].eq(requested_timestamp)].copy()
        if hourly.empty:
            continue
        coalition_values = load_coalition_values(run.results_dir)
        player_names = load_player_names(run.results_dir, hourly)
        full_mask = (1 << len(player_names)) - 1

        for row in hourly.to_dict(orient="records"):
            row_timestamp = pd.Timestamp(row["timestamp_hour"])
            if row_timestamp not in targets.index:
                continue
            actual_demand = target_row_by_zip(targets.loc[row_timestamp])
            actual_total_demand = float(actual_demand.sum())
            if actual_total_demand <= 0:
                continue

            feature_order = feature_candidates_for_row(
                row,
                player_names=player_names,
                feature=feature,
                feature_sign=feature_sign,
            )
            for feature_name in feature_order:
                if feature_name not in player_names:
                    continue
                neutralized_mask = full_mask & ~(1 << player_names.index(feature_name))
                neutralized_row = select_coalition_row(
                    coalition_values,
                    timestamp_hour=row_timestamp,
                    coalition_mask=neutralized_mask,
                )
                if neutralized_row is None:
                    continue

                full_selected = _parse_zip_code_list(row["full_selected_zip_codes"])
                baseline_selected = _parse_zip_code_list(row["baseline_selected_zip_codes"])
                oracle_selected = _parse_zip_code_list(row["oracle_selected_zip_codes"])
                neutralized_selected = _parse_zip_code_list(
                    neutralized_row["decision_selected_facility_zip_codes"]
                )

                full_covered = coverage_zip_codes(
                    full_selected,
                    distance_matrix=distance_matrix,
                    coverage_radius_km=run.coverage_radius_km,
                )
                baseline_covered = coverage_zip_codes(
                    baseline_selected,
                    distance_matrix=distance_matrix,
                    coverage_radius_km=run.coverage_radius_km,
                )
                oracle_covered = coverage_zip_codes(
                    oracle_selected,
                    distance_matrix=distance_matrix,
                    coverage_radius_km=run.coverage_radius_km,
                )
                neutralized_covered = coverage_zip_codes(
                    neutralized_selected,
                    distance_matrix=distance_matrix,
                    coverage_radius_km=run.coverage_radius_km,
                )

                feature_dva_value = float(row.get(f"decision_shap_{feature_name}", 0.0))
                baseline_demand = realized_covered_demand(
                    actual_demand,
                    baseline_covered,
                )
                full_demand = realized_covered_demand(actual_demand, full_covered)
                oracle_demand = realized_covered_demand(actual_demand, oracle_covered)
                neutralized_demand = realized_covered_demand(
                    actual_demand,
                    neutralized_covered,
                )
                baseline_shift = len(set(baseline_selected) ^ set(full_selected))
                neutralized_shift = len(set(neutralized_selected) ^ set(full_selected))
                score = (
                    1000.0 * float(neutralized_shift > 0)
                    + 500.0 * float(baseline_shift > 0)
                    + 20.0 * neutralized_shift
                    + 10.0 * baseline_shift
                    + 5.0 * abs(full_demand - neutralized_demand)
                    + 2.0 * abs(full_demand - baseline_demand)
                    + max(oracle_demand - full_demand, 0.0)
                    + 100.0 * abs(feature_dva_value)
                )
                candidate = RepresentativeCase(
                    run=run,
                    timestamp_hour=row_timestamp,
                    feature=feature_name,
                    feature_dva_value=feature_dva_value,
                    score=score,
                    actual_total_demand=actual_total_demand,
                    baseline_selected_zip_codes=baseline_selected,
                    full_selected_zip_codes=full_selected,
                    oracle_selected_zip_codes=oracle_selected,
                    neutralized_selected_zip_codes=neutralized_selected,
                    baseline_covered_zip_codes=baseline_covered,
                    full_covered_zip_codes=full_covered,
                    oracle_covered_zip_codes=oracle_covered,
                    neutralized_covered_zip_codes=neutralized_covered,
                    baseline_covered_demand=baseline_demand,
                    full_covered_demand=full_demand,
                    oracle_covered_demand=oracle_demand,
                    neutralized_covered_demand=neutralized_demand,
                    geography_source="",
                )
                if best is None or candidate.score > best.score:
                    best = candidate
                if feature != "auto":
                    break

    if best is None:
        raise ValueError(
            "Could not find a representative EMS case with matching hourly and "
            "coalition outputs."
        )
    return best


def compute_aggregate_zone_summary(
    runs: Sequence[RegimeRun],
    *,
    targets: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    zip_codes: Sequence[str],
    feature: str,
    metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    zip_index = pd.Index([str(zip_code) for zip_code in zip_codes], name="zip_code")
    n_zones = len(zip_index)
    panel_a_sum = np.zeros(n_zones, dtype=float)
    full_minus_baseline_incident_sum = np.zeros(n_zones, dtype=float)
    full_minus_neutralized_incident_sum = np.zeros(n_zones, dtype=float)
    full_minus_baseline_coverage_sum = np.zeros(n_zones, dtype=float)
    full_minus_neutralized_coverage_sum = np.zeros(n_zones, dtype=float)
    baseline_selected_sum = np.zeros(n_zones, dtype=float)
    full_selected_sum = np.zeros(n_zones, dtype=float)
    oracle_selected_sum = np.zeros(n_zones, dtype=float)
    neutralized_selected_sum = np.zeros(n_zones, dtype=float)
    observation_count = 0

    for run in runs:
        hourly = load_hourly_shap(run.results_dir)
        coalition_values = load_coalition_values(run.results_dir)
        player_names = load_player_names(run.results_dir, hourly)
        if feature not in player_names:
            raise ValueError(f"Feature {feature!r} is not a player in {run.results_dir}.")
        full_mask = (1 << len(player_names)) - 1
        neutralized_mask = full_mask & ~(1 << player_names.index(feature))
        neutralized_rows = coalition_values.loc[
            coalition_values["coalition_mask"].eq(neutralized_mask)
        ].set_index("timestamp_hour")

        for row in hourly.to_dict(orient="records"):
            timestamp_hour = pd.Timestamp(row["timestamp_hour"])
            if timestamp_hour not in targets.index:
                continue
            if timestamp_hour not in neutralized_rows.index:
                continue

            actual = (
                target_row_by_zip(targets.loc[timestamp_hour])
                .reindex(zip_index)
                .fillna(0.0)
                .astype(float)
            )
            baseline_covered = set(_parse_zip_code_list(row["baseline_covered_zip_codes"]))
            full_covered = set(_parse_zip_code_list(row["full_covered_zip_codes"]))
            oracle_covered = set(_parse_zip_code_list(row["oracle_covered_zip_codes"]))
            neutralized_selected_value = neutralized_rows.loc[
                timestamp_hour,
                "decision_selected_facility_zip_codes",
            ]
            if isinstance(neutralized_selected_value, pd.Series):
                neutralized_selected_value = neutralized_selected_value.iloc[0]
            neutralized_selected = _parse_zip_code_list(neutralized_selected_value)
            neutralized_covered = set(
                coverage_zip_codes(
                    neutralized_selected,
                    distance_matrix=distance_matrix,
                    coverage_radius_km=run.coverage_radius_km,
                )
            )

            values = panel_a_values_for_hour(
                actual,
                baseline_covered=baseline_covered,
                full_covered=full_covered,
                oracle_covered=oracle_covered,
                metric=metric,
            )
            baseline_indicator = zip_membership_indicator(zip_index, baseline_covered)
            full_indicator = zip_membership_indicator(zip_index, full_covered)
            neutralized_indicator = zip_membership_indicator(zip_index, neutralized_covered)

            panel_a_sum += values.reindex(zip_index).fillna(0.0).to_numpy(dtype=float)
            full_minus_baseline_coverage = full_indicator - baseline_indicator
            full_minus_neutralized_coverage = full_indicator - neutralized_indicator
            actual_values = actual.to_numpy(dtype=float)
            full_minus_baseline_incident_sum += (
                actual_values * full_minus_baseline_coverage
            )
            full_minus_neutralized_incident_sum += (
                actual_values * full_minus_neutralized_coverage
            )
            full_minus_baseline_coverage_sum += full_minus_baseline_coverage
            full_minus_neutralized_coverage_sum += full_minus_neutralized_coverage

            baseline_selected_sum += zip_membership_indicator(
                zip_index,
                set(_parse_zip_code_list(row["baseline_selected_zip_codes"])),
            )
            full_selected_sum += zip_membership_indicator(
                zip_index,
                set(_parse_zip_code_list(row["full_selected_zip_codes"])),
            )
            oracle_selected_sum += zip_membership_indicator(
                zip_index,
                set(_parse_zip_code_list(row["oracle_selected_zip_codes"])),
            )
            neutralized_selected_sum += zip_membership_indicator(
                zip_index,
                set(neutralized_selected),
            )
            observation_count += 1

    if observation_count == 0:
        raise ValueError("No matching hourly observations found for aggregate maps.")

    zone_summary = pd.DataFrame(
        {
            "zip_code": zip_index.astype(str),
            "panel_a_sum": panel_a_sum,
            "observation_count": observation_count,
            "full_minus_baseline_covered_incidents_sum": (
                full_minus_baseline_incident_sum
            ),
            "full_minus_neutralized_covered_incidents_sum": (
                full_minus_neutralized_incident_sum
            ),
            "full_minus_baseline_coverage_frequency_sum": (
                full_minus_baseline_coverage_sum
            ),
            "full_minus_neutralized_coverage_frequency_sum": (
                full_minus_neutralized_coverage_sum
            ),
        }
    )
    zone_summary["panel_a_mean"] = zone_summary["panel_a_sum"] / observation_count
    zone_summary["panel_a_metric"] = metric
    zone_summary["full_minus_baseline_covered_incidents_mean"] = (
        zone_summary["full_minus_baseline_covered_incidents_sum"] / observation_count
    )
    zone_summary["full_minus_neutralized_covered_incidents_mean"] = (
        zone_summary["full_minus_neutralized_covered_incidents_sum"]
        / observation_count
    )
    zone_summary["full_minus_baseline_coverage_frequency_difference"] = (
        zone_summary["full_minus_baseline_coverage_frequency_sum"] / observation_count
    )
    zone_summary["full_minus_neutralized_coverage_frequency_difference"] = (
        zone_summary["full_minus_neutralized_coverage_frequency_sum"]
        / observation_count
    )
    zone_summary["neutralized_feature"] = feature

    staging_frequency = pd.DataFrame(
        {
            "zip_code": zip_index.astype(str),
            "observation_count": observation_count,
            "baseline_selected_count": baseline_selected_sum,
            "full_selected_count": full_selected_sum,
            "oracle_selected_count": oracle_selected_sum,
            "neutralized_selected_count": neutralized_selected_sum,
        }
    )
    staging_frequency["baseline_selected_rate"] = (
        staging_frequency["baseline_selected_count"] / observation_count
    )
    staging_frequency["full_selected_rate"] = (
        staging_frequency["full_selected_count"] / observation_count
    )
    staging_frequency["oracle_selected_rate"] = (
        staging_frequency["oracle_selected_count"] / observation_count
    )
    staging_frequency["neutralized_selected_rate"] = (
        staging_frequency["neutralized_selected_count"] / observation_count
    )
    staging_frequency["full_minus_oracle_selected_rate"] = (
        staging_frequency["full_selected_rate"]
        - staging_frequency["oracle_selected_rate"]
    )
    staging_frequency["full_minus_baseline_selected_rate"] = (
        staging_frequency["full_selected_rate"]
        - staging_frequency["baseline_selected_rate"]
    )
    staging_frequency["full_minus_neutralized_selected_rate"] = (
        staging_frequency["full_selected_rate"]
        - staging_frequency["neutralized_selected_rate"]
    )
    staging_frequency["neutralized_feature"] = feature
    return zone_summary, staging_frequency


def zip_membership_indicator(zip_index: pd.Index, members: set[str]) -> np.ndarray:
    return np.array([zip_code in members for zip_code in zip_index], dtype=float)


def build_panel_value_summary(
    zone_summary: pd.DataFrame,
    *,
    panel_a_metric: str,
) -> pd.DataFrame:
    panel_specs = [
        ("A", panel_a_colorbar_label(panel_a_metric), "panel_a_mean"),
        (
            "B",
            "FM Covered - Baseline Covered",
            "full_minus_baseline_covered_incidents_mean",
        ),
        (
            "C",
            "FM Covered - Neutralized Covered",
            "full_minus_neutralized_covered_incidents_mean",
        ),
    ]
    rows = []
    for panel, quantity, value_column in panel_specs:
        values = zone_summary[value_column].astype(float)
        rows.append(
            {
                "panel": panel,
                "quantity": quantity,
                "spatial_sum_mean_per_run_hour": float(values.sum()),
                "spatial_mean_per_zip_run_hour": float(values.mean()),
                "positive_spatial_sum_mean_per_run_hour": float(
                    values.clip(lower=0).sum()
                ),
                "negative_spatial_sum_mean_per_run_hour": float(
                    values.clip(upper=0).sum()
                ),
                "zip_count": int(values.notna().sum()),
                "units": "realized incidents per run-hour",
            }
        )
    return pd.DataFrame(rows)


def panel_a_values_for_hour(
    actual: pd.Series,
    *,
    baseline_covered: set[str],
    full_covered: set[str],
    oracle_covered: set[str],
    metric: str,
) -> pd.Series:
    zip_index = actual.index.astype(str)
    values = actual.astype(float).copy()
    if metric == "realized_demand":
        return values
    if metric == "oracle_uncovered_demand":
        return values * np.array([zip_code not in oracle_covered for zip_code in zip_index])
    if metric == "full_minus_oracle_covered":
        full_indicator = np.array([zip_code in full_covered for zip_code in zip_index])
        oracle_indicator = np.array([zip_code in oracle_covered for zip_code in zip_index])
        return values * (full_indicator.astype(float) - oracle_indicator.astype(float))
    if metric == "oracle_minus_full_uncovered":
        full_uncovered = np.array([zip_code not in full_covered for zip_code in zip_index])
        oracle_uncovered = np.array([zip_code not in oracle_covered for zip_code in zip_index])
        return values * (full_uncovered.astype(float) - oracle_uncovered.astype(float))
    if metric == "oracle_minus_baseline_uncovered":
        baseline_uncovered = np.array(
            [zip_code not in baseline_covered for zip_code in zip_index]
        )
        oracle_uncovered = np.array([zip_code not in oracle_covered for zip_code in zip_index])
        return values * (baseline_uncovered.astype(float) - oracle_uncovered.astype(float))
    raise ValueError(f"Unknown panel A metric: {metric}")


def plot_regime_map(
    *,
    geography: gpd.GeoDataFrame,
    zone_summary: pd.DataFrame,
    staging_frequency: pd.DataFrame,
    representative: RepresentativeCase,
    panel_a_metric: str,
    output_paths: tuple[Path, Path],
    include_zip_labels: bool,
) -> None:
    apply_plot_style()
    plot_geography = geography.to_crs("EPSG:4326").copy()
    panel_geo = plot_geography.merge(
        zone_summary.loc[
            :,
            [
                "zip_code",
                "panel_a_mean",
                "full_minus_baseline_covered_incidents_mean",
                "full_minus_neutralized_covered_incidents_mean",
            ],
        ],
        on="zip_code",
        how="left",
    )
    marker_geo = plot_geography.merge(staging_frequency, on="zip_code", how="left")
    fig, axes = plt.subplots(1, 3, figsize=(15.7, 7.1), constrained_layout=True)

    panel_a_mappable = plot_panel_a(
        axes[0],
        panel_geo,
        panel_a_metric=panel_a_metric,
    )
    plot_staging_frequency_markers(
        axes[0],
        marker_geo,
        first_column="oracle_selected_rate",
        second_column="full_selected_rate",
        first_label="Oracle",
        second_label="Full model",
        first_marker="o",
        second_marker="*",
        first_color=ORACLE_COLOR,
        second_color=FULL_COLOR,
        legend_loc="upper left",
        show_size_legend=False,
    )
    panel_b_mappable = plot_delta_panel(
        axes[1],
        panel_geo,
        value_column="full_minus_baseline_covered_incidents_mean",
        title="B. Incidents covered by Full Model - Baseline",
    )
    plot_staging_frequency_markers(
        axes[1],
        marker_geo,
        first_column="baseline_selected_rate",
        second_column="full_selected_rate",
        first_label="Baseline",
        second_label="Full model",
        first_marker="o",
        second_marker="*",
        first_color=BASELINE_COLOR,
        second_color=FULL_COLOR,
        legend_loc="upper left",
        show_size_legend=True,
    )
    panel_c_mappable = plot_delta_panel(
        axes[2],
        panel_geo,
        value_column="full_minus_neutralized_covered_incidents_mean",
        title="C. Incidents covered by Full Model - Neutralized Model",
    )
    plot_staging_frequency_markers(
        axes[2],
        marker_geo,
        first_column="neutralized_selected_rate",
        second_column="full_selected_rate",
        first_label=f"Without {FEATURE_LABELS.get(representative.feature, representative.feature)}",
        second_label="Full model",
        first_marker="o",
        second_marker="*",
        first_color=NEUTRALIZED_COLOR,
        second_color=FULL_COLOR,
        legend_loc="upper left",
        show_size_legend=False,
    )

    for ax in axes:
        ax.set_axis_off()
        ax.set_aspect("equal")
        if include_zip_labels:
            add_zip_labels(ax, plot_geography)

    set_common_map_extent(axes, plot_geography)
    colorbar = fig.colorbar(
        panel_a_mappable,
        ax=axes[0],
        fraction=0.047,
        pad=0.015,
    )
    colorbar.set_label(panel_a_colorbar_label(panel_a_metric), fontsize=8.5)
    panel_b_colorbar = fig.colorbar(
        panel_b_mappable,
        ax=axes[1],
        fraction=0.047,
        pad=0.015,
    )
    panel_b_colorbar.set_label(
        "FM Covered - Baseline Covered",
        fontsize=8.5,
    )
    panel_c_colorbar = fig.colorbar(
        panel_c_mappable,
        ax=axes[2],
        fraction=0.047,
        pad=0.015,
    )
    panel_c_colorbar.set_label(
        "FM Covered - Neutralized Covered",
        fontsize=8.5,
    )

    for output_path in output_paths:
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_panel_a(
    ax: plt.Axes,
    panel_geo: gpd.GeoDataFrame,
    *,
    panel_a_metric: str,
) -> plt.cm.ScalarMappable:
    values = panel_geo["panel_a_mean"].astype(float)
    diverging_metric = panel_a_metric in {
        "full_minus_oracle_covered",
        "oracle_minus_full_uncovered",
        "oracle_minus_baseline_uncovered",
    }
    cmap_name = "cmc.vik" if diverging_metric else "cmc.lajolla"
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(MISSING_ZONE_COLOR)
    if diverging_metric and values.notna().any():
        abs_limit = float(np.nanmax(np.abs(values.to_numpy(dtype=float))))
        if abs_limit == 0.0:
            abs_limit = 1.0
        norm: Normalize = TwoSlopeNorm(vmin=-abs_limit, vcenter=0.0, vmax=abs_limit)
    else:
        vmax = float(np.nanmax(values.to_numpy(dtype=float))) if values.notna().any() else 1.0
        norm = Normalize(vmin=0.0, vmax=vmax)

    panel_geo.plot(
        ax=ax,
        column="panel_a_mean",
        cmap=cmap,
        norm=norm,
        edgecolor=BASE_EDGE_COLOR,
        linewidth=0.42,
        missing_kwds={"color": MISSING_ZONE_COLOR},
    )
    panel_geo.boundary.plot(ax=ax, color="white", linewidth=0.18, alpha=0.95)
    ax.set_title(
        "A. Incidents covered by Full Model - Oracle",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
    )
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    return scalar_mappable


def plot_delta_panel(
    ax: plt.Axes,
    panel_geo: gpd.GeoDataFrame,
    *,
    value_column: str,
    title: str,
) -> plt.cm.ScalarMappable:
    values = panel_geo[value_column].astype(float)
    cmap = plt.get_cmap("cmc.vik").copy()
    cmap.set_bad(MISSING_ZONE_COLOR)
    if values.notna().any():
        abs_limit = float(np.nanmax(np.abs(values.to_numpy(dtype=float))))
        if abs_limit == 0.0:
            abs_limit = 1.0
    else:
        abs_limit = 1.0
    norm = TwoSlopeNorm(vmin=-abs_limit, vcenter=0.0, vmax=abs_limit)

    panel_geo.plot(
        ax=ax,
        column=value_column,
        cmap=cmap,
        norm=norm,
        edgecolor=BASE_EDGE_COLOR,
        linewidth=0.42,
        missing_kwds={"color": MISSING_ZONE_COLOR},
    )
    panel_geo.boundary.plot(ax=ax, color="white", linewidth=0.18, alpha=0.95)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    return scalar_mappable


def plot_staging_frequency_markers(
    ax: plt.Axes,
    geography: gpd.GeoDataFrame,
    *,
    first_column: str,
    second_column: str,
    first_label: str,
    second_label: str,
    first_marker: str,
    second_marker: str,
    first_color: str,
    second_color: str,
    legend_loc: str,
    show_size_legend: bool,
) -> None:
    points = geography.geometry.representative_point()
    first_rates = geography[first_column].fillna(0.0).astype(float).to_numpy()
    second_rates = geography[second_column].fillna(0.0).astype(float).to_numpy()
    x = np.array([point.x for point in points])
    y = np.array([point.y for point in points])
    first_mask = first_rates > 0
    second_mask = second_rates > 0

    if first_mask.any():
        ax.scatter(
            x[first_mask],
            y[first_mask],
            s=marker_sizes(first_rates[first_mask]),
            marker=first_marker,
            facecolors=marker_face_color(first_marker, first_color),
            edgecolors=first_color,
            linewidths=1.35,
            alpha=0.9,
            zorder=13,
        )
    if second_mask.any():
        ax.scatter(
            x[second_mask],
            y[second_mask],
            s=marker_sizes(second_rates[second_mask]),
            marker=second_marker,
            facecolors=marker_face_color(second_marker, second_color),
            edgecolors="#1f1f1f",
            linewidths=0.55,
            alpha=0.9,
            zorder=14,
        )

    symbol_handles = marker_symbol_handles(
        first_marker=first_marker,
        second_marker=second_marker,
        first_label=first_label,
        second_label=second_label,
        first_color=first_color,
        second_color=second_color,
    )
    symbol_legend = ax.legend(
        handles=symbol_handles,
        loc=legend_loc,
        fontsize=7.1,
        frameon=True,
        framealpha=0.92,
        borderpad=0.42,
        handlelength=1.35,
    )
    ax.add_artist(symbol_legend)
    if not show_size_legend:
        return

    size_rates = (1.0, 0.7, 0.4, 0.1)
    size_handles = [
        (
            legend_marker_handle(
                marker=first_marker,
                color=first_color,
                size_rate=rate,
                markeredgecolor=first_color,
            ),
            legend_marker_handle(
                marker=second_marker,
                color=second_color,
                size_rate=rate,
                markeredgecolor="#1f1f1f",
            ),
        )
        for rate in size_rates
    ]
    size_labels = [f"{rate:.0%}" for rate in size_rates]
    ax.legend(
        handles=size_handles,
        labels=size_labels,
        title="Selection rate",
        handler_map={tuple: HandlerTuple(ndivide=None, pad=2.0)},
        loc="upper right",
        bbox_to_anchor=(1.0, 0.82),
        fontsize=7.1,
        title_fontsize=7.1,
        frameon=True,
        framealpha=0.92,
        borderpad=0.62,
        handlelength=4.6,
        handleheight=2.25,
        handletextpad=1.0,
        labelspacing=1.6,
    )


def marker_symbol_handles(
    *,
    first_marker: str,
    second_marker: str,
    first_label: str,
    second_label: str,
    first_color: str,
    second_color: str,
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=first_marker,
            color="none",
            markerfacecolor=marker_face_color(first_marker, first_color),
            markeredgecolor=first_color,
            markeredgewidth=1.5,
            markersize=6.2,
            label=first_label,
        ),
        Line2D(
            [0],
            [0],
            marker=second_marker,
            color="none",
            markerfacecolor=marker_face_color(second_marker, second_color),
            markeredgecolor="#1f1f1f",
            markeredgewidth=0.55,
            markersize=9.0,
            label=second_label,
        ),
    ]


def legend_marker_handle(
    *,
    marker: str,
    color: str,
    size_rate: float,
    markeredgecolor: str,
) -> Line2D:
    return Line2D(
        [0],
        [0],
        marker=marker,
        color="none",
        markerfacecolor=marker_face_color(marker, color),
        markeredgecolor=markeredgecolor,
        markeredgewidth=1.1 if marker != "*" else 0.55,
        markersize=float(np.sqrt(marker_sizes(np.asarray([size_rate]))[0])),
    )


def marker_face_color(marker: str, color: str) -> str:
    return color if marker == "*" else "white"


def marker_sizes(
    rates: np.ndarray,
    *,
    base: float = 20.0,
    scale: float = 280.0,
) -> np.ndarray:
    return base + scale * np.sqrt(np.clip(rates, 0.0, 1.0))


def plot_coverage_panel(
    ax: plt.Axes,
    geography: gpd.GeoDataFrame,
    *,
    first_covered: set[str],
    second_covered: set[str],
    first_selected: Sequence[str],
    second_selected: Sequence[str],
    first_label: str,
    second_label: str,
    first_color: str,
    second_color: str,
    title: str,
    legend_loc: str = "lower left",
) -> None:
    plot_geo = geography.copy()
    plot_geo["coverage_category"] = plot_geo["zip_code"].astype(str).map(
        lambda zip_code: coverage_category(zip_code, first_covered, second_covered)
    )
    color_map = {
        "neither": BACKGROUND_ZONE_COLOR,
        "both": BOTH_COVERED_COLOR,
        "first_only": first_color,
        "second_only": second_color,
    }
    alpha_map = {
        "neither": 1.0,
        "both": 0.34,
        "first_only": 0.36,
        "second_only": 0.36,
    }
    for category in ("neither", "both", "first_only", "second_only"):
        category_geo = plot_geo.loc[plot_geo["coverage_category"].eq(category)]
        if category_geo.empty:
            continue
        category_geo.plot(
            ax=ax,
            color=color_map[category],
            edgecolor=BASE_EDGE_COLOR,
            linewidth=0.35,
            alpha=alpha_map[category],
        )
    plot_geo.boundary.plot(ax=ax, color="white", linewidth=0.18, alpha=0.85)

    first_boundary = plot_geo.loc[plot_geo["zip_code"].astype(str).isin(first_covered)]
    second_boundary = plot_geo.loc[plot_geo["zip_code"].astype(str).isin(second_covered)]
    if not first_boundary.empty:
        first_boundary.boundary.plot(
            ax=ax,
            color=first_color,
            linewidth=1.05,
            alpha=0.86,
        )
    if not second_boundary.empty:
        second_boundary.boundary.plot(
            ax=ax,
            color=second_color,
            linewidth=1.05,
            alpha=0.90,
        )

    plot_selected_markers(
        ax,
        geography,
        selected_zip_codes=first_selected,
        marker="o",
        label=first_label,
        color=first_color,
        size=50,
    )
    plot_selected_markers(
        ax,
        geography,
        selected_zip_codes=second_selected,
        marker="*",
        label=second_label,
        color=second_color,
        size=105,
    )
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(
        handles=[
            Patch(facecolor=BOTH_COVERED_COLOR, edgecolor="none", alpha=0.34, label="Covered by both"),
            Patch(facecolor=first_color, edgecolor="none", alpha=0.36, label=f"{first_label} only"),
            Patch(facecolor=second_color, edgecolor="none", alpha=0.36, label=f"{second_label} only"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=first_color,
                markeredgewidth=1.5,
                markersize=6.2,
                label=f"{first_label} staging ZIP",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor=second_color,
                markeredgecolor="#1f1f1f",
                markeredgewidth=0.55,
                markersize=9.0,
                label=f"{second_label} staging ZIP",
            ),
        ],
        loc=legend_loc,
        fontsize=7.1,
        frameon=True,
        framealpha=0.92,
        borderpad=0.42,
        handlelength=1.35,
    )


def coverage_category(
    zip_code: str,
    first_covered: set[str],
    second_covered: set[str],
) -> str:
    in_first = zip_code in first_covered
    in_second = zip_code in second_covered
    if in_first and in_second:
        return "both"
    if in_first:
        return "first_only"
    if in_second:
        return "second_only"
    return "neither"


def plot_selected_markers(
    ax: plt.Axes,
    geography: gpd.GeoDataFrame,
    *,
    selected_zip_codes: Sequence[str],
    marker: str,
    label: str,
    color: str,
    size: float,
) -> None:
    selected = geography.loc[geography["zip_code"].astype(str).isin(set(selected_zip_codes))]
    if selected.empty:
        return
    points = selected.geometry.representative_point()
    ax.scatter(
        points.x,
        points.y,
        s=size,
        marker=marker,
        facecolors="white" if marker != "*" else color,
        edgecolors=color if marker != "*" else "#1f1f1f",
        linewidths=1.35 if marker != "*" else 0.55,
        zorder=12,
        label=label,
    )


def add_feature_shift_annotation(
    ax: plt.Axes,
    geography: gpd.GeoDataFrame,
    representative: RepresentativeCase,
) -> None:
    neutralized_set = set(representative.neutralized_selected_zip_codes)
    full_set = set(representative.full_selected_zip_codes)
    removed = tuple(sorted(neutralized_set - full_set))
    added = tuple(sorted(full_set - neutralized_set))
    demand_delta = representative.full_covered_demand - representative.neutralized_covered_demand
    feature_label = FEATURE_LABELS.get(representative.feature, representative.feature)

    if removed and added:
        extra_swaps = max(len(removed), len(added)) - 1
        moved_text = f"{feature_label} moved one unit from {removed[0]} to {added[0]}"
        if extra_swaps > 0:
            moved_text += f" (+{extra_swaps} more swap{'s' if extra_swaps != 1 else ''})"
        draw_shift_arrow(ax, geography, from_zip=removed[0], to_zip=added[0])
    else:
        moved_text = f"{feature_label} did not change selected staging ZIPs"
    annotation = f"{moved_text}\n{_demand_delta_label(demand_delta)} realized covered demand"
    ax.text(
        0.02,
        0.025,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        color=ANNOTATION_COLOR,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "#bdbdbd",
            "linewidth": 0.6,
            "alpha": 0.94,
        },
        zorder=20,
    )


def draw_shift_arrow(
    ax: plt.Axes,
    geography: gpd.GeoDataFrame,
    *,
    from_zip: str,
    to_zip: str,
) -> None:
    point_by_zip = {
        str(zip_code): point
        for zip_code, point in zip(
            geography["zip_code"].astype(str),
            geography.geometry.representative_point(),
        )
    }
    if from_zip not in point_by_zip or to_zip not in point_by_zip:
        return
    from_point = point_by_zip[from_zip]
    to_point = point_by_zip[to_zip]
    ax.annotate(
        "",
        xy=(to_point.x, to_point.y),
        xytext=(from_point.x, from_point.y),
        arrowprops={
            "arrowstyle": "->",
            "color": "#242424",
            "linewidth": 1.4,
            "shrinkA": 5,
            "shrinkB": 5,
            "alpha": 0.88,
        },
        zorder=19,
    )


def add_zip_labels(ax: plt.Axes, geography: gpd.GeoDataFrame) -> None:
    points = geography.geometry.representative_point()
    for zip_code, point in zip(geography["zip_code"].astype(str), points):
        ax.text(
            point.x,
            point.y,
            zip_code,
            ha="center",
            va="center",
            fontsize=4.2,
            color="#1f1f1f",
            alpha=0.78,
            zorder=30,
        )


def set_common_map_extent(
    axes: Sequence[plt.Axes],
    geography: gpd.GeoDataFrame,
    *,
    pad_fraction: float = 0.035,
) -> None:
    minx, miny, maxx, maxy = geography.total_bounds
    width = maxx - minx
    height = maxy - miny
    xpad = width * pad_fraction
    ypad = height * pad_fraction
    for ax in axes:
        ax.set_xlim(minx - xpad, maxx + xpad)
        ax.set_ylim(miny - ypad, maxy + ypad)


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "figure.dpi": 320,
            "savefig.dpi": 320,
        }
    )


def panel_a_title_label(metric: str) -> str:
    labels = {
        "full_minus_oracle_covered": "Mean full-minus-oracle covered incidents",
        "oracle_minus_full_uncovered": "Mean oracle-minus-full uncovered incidents",
        "oracle_minus_baseline_uncovered": "Mean oracle-minus-baseline uncovered incidents",
        "oracle_uncovered_demand": "Mean oracle-uncovered incidents",
        "realized_demand": "Mean realized incidents",
    }
    return labels[metric]


def panel_a_colorbar_label(metric: str) -> str:
    labels = {
        "full_minus_oracle_covered": "FM Covered - Oracle Covered",
        "oracle_minus_full_uncovered": "Oracle Covered - FM Covered",
        "oracle_minus_baseline_uncovered": "Oracle Covered - Baseline Covered",
        "oracle_uncovered_demand": "Oracle-uncovered incidents per hour",
        "realized_demand": "Realized incidents per hour",
    }
    return labels[metric]


def load_hourly_shap(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "hourly_shap.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing hourly_shap.csv: {path}")
    frame = pd.read_csv(path)
    required_columns = {
        "timestamp_hour",
        "actual_total_demand",
        "baseline_selected_zip_codes",
        "full_selected_zip_codes",
        "oracle_selected_zip_codes",
        "baseline_covered_zip_codes",
        "full_covered_zip_codes",
        "oracle_covered_zip_codes",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    frame["timestamp_hour"] = pd.to_datetime(frame["timestamp_hour"])
    return frame


def load_coalition_values(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "coalition_values.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing coalition_values.csv: {path}")
    frame = pd.read_csv(
        path,
        usecols=[
            "timestamp_hour",
            "coalition_mask",
            "decision_value",
            "decision_selected_facility_zip_codes",
        ],
    )
    frame["timestamp_hour"] = pd.to_datetime(frame["timestamp_hour"])
    frame["coalition_mask"] = frame["coalition_mask"].astype(int)
    return frame


def load_player_names(run_dir: Path, hourly: pd.DataFrame) -> tuple[str, ...]:
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        player_names = tuple(str(player) for player in metadata.get("player_names", ()))
        if player_names:
            return player_names
    feature_names = [
        column.removeprefix("decision_shap_")
        for column in hourly.columns
        if str(column).startswith("decision_shap_")
    ]
    return tuple(
        feature
        for feature in PREFERRED_FEATURE_ORDER
        if feature in set(feature_names)
    ) + tuple(
        feature
        for feature in feature_names
        if feature not in set(PREFERRED_FEATURE_ORDER)
    )


def feature_candidates_for_row(
    row: dict[str, Any],
    *,
    player_names: Sequence[str],
    feature: str,
    feature_sign: str,
) -> tuple[str, ...]:
    if feature != "auto":
        return (feature,)
    values: list[tuple[str, float]] = []
    for feature_name in player_names:
        value = float(row.get(f"decision_shap_{feature_name}", 0.0))
        if feature_sign == "positive" and value <= 0:
            continue
        if feature_sign == "negative" and value >= 0:
            continue
        values.append((feature_name, value))
    if not values:
        values = [
            (feature_name, float(row.get(f"decision_shap_{feature_name}", 0.0)))
            for feature_name in player_names
        ]
    preferred_rank = {feature_name: idx for idx, feature_name in enumerate(PREFERRED_FEATURE_ORDER)}
    values.sort(
        key=lambda item: (
            -abs(item[1]),
            preferred_rank.get(item[0], len(preferred_rank)),
            item[0],
        )
    )
    return tuple(feature_name for feature_name, _ in values)


def select_coalition_row(
    coalition_values: pd.DataFrame,
    *,
    timestamp_hour: pd.Timestamp,
    coalition_mask: int,
) -> dict[str, Any] | None:
    frame = coalition_values.loc[
        coalition_values["timestamp_hour"].eq(timestamp_hour)
        & coalition_values["coalition_mask"].eq(coalition_mask)
    ]
    if frame.empty:
        return None
    return frame.iloc[0].to_dict()


def load_target_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing EMS target table: {path}")
    frame = pd.read_csv(path)
    if "timestamp_hour" not in frame.columns:
        raise KeyError(f"{path} is missing timestamp_hour.")
    frame["timestamp_hour"] = pd.to_datetime(frame["timestamp_hour"])
    return frame.set_index("timestamp_hour").sort_index()


def load_distance_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing EMS distance matrix: {path}")
    matrix = pd.read_parquet(path)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    return matrix


def target_row_by_zip(row: pd.Series) -> pd.Series:
    values: dict[str, float] = {}
    prefix = "target_ems_incident_count_zip_"
    for column, value in row.items():
        column_name = str(column)
        if column_name.startswith(prefix):
            values[column_name.removeprefix(prefix)] = float(value)
    return pd.Series(values, dtype=float)


def coverage_zip_codes(
    selected_zip_codes: Sequence[str],
    *,
    distance_matrix: pd.DataFrame,
    coverage_radius_km: float,
) -> tuple[str, ...]:
    selected = [zip_code for zip_code in selected_zip_codes if zip_code in distance_matrix.index]
    if not selected:
        return ()
    covered = distance_matrix.loc[selected].le(coverage_radius_km + 1e-9).any(axis=0)
    return tuple(sorted(covered.index[covered].astype(str)))


def realized_covered_demand(
    actual_demand: pd.Series,
    covered_zip_codes: Iterable[str],
) -> float:
    covered = [zip_code for zip_code in covered_zip_codes if zip_code in actual_demand.index]
    if not covered:
        return 0.0
    return float(actual_demand.loc[covered].sum())


def load_zip_zone_geography(
    *,
    geojson_path: Path,
    geography_csv_path: Path,
) -> tuple[gpd.GeoDataFrame, str]:
    if geojson_path.exists():
        geography = gpd.read_file(geojson_path)
        if "zip_code" in geography.columns:
            geography = geography.copy()
            geography["zip_code"] = geography["zip_code"].astype(str)
        elif {"modzcta", "label", "zcta"}.issubset(geography.columns):
            geography = build_model_zip_geography_from_modzcta(
                geography,
                geography_csv_path=geography_csv_path,
            )
        else:
            raise KeyError(
                f"{geojson_path} must include zip_code or raw MODZCTA columns "
                "modzcta, label, and zcta."
            )
        return (
            geography.sort_values("zip_code").reset_index(drop=True),
            str(geojson_path),
        )
    geography = build_centroid_voronoi_geography(geography_csv_path)
    return geography, f"approximate centroid Voronoi from {geography_csv_path}"


def build_model_zip_geography_from_modzcta(
    modzcta: gpd.GeoDataFrame,
    *,
    geography_csv_path: Path,
) -> gpd.GeoDataFrame:
    if not geography_csv_path.exists():
        raise FileNotFoundError(f"Missing EMS geography CSV: {geography_csv_path}")
    model_geography = pd.read_csv(geography_csv_path, dtype={"zip_code": str})
    if "zip_code" not in model_geography.columns:
        raise KeyError(f"{geography_csv_path} is missing zip_code.")
    model_zip_codes = set(model_geography["zip_code"].astype(str))

    rows: list[dict[str, Any]] = []
    for row in modzcta.itertuples(index=False):
        alias_zip_codes = extract_zip_codes(
            getattr(row, "modzcta"),
            getattr(row, "label"),
            getattr(row, "zcta"),
        )
        canonical_zip_code = choose_canonical_zip_code(
            getattr(row, "modzcta"),
            alias_zip_codes,
        )
        if canonical_zip_code not in model_zip_codes:
            continue
        rows.append(
            {
                "zip_code": canonical_zip_code,
                "modzcta": str(getattr(row, "modzcta")),
                "alias_zip_codes": "|".join(sorted(alias_zip_codes)),
                "alias_zip_count": len(alias_zip_codes),
                "pop_est": getattr(row, "pop_est", None),
                "geometry": getattr(row, "geometry"),
            }
        )

    if not rows:
        raise ValueError(
            "No raw MODZCTA polygons match the processed EMS ZIP universe."
        )

    geography = gpd.GeoDataFrame(rows, crs=modzcta.crs)
    geography = geography.dissolve(
        by="zip_code",
        as_index=False,
        aggfunc={
            "modzcta": "first",
            "alias_zip_codes": "first",
            "alias_zip_count": "first",
            "pop_est": "first",
        },
    )
    missing_zip_codes = sorted(model_zip_codes - set(geography["zip_code"].astype(str)))
    if missing_zip_codes:
        raise ValueError(
            "MODZCTA polygons are missing model ZIPs: "
            + ", ".join(missing_zip_codes[:10])
        )
    if geography.crs is None:
        geography = geography.set_crs("EPSG:4326")
    geography["zip_code"] = geography["zip_code"].astype(str)
    return geography.to_crs("EPSG:4326")


def extract_zip_codes(*values: object) -> set[str]:
    zip_codes: set[str] = set()
    for value in values:
        zip_codes.update(re.findall(r"\b\d{5}\b", str(value)))
    return zip_codes


def choose_canonical_zip_code(modzcta_value: object, alias_zip_codes: set[str]) -> str:
    modzcta_zip_codes = extract_zip_codes(modzcta_value)
    if len(modzcta_zip_codes) == 1:
        return next(iter(modzcta_zip_codes))
    if not alias_zip_codes:
        raise ValueError(f"Could not resolve a canonical ZIP for {modzcta_value!r}.")
    return sorted(alias_zip_codes)[0]


def build_centroid_voronoi_geography(geography_csv_path: Path) -> gpd.GeoDataFrame:
    if not geography_csv_path.exists():
        raise FileNotFoundError(f"Missing EMS geography CSV: {geography_csv_path}")
    frame = pd.read_csv(geography_csv_path, dtype={"zip_code": str})
    required_columns = {"zip_code", "centroid_lon", "centroid_lat"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise KeyError(
            f"{geography_csv_path} is missing columns: {', '.join(sorted(missing))}"
        )
    points = gpd.GeoDataFrame(
        frame.copy(),
        geometry=gpd.points_from_xy(frame["centroid_lon"], frame["centroid_lat"]),
        crs="EPSG:4326",
    ).to_crs(NYC_AREA_CRS)
    coordinates = np.column_stack([points.geometry.x.to_numpy(), points.geometry.y.to_numpy()])
    voronoi = Voronoi(coordinates)
    regions, vertices = voronoi_finite_polygons_2d(voronoi)
    boundary = MultiPoint([tuple(coordinate) for coordinate in coordinates]).convex_hull.buffer(
        4200.0
    )

    polygons = []
    for region in regions:
        polygon = Polygon(vertices[region]).intersection(boundary)
        if polygon.is_empty:
            polygon = Polygon(vertices[region]).buffer(0)
        polygons.append(polygon)

    zone_geo = gpd.GeoDataFrame(points.drop(columns="geometry"), geometry=polygons, crs=NYC_AREA_CRS)
    zone_geo = zone_geo.to_crs("EPSG:4326")
    zone_geo["zip_code"] = zone_geo["zip_code"].astype(str)
    return zone_geo.sort_values("zip_code").reset_index(drop=True)


def voronoi_finite_polygons_2d(
    voronoi: Voronoi,
    *,
    radius: float | None = None,
) -> tuple[list[list[int]], np.ndarray]:
    if voronoi.points.shape[1] != 2:
        raise ValueError("Only 2D Voronoi diagrams are supported.")
    if radius is None:
        radius = float(np.ptp(voronoi.points, axis=0).max() * 2.0)

    new_regions: list[list[int]] = []
    new_vertices = voronoi.vertices.tolist()
    center = voronoi.points.mean(axis=0)
    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (point_a, point_b), (vertex_a, vertex_b) in zip(
        voronoi.ridge_points,
        voronoi.ridge_vertices,
    ):
        all_ridges.setdefault(point_a, []).append((point_b, vertex_a, vertex_b))
        all_ridges.setdefault(point_b, []).append((point_a, vertex_a, vertex_b))

    for point_idx, region_idx in enumerate(voronoi.point_region):
        vertices = voronoi.regions[region_idx]
        if all(vertex_idx >= 0 for vertex_idx in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[point_idx]
        new_region = [vertex_idx for vertex_idx in vertices if vertex_idx >= 0]
        for point_b, vertex_a, vertex_b in ridges:
            if vertex_a >= 0 and vertex_b >= 0:
                continue
            finite_vertex = vertex_a if vertex_a >= 0 else vertex_b
            tangent = voronoi.points[point_b] - voronoi.points[point_idx]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = voronoi.points[[point_idx, point_b]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = voronoi.vertices[finite_vertex] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        region_vertices = np.asarray([new_vertices[vertex_idx] for vertex_idx in new_region])
        centroid = region_vertices.mean(axis=0)
        angles = np.arctan2(
            region_vertices[:, 1] - centroid[1],
            region_vertices[:, 0] - centroid[0],
        )
        new_region = [vertex for _, vertex in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def _parse_zip_code_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    if value is None or pd.isna(value):
        return ()
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list of ZIP codes, got {value!r}.")
    return tuple(str(item) for item in parsed)


def _resolve_manifest_path(manifest_path: Path, path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return path


def _number_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def _demand_delta_label(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.1f}"


def representative_to_metadata(representative: RepresentativeCase) -> dict[str, Any]:
    return {
        "model_id": representative.run.model_id,
        "setting_id": representative.run.setting_id,
        "results_dir": str(representative.run.results_dir),
        "timestamp_hour": str(representative.timestamp_hour),
        "feature": representative.feature,
        "feature_label": FEATURE_LABELS.get(representative.feature, representative.feature),
        "feature_dva_value": representative.feature_dva_value,
        "selection_score": representative.score,
        "actual_total_demand": representative.actual_total_demand,
        "baseline_selected_zip_codes": representative.baseline_selected_zip_codes,
        "full_selected_zip_codes": representative.full_selected_zip_codes,
        "oracle_selected_zip_codes": representative.oracle_selected_zip_codes,
        "neutralized_selected_zip_codes": representative.neutralized_selected_zip_codes,
        "baseline_covered_demand": representative.baseline_covered_demand,
        "full_covered_demand": representative.full_covered_demand,
        "oracle_covered_demand": representative.oracle_covered_demand,
        "neutralized_covered_demand": representative.neutralized_covered_demand,
        "baseline_to_full_covered_demand_delta": (
            representative.full_covered_demand - representative.baseline_covered_demand
        ),
        "neutralized_to_full_covered_demand_delta": (
            representative.full_covered_demand - representative.neutralized_covered_demand
        ),
        "geography_source": representative.geography_source,
    }


def representative_context_frame(representative: RepresentativeCase) -> pd.DataFrame:
    neutralized_set = set(representative.neutralized_selected_zip_codes)
    full_set = set(representative.full_selected_zip_codes)
    baseline_set = set(representative.baseline_selected_zip_codes)
    return pd.DataFrame(
        [
            {
                "model_id": representative.run.model_id,
                "setting_id": representative.run.setting_id,
                "timestamp_hour": representative.timestamp_hour,
                "neutralized_feature": representative.feature,
                "neutralized_feature_label": FEATURE_LABELS.get(
                    representative.feature,
                    representative.feature,
                ),
                "feature_dva_value": representative.feature_dva_value,
                "actual_total_demand": representative.actual_total_demand,
                "baseline_covered_demand": representative.baseline_covered_demand,
                "full_covered_demand": representative.full_covered_demand,
                "neutralized_covered_demand": (
                    representative.neutralized_covered_demand
                ),
                "oracle_covered_demand": representative.oracle_covered_demand,
                "full_minus_baseline_covered_demand": (
                    representative.full_covered_demand
                    - representative.baseline_covered_demand
                ),
                "full_minus_neutralized_covered_demand": (
                    representative.full_covered_demand
                    - representative.neutralized_covered_demand
                ),
                "full_minus_oracle_covered_demand": (
                    representative.full_covered_demand
                    - representative.oracle_covered_demand
                ),
                "baseline_selected_zip_codes": "|".join(
                    representative.baseline_selected_zip_codes
                ),
                "full_selected_zip_codes": "|".join(
                    representative.full_selected_zip_codes
                ),
                "neutralized_selected_zip_codes": "|".join(
                    representative.neutralized_selected_zip_codes
                ),
                "oracle_selected_zip_codes": "|".join(
                    representative.oracle_selected_zip_codes
                ),
                "baseline_to_full_removed_zip_codes": "|".join(
                    sorted(baseline_set - full_set)
                ),
                "baseline_to_full_added_zip_codes": "|".join(
                    sorted(full_set - baseline_set)
                ),
                "neutralized_to_full_removed_zip_codes": "|".join(
                    sorted(neutralized_set - full_set)
                ),
                "neutralized_to_full_added_zip_codes": "|".join(
                    sorted(full_set - neutralized_set)
                ),
                "geography_source": representative.geography_source,
            }
        ]
    )


if __name__ == "__main__":
    main()
