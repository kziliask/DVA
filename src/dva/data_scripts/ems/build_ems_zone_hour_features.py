from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


DEFAULT_RAW_PATH = Path("data/ems_data/EMS_Incident_Dispatch_Data_20260501.csv")
DEFAULT_TAXI_FEATURES_PATH = Path(
    "data/nyc_data/processed/nyc_taxi_zone_hour_features_2025_manhattan.csv",
)
DEFAULT_ZONE_LOOKUP_PATH = Path("data/nyc_data/interim/taxi_zone_lookup_enriched.parquet")
DEFAULT_MODZCTA_PATH = Path("data/nyc_data/interim/modzcta_boundaries.geojson")
DEFAULT_OUTPUT_DIR = Path("data/ems_data/processed")
DEFAULT_BASELINE_WEEKS = 8
DEFAULT_CHUNKSIZE = 750_000
DEFAULT_EXCLUDED_ZIP_CODES = ("10468",)
EMS_DATETIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
NYC_AREA_CRS = "EPSG:2263"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Manhattan taxi-zone hourly EMS incident feature table for set-covering "
            "experiments."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--taxi-features-path", type=Path, default=DEFAULT_TAXI_FEATURES_PATH)
    parser.add_argument("--zone-lookup-path", type=Path, default=DEFAULT_ZONE_LOOKUP_PATH)
    parser.add_argument("--modzcta-path", type=Path, default=DEFAULT_MODZCTA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--baseline-weeks",
        type=int,
        default=DEFAULT_BASELINE_WEEKS,
        help="Number of previous same hour-of-week observations to average.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNKSIZE,
        help="EMS CSV rows per chunk.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional output start timestamp. Defaults to the first taxi feature hour.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional output end timestamp. Defaults to min(taxi max hour, EMS max hour).",
    )
    parser.add_argument(
        "--exclude-zip-codes",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_ZIP_CODES),
        help="ZIP codes to remove from the Manhattan EMS analysis universe.",
    )
    return parser.parse_args()


def load_taxi_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing taxi feature table: {path}")

    taxi = pd.read_csv(
        path,
        usecols=[
            "timestamp_hour",
            "zone_id",
            "zone_name",
            "borough",
            "service_zone",
            "centroid_lon",
            "centroid_lat",
            "zone_area_km2",
            "pickup_count_lag_1",
            "temp_c",
            "precip_mm",
        ],
    )
    taxi["timestamp_hour"] = pd.to_datetime(taxi["timestamp_hour"])
    taxi["zone_id"] = taxi["zone_id"].astype(int)
    return taxi.rename(columns={"pickup_count_lag_1": "taxi_pickups_lag_1"})


def load_manhattan_zones(path: Path, zone_ids: set[int]) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing taxi zone lookup: {path}")

    zones = gpd.read_parquet(path)
    zones["zone_id"] = zones["zone_id"].astype(int)
    zones = zones.loc[zones["zone_id"].isin(zone_ids)].copy()
    zones = zones.sort_values("zone_id").reset_index(drop=True)
    if zones.empty:
        raise ValueError("No Manhattan taxi zones found for the taxi feature zone IDs.")
    return zones


def iter_zip_codes(*values: object) -> set[str]:
    zip_codes: set[str] = set()
    for value in values:
        zip_codes.update(re.findall(r"\b\d{5}\b", str(value)))
    return zip_codes


def normalize_zip_codes(zip_codes: object) -> set[str]:
    if zip_codes is None:
        return set()
    if isinstance(zip_codes, str):
        return iter_zip_codes(zip_codes)
    if not isinstance(zip_codes, Iterable):
        return iter_zip_codes(zip_codes)
    normalized: set[str] = set()
    for zip_code in zip_codes:
        normalized.update(iter_zip_codes(zip_code))
    return normalized


def build_zip_to_zone_crosswalk(
    zones: gpd.GeoDataFrame,
    modzcta_path: Path,
    *,
    excluded_zip_codes: object = DEFAULT_EXCLUDED_ZIP_CODES,
) -> pd.DataFrame:
    if not modzcta_path.exists():
        raise FileNotFoundError(
            f"Missing MODZCTA geography: {modzcta_path}. "
            "Download it from https://data.cityofnewyork.us/resource/pri4-ifjk.geojson",
        )

    modzcta = gpd.read_file(modzcta_path)
    required_columns = {"modzcta", "label", "zcta", "geometry"}
    missing_columns = required_columns.difference(modzcta.columns)
    if missing_columns:
        raise ValueError(
            f"MODZCTA file is missing required columns: {sorted(missing_columns)}",
        )

    modzcta = modzcta[["modzcta", "label", "zcta", "geometry"]].copy()
    modzcta["modzcta"] = modzcta["modzcta"].astype(str)
    zones_for_overlay = zones[
        ["zone_id", "zone_name", "borough", "service_zone", "geometry"]
    ].copy()

    if modzcta.crs is None:
        modzcta = modzcta.set_crs("EPSG:4326")
    if zones_for_overlay.crs is None:
        zones_for_overlay = zones_for_overlay.set_crs("EPSG:4326")

    modzcta = modzcta.to_crs(NYC_AREA_CRS)
    zones_for_overlay = zones_for_overlay.to_crs(NYC_AREA_CRS)

    overlaps = gpd.overlay(
        modzcta,
        zones_for_overlay,
        how="intersection",
        keep_geom_type=False,
    )
    if overlaps.empty:
        raise ValueError("MODZCTA polygons did not overlap the Manhattan taxi zones.")

    overlaps["overlap_area_km2"] = overlaps.geometry.area / 1_000_000
    overlaps = overlaps.loc[overlaps["overlap_area_km2"] > 0].copy()

    excluded = normalize_zip_codes(excluded_zip_codes)
    zip_rows: list[dict[str, object]] = []
    for row in overlaps.itertuples(index=False):
        zip_codes = iter_zip_codes(row.modzcta, row.label, row.zcta) - excluded
        for zip_code in zip_codes:
            zip_rows.append(
                {
                    "zip_code": zip_code,
                    "modzcta": str(row.modzcta),
                    "zone_id": int(row.zone_id),
                    "zone_name": str(row.zone_name),
                    "borough": str(row.borough),
                    "service_zone": str(row.service_zone),
                    "overlap_area_km2": float(row.overlap_area_km2),
                },
            )

    if not zip_rows:
        raise ValueError("No ZIP-to-zone overlaps remain after applying exclusions.")

    crosswalk = pd.DataFrame(zip_rows)
    crosswalk = (
        crosswalk.groupby(
            ["zip_code", "modzcta", "zone_id", "zone_name", "borough", "service_zone"],
            as_index=False,
            observed=True,
        )["overlap_area_km2"]
        .sum()
        .sort_values(["zip_code", "zone_id"])
        .reset_index(drop=True)
    )
    crosswalk["zip_total_overlap_area_km2"] = crosswalk.groupby("zip_code")[
        "overlap_area_km2"
    ].transform("sum")
    crosswalk["zip_zone_weight"] = (
        crosswalk["overlap_area_km2"] / crosswalk["zip_total_overlap_area_km2"]
    )
    return crosswalk


def parse_hour_series(datetime_series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        datetime_series,
        format=EMS_DATETIME_FORMAT,
        errors="coerce",
    ).dt.floor("h")


def aggregate_ems_csv(
    raw_path: Path,
    *,
    history_start: pd.Timestamp,
    read_end: pd.Timestamp,
    chunksize: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing EMS raw CSV: {raw_path}")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")

    read_end_exclusive = read_end + pd.Timedelta(hours=1)
    candidate_years = {
        str(year)
        for year in range(history_start.year, read_end_exclusive.year + 1)
    }
    zip_hour_frames: list[pd.DataFrame] = []
    citywide_hour_frames: list[pd.DataFrame] = []
    stats = {
        "raw_rows_scanned": 0.0,
        "rows_in_year_candidates": 0.0,
        "rows_in_time_window": 0.0,
        "manhattan_rows_in_time_window": 0.0,
        "manhattan_rows_with_valid_zip": 0.0,
    }

    usecols = ["INCIDENT_DATETIME", "BOROUGH", "ZIPCODE"]
    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            raw_path,
            usecols=usecols,
            dtype=str,
            chunksize=chunksize,
        ),
        start=1,
    ):
        stats["raw_rows_scanned"] += float(len(chunk))

        year_mask = chunk["INCIDENT_DATETIME"].str.slice(6, 10).isin(candidate_years)
        stats["rows_in_year_candidates"] += float(year_mask.sum())
        if not year_mask.any():
            continue

        chunk = chunk.loc[year_mask].copy()
        chunk["timestamp_hour"] = parse_hour_series(chunk["INCIDENT_DATETIME"])
        chunk = chunk.dropna(subset=["timestamp_hour"])
        chunk = chunk.loc[
            chunk["timestamp_hour"].ge(history_start)
            & chunk["timestamp_hour"].lt(read_end_exclusive)
        ].copy()
        stats["rows_in_time_window"] += float(len(chunk))
        if chunk.empty:
            continue

        citywide_hour_frames.append(
            chunk.groupby("timestamp_hour", observed=True)
            .size()
            .rename("citywide_ems_incidents")
            .reset_index(),
        )

        manhattan = chunk.loc[chunk["BOROUGH"].eq("MANHATTAN")].copy()
        stats["manhattan_rows_in_time_window"] += float(len(manhattan))
        if manhattan.empty:
            continue

        manhattan["zip_code"] = manhattan["ZIPCODE"].str.extract(r"(\d{5})", expand=False)
        manhattan = manhattan.dropna(subset=["zip_code"])
        stats["manhattan_rows_with_valid_zip"] += float(len(manhattan))
        if manhattan.empty:
            continue

        zip_hour_frames.append(
            manhattan.groupby(["timestamp_hour", "zip_code"], observed=True)
            .size()
            .rename("ems_incidents_raw")
            .reset_index(),
        )

        if chunk_idx % 5 == 0:
            print(
                f"Scanned {int(stats['raw_rows_scanned']):,} EMS rows; "
                f"{int(stats['manhattan_rows_with_valid_zip']):,} Manhattan rows with ZIPs.",
                flush=True,
            )

    if not citywide_hour_frames:
        raise ValueError("No EMS rows found in the requested time window.")

    citywide_hourly = (
        pd.concat(citywide_hour_frames, ignore_index=True)
        .groupby("timestamp_hour", as_index=False, observed=True)["citywide_ems_incidents"]
        .sum()
        .sort_values("timestamp_hour")
        .reset_index(drop=True)
    )
    if zip_hour_frames:
        zip_hourly = (
            pd.concat(zip_hour_frames, ignore_index=True)
            .groupby(["timestamp_hour", "zip_code"], as_index=False, observed=True)[
                "ems_incidents_raw"
            ]
            .sum()
            .sort_values(["timestamp_hour", "zip_code"])
            .reset_index(drop=True)
        )
    else:
        zip_hourly = pd.DataFrame(
            columns=["timestamp_hour", "zip_code", "ems_incidents_raw"],
        )

    return zip_hourly, citywide_hourly, stats


def allocate_zip_incidents_to_zones(
    zip_hourly: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if zip_hourly.empty:
        return (
            pd.DataFrame(columns=["timestamp_hour", "zone_id", "ems_incident_count"]),
            pd.DataFrame(columns=["zip_code", "unmapped_ems_incidents_raw"]),
        )

    known_zips = set(crosswalk["zip_code"].astype(str))
    unmatched = zip_hourly.loc[~zip_hourly["zip_code"].isin(known_zips)].copy()
    unmatched_totals = (
        unmatched.groupby("zip_code", as_index=False, observed=True)["ems_incidents_raw"]
        .sum()
        .rename(columns={"ems_incidents_raw": "unmapped_ems_incidents_raw"})
        .sort_values("unmapped_ems_incidents_raw", ascending=False)
        .reset_index(drop=True)
    )

    mapped = zip_hourly.merge(crosswalk, on="zip_code", how="inner")
    mapped["ems_incident_count"] = (
        mapped["ems_incidents_raw"].astype(float) * mapped["zip_zone_weight"].astype(float)
    )
    zone_hourly = (
        mapped.groupby(["timestamp_hour", "zone_id"], as_index=False, observed=True)[
            "ems_incident_count"
        ]
        .sum()
        .sort_values(["timestamp_hour", "zone_id"])
        .reset_index(drop=True)
    )
    return zone_hourly, unmatched_totals


def build_dense_panel(
    zone_ids: list[int],
    zone_hourly: pd.DataFrame,
    *,
    history_start: pd.Timestamp,
    output_end: pd.Timestamp,
) -> pd.DataFrame:
    full_hours = pd.date_range(history_start, output_end, freq="h")
    full_index = pd.MultiIndex.from_product(
        [full_hours, zone_ids],
        names=["timestamp_hour", "zone_id"],
    )
    panel = full_index.to_frame(index=False)
    panel = panel.merge(zone_hourly, on=["timestamp_hour", "zone_id"], how="left")
    panel["ems_incident_count"] = panel["ems_incident_count"].fillna(0.0)
    return panel.sort_values(["zone_id", "timestamp_hour"]).reset_index(drop=True)


def build_neighbor_adjacency(zones: gpd.GeoDataFrame, zone_ids: list[int]) -> pd.DataFrame:
    adjacency = pd.DataFrame(0.0, index=zone_ids, columns=zone_ids)
    valid_zone_ids = set(zone_ids)
    for row in zones[["zone_id", "neighbor_zone_ids"]].itertuples(index=False):
        neighbors = json.loads(row.neighbor_zone_ids) if pd.notna(row.neighbor_zone_ids) else []
        valid_neighbors = [int(neighbor) for neighbor in neighbors if int(neighbor) in valid_zone_ids]
        if valid_neighbors:
            adjacency.loc[valid_neighbors, int(row.zone_id)] = 1.0
    return adjacency


def add_ems_features(
    panel: pd.DataFrame,
    zones: gpd.GeoDataFrame,
    *,
    baseline_weeks: int,
) -> pd.DataFrame:
    if baseline_weeks <= 0:
        raise ValueError("baseline_weeks must be positive.")

    panel = panel.copy()
    panel["ems_incidents_lag_1"] = panel.groupby("zone_id", sort=False)[
        "ems_incident_count"
    ].shift(1)

    baseline_columns: list[str] = []
    for week_idx in range(1, baseline_weeks + 1):
        column = f"_ems_same_hour_week_lag_{week_idx}"
        panel[column] = panel.groupby("zone_id", sort=False)["ems_incident_count"].shift(
            168 * week_idx,
        )
        baseline_columns.append(column)
    panel["zone_hour_baseline"] = panel[baseline_columns].mean(axis=1)
    panel = panel.drop(columns=baseline_columns)

    zone_ids = sorted(panel["zone_id"].astype(int).unique().tolist())
    full_hours = pd.Index(panel["timestamp_hour"].drop_duplicates()).sort_values()
    adjacency = build_neighbor_adjacency(zones, zone_ids)
    degrees = adjacency.sum(axis=0).replace(0.0, 1.0)
    ems_wide = (
        panel.pivot(index="timestamp_hour", columns="zone_id", values="ems_incident_count")
        .reindex(index=full_hours, columns=zone_ids)
        .fillna(0.0)
        .sort_index(axis=1)
    )
    neighbor_lag_1 = ems_wide.shift(1).dot(adjacency).div(degrees, axis=1)
    neighbor_features = (
        neighbor_lag_1.stack(future_stack=True)
        .rename("neighbor_ems_incidents_lag_1_mean")
        .reset_index()
    )
    neighbor_features.columns = [
        "timestamp_hour",
        "zone_id",
        "neighbor_ems_incidents_lag_1_mean",
    ]

    panel = panel.merge(neighbor_features, on=["timestamp_hour", "zone_id"], how="left")
    panel["hour"] = panel["timestamp_hour"].dt.hour.astype(int)
    panel["month"] = panel["timestamp_hour"].dt.month.astype(int)
    panel["day_of_week"] = panel["timestamp_hour"].dt.dayofweek.astype(int)
    return panel


def add_citywide_feature(
    panel: pd.DataFrame,
    citywide_hourly: pd.DataFrame,
    *,
    history_start: pd.Timestamp,
    output_end: pd.Timestamp,
) -> pd.DataFrame:
    full_hours = pd.DataFrame(
        {"timestamp_hour": pd.date_range(history_start, output_end, freq="h")},
    )
    citywide = full_hours.merge(citywide_hourly, on="timestamp_hour", how="left")
    citywide["citywide_ems_incidents"] = citywide["citywide_ems_incidents"].fillna(0.0)
    citywide["citywide_ems_incidents_lag_1"] = citywide[
        "citywide_ems_incidents"
    ].shift(1)
    return panel.merge(
        citywide[["timestamp_hour", "citywide_ems_incidents_lag_1"]],
        on="timestamp_hour",
        how="left",
    )


def finalize_panel(
    panel: pd.DataFrame,
    taxi: pd.DataFrame,
    zones: gpd.GeoDataFrame,
    *,
    output_start: pd.Timestamp,
    output_end: pd.Timestamp,
) -> pd.DataFrame:
    taxi_features = taxi[
        [
            "timestamp_hour",
            "zone_id",
            "taxi_pickups_lag_1",
            "temp_c",
            "precip_mm",
        ]
    ].copy()
    static_features = zones[
        [
            "zone_id",
            "zone_name",
            "borough",
            "service_zone",
            "centroid_lon",
            "centroid_lat",
            "zone_area_km2",
        ]
    ].copy()

    panel = panel.merge(taxi_features, on=["timestamp_hour", "zone_id"], how="left")
    panel = panel.merge(static_features, on="zone_id", how="left")
    panel = panel.loc[
        panel["timestamp_hour"].ge(output_start) & panel["timestamp_hour"].le(output_end)
    ].copy()

    required_feature_columns = [
        "ems_incidents_lag_1",
        "neighbor_ems_incidents_lag_1_mean",
        "taxi_pickups_lag_1",
        "temp_c",
        "precip_mm",
        "zone_hour_baseline",
        "citywide_ems_incidents_lag_1",
        "hour",
        "month",
        "day_of_week",
    ]
    final_panel = panel.loc[panel[required_feature_columns].notna().all(axis=1)].copy()

    int_columns = ["zone_id", "hour", "month", "day_of_week"]
    for column in int_columns:
        final_panel[column] = final_panel[column].astype(int)

    ordered_columns = [
        "timestamp_hour",
        "zone_id",
        "zone_name",
        "borough",
        "service_zone",
        "centroid_lon",
        "centroid_lat",
        "zone_area_km2",
        "hour",
        "month",
        "day_of_week",
        "ems_incident_count",
        "ems_incidents_lag_1",
        "neighbor_ems_incidents_lag_1_mean",
        "taxi_pickups_lag_1",
        "temp_c",
        "precip_mm",
        "zone_hour_baseline",
        "citywide_ems_incidents_lag_1",
    ]
    return final_panel[ordered_columns].sort_values(
        ["timestamp_hour", "zone_id"],
    ).reset_index(drop=True)


def write_model_columns(final_panel: pd.DataFrame, output_dir: Path) -> Path:
    model_columns = [
        "hour",
        "month",
        "day_of_week",
        "ems_incidents_lag_1",
        "neighbor_ems_incidents_lag_1_mean",
        "taxi_pickups_lag_1",
        "temp_c",
        "precip_mm",
        "zone_hour_baseline",
        "citywide_ems_incidents_lag_1",
        "ems_incident_count",
    ]
    model_path = output_dir / "ems_zone_hour_features_2025_manhattan_model_cols.csv"
    final_panel[model_columns].to_csv(model_path, index=False)
    return model_path


def main() -> None:
    args = parse_args()
    taxi = load_taxi_features(args.taxi_features_path)
    zone_ids = sorted(taxi["zone_id"].astype(int).unique().tolist())
    zones = load_manhattan_zones(args.zone_lookup_path, set(zone_ids))

    taxi_min_hour = pd.Timestamp(taxi["timestamp_hour"].min())
    taxi_max_hour = pd.Timestamp(taxi["timestamp_hour"].max())
    output_start = pd.Timestamp(args.start) if args.start else taxi_min_hour
    requested_end = pd.Timestamp(args.end) if args.end else taxi_max_hour
    if requested_end < output_start:
        raise ValueError("Output end must be greater than or equal to output start.")

    history_start = output_start - pd.Timedelta(weeks=args.baseline_weeks)
    print(
        f"Building EMS panel for {len(zone_ids)} Manhattan taxi zones from "
        f"{output_start} with {args.baseline_weeks} baseline weeks.",
        flush=True,
    )

    excluded_zip_codes = normalize_zip_codes(args.exclude_zip_codes)
    crosswalk = build_zip_to_zone_crosswalk(
        zones,
        args.modzcta_path,
        excluded_zip_codes=excluded_zip_codes,
    )
    zip_hourly, citywide_hourly, scan_stats = aggregate_ems_csv(
        args.raw_path,
        history_start=history_start,
        read_end=requested_end,
        chunksize=args.chunksize,
    )
    if excluded_zip_codes:
        zip_hourly = zip_hourly.loc[
            ~zip_hourly["zip_code"].astype(str).isin(excluded_zip_codes)
        ].copy()
    zone_hourly, unmatched_zip_totals = allocate_zip_incidents_to_zones(
        zip_hourly,
        crosswalk,
    )

    ems_max_hour = pd.Timestamp(citywide_hourly["timestamp_hour"].max())
    output_end = min(requested_end, ems_max_hour, taxi_max_hour)
    dense_panel = build_dense_panel(
        zone_ids,
        zone_hourly,
        history_start=history_start,
        output_end=output_end,
    )
    dense_panel = add_ems_features(
        dense_panel,
        zones,
        baseline_weeks=args.baseline_weeks,
    )
    dense_panel = add_citywide_feature(
        dense_panel,
        citywide_hourly,
        history_start=history_start,
        output_end=output_end,
    )
    final_panel = finalize_panel(
        dense_panel,
        taxi,
        zones,
        output_start=output_start,
        output_end=output_end,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "ems_zone_hour_features_2025_manhattan.csv"
    crosswalk_path = output_dir / "ems_zip_to_taxi_zone_crosswalk.csv"
    unmatched_path = output_dir / "ems_unmapped_zip_totals.csv"
    summary_path = output_dir / "ems_zone_hour_features_2025_manhattan_summary.json"

    final_panel.to_csv(feature_path, index=False)
    model_path = write_model_columns(final_panel, output_dir)
    crosswalk.to_csv(crosswalk_path, index=False)
    unmatched_zip_totals.to_csv(unmatched_path, index=False)

    mapped_incidents = float(zone_hourly["ems_incident_count"].sum())
    raw_manhattan_zip_incidents = float(zip_hourly["ems_incidents_raw"].sum())
    unmapped_incidents = float(unmatched_zip_totals["unmapped_ems_incidents_raw"].sum())
    summary = {
        **scan_stats,
        "baseline_weeks": float(args.baseline_weeks),
        "zone_count": float(len(zone_ids)),
        "output_hour_count": float(final_panel["timestamp_hour"].nunique()),
        "output_row_count": float(len(final_panel)),
        "output_start": str(final_panel["timestamp_hour"].min()),
        "output_end": str(final_panel["timestamp_hour"].max()),
        "history_start": str(history_start),
        "ems_raw_max_hour_in_window": str(ems_max_hour),
        "taxi_feature_min_hour": str(taxi_min_hour),
        "taxi_feature_max_hour": str(taxi_max_hour),
        "raw_manhattan_zip_incidents_in_window": raw_manhattan_zip_incidents,
        "mapped_manhattan_incidents_in_window": mapped_incidents,
        "unmapped_manhattan_zip_incidents_in_window": unmapped_incidents,
        "mapped_share_of_manhattan_zip_incidents": (
            mapped_incidents / raw_manhattan_zip_incidents
            if raw_manhattan_zip_incidents
            else np.nan
        ),
        "crosswalk_zip_count": float(crosswalk["zip_code"].nunique()),
        "source_raw_path": str(args.raw_path),
        "source_taxi_features_path": str(args.taxi_features_path),
        "source_modzcta_path": str(args.modzcta_path),
    }
    pd.Series(summary).to_json(summary_path, indent=2)

    print(f"Wrote EMS feature table to {feature_path}")
    print(f"Wrote modeling-only columns to {model_path}")
    print(f"Wrote ZIP-to-zone crosswalk to {crosswalk_path}")
    print(f"Wrote unmapped ZIP totals to {unmatched_path}")
    print(f"Wrote validation summary to {summary_path}")
    print(
        f"Final rows: {len(final_panel):,}; "
        f"hours: {final_panel['timestamp_hour'].nunique():,}; "
        f"zones: {final_panel['zone_id'].nunique():,}",
    )


if __name__ == "__main__":
    main()
