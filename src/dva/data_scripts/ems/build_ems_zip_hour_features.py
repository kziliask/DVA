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
DEFAULT_WEATHER_PATH = Path("data/nyc_data/interim/weather_hourly_nyc.csv")
DEFAULT_ZONE_LOOKUP_PATH = Path("data/nyc_data/interim/taxi_zone_lookup_enriched.parquet")
DEFAULT_MODZCTA_PATH = Path("data/nyc_data/interim/modzcta_boundaries.geojson")
DEFAULT_OUTPUT_DIR = Path("data/ems_data/processed")
DEFAULT_BASELINE_WEEKS = 8
DEFAULT_CHUNKSIZE = 750_000
DEFAULT_OUTPUT_START = pd.Timestamp("2025-01-01 00:00:00")
DEFAULT_EXCLUDED_ZIP_CODES = ("10468",)
EMS_DATETIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
NYC_AREA_CRS = "EPSG:2263"
EARTH_RADIUS_KM = 6371.0088


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a dense Manhattan EMS ZIP-hour table and ZIP centroid distance "
            "matrix for set-covering experiments."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--weather-path", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--zone-lookup-path", type=Path, default=DEFAULT_ZONE_LOOKUP_PATH)
    parser.add_argument("--modzcta-path", type=Path, default=DEFAULT_MODZCTA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--baseline-weeks",
        type=int,
        default=DEFAULT_BASELINE_WEEKS,
        help="Number of previous same hour-of-week observations to average.",
    )
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument(
        "--start",
        type=str,
        default=str(DEFAULT_OUTPUT_START),
        help="Output start timestamp.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional output end timestamp. Defaults to min(weather max hour, EMS max hour).",
    )
    parser.add_argument(
        "--exclude-zip-codes",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_ZIP_CODES),
        help="ZIP codes to remove from the Manhattan EMS analysis universe.",
    )
    return parser.parse_args()


def extract_zip_codes(*values: object) -> set[str]:
    zip_codes: set[str] = set()
    for value in values:
        zip_codes.update(re.findall(r"\b\d{5}\b", str(value)))
    return zip_codes


def normalize_zip_codes(zip_codes: object) -> set[str]:
    if zip_codes is None:
        return set()
    if isinstance(zip_codes, str):
        return extract_zip_codes(zip_codes)
    if not isinstance(zip_codes, Iterable):
        return extract_zip_codes(zip_codes)
    normalized: set[str] = set()
    for zip_code in zip_codes:
        normalized.update(extract_zip_codes(zip_code))
    return normalized


def choose_canonical_zip_code(modzcta_value: object, alias_zip_codes: set[str]) -> str:
    modzcta_zip_codes = extract_zip_codes(modzcta_value)
    if len(modzcta_zip_codes) == 1:
        return next(iter(modzcta_zip_codes))
    if not alias_zip_codes:
        raise ValueError(f"Could not resolve a canonical ZIP for MODZCTA {modzcta_value!r}.")
    return sorted(alias_zip_codes)[0]


def parse_hour_series(datetime_series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        datetime_series,
        format=EMS_DATETIME_FORMAT,
        errors="coerce",
    ).dt.floor("h")


def load_weather(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing NYC weather CSV: {path}")
    weather = pd.read_csv(
        path,
        usecols=["timestamp_hour", "temp_c", "precip_mm"],
        parse_dates=["timestamp_hour"],
    )
    return weather.sort_values("timestamp_hour").reset_index(drop=True)


def load_manhattan_boundary(zone_lookup_path: Path) -> gpd.GeoSeries:
    if not zone_lookup_path.exists():
        raise FileNotFoundError(f"Missing taxi zone lookup: {zone_lookup_path}")

    zones = gpd.read_parquet(zone_lookup_path)
    manhattan_zones = zones.loc[zones["borough"].eq("Manhattan"), ["geometry"]].copy()
    if manhattan_zones.empty:
        raise ValueError("No Manhattan geometries found in taxi zone lookup.")
    if manhattan_zones.crs is None:
        manhattan_zones = manhattan_zones.set_crs("EPSG:4326")
    return manhattan_zones.to_crs(NYC_AREA_CRS).union_all()


def build_zip_geography(
    modzcta_path: Path,
    zone_lookup_path: Path,
    *,
    excluded_zip_codes: object = DEFAULT_EXCLUDED_ZIP_CODES,
) -> gpd.GeoDataFrame:
    if not modzcta_path.exists():
        raise FileNotFoundError(
            f"Missing MODZCTA geography: {modzcta_path}. "
            "Download it from https://data.cityofnewyork.us/resource/pri4-ifjk.geojson",
        )

    modzcta = gpd.read_file(modzcta_path)
    missing_columns = {"modzcta", "label", "zcta", "geometry"}.difference(modzcta.columns)
    if missing_columns:
        raise ValueError(f"MODZCTA file is missing columns: {sorted(missing_columns)}")
    if modzcta.crs is None:
        modzcta = modzcta.set_crs("EPSG:4326")

    excluded = normalize_zip_codes(excluded_zip_codes)
    rows: list[dict[str, object]] = []
    for row in modzcta[["modzcta", "label", "zcta", "pop_est", "geometry"]].itertuples(index=False):
        alias_zip_codes = extract_zip_codes(row.modzcta, row.label, row.zcta) - excluded
        if not alias_zip_codes:
            continue
        canonical_zip_code = choose_canonical_zip_code(row.modzcta, alias_zip_codes)
        if canonical_zip_code in excluded:
            continue
        rows.append(
            {
                "zip_code": canonical_zip_code,
                "modzcta": str(row.modzcta),
                "alias_zip_codes": "|".join(sorted(alias_zip_codes)),
                "alias_zip_count": len(alias_zip_codes),
                "pop_est": row.pop_est,
                "geometry": row.geometry,
            },
        )

    if not rows:
        raise ValueError("No MODZCTA ZIP geometries remain after applying exclusions.")

    zip_geo = gpd.GeoDataFrame(rows, crs=modzcta.crs)
    zip_geo = zip_geo.dissolve(
        by="zip_code",
        as_index=False,
        aggfunc={
            "modzcta": "first",
            "alias_zip_codes": "first",
            "alias_zip_count": "first",
            "pop_est": "first",
        },
    )
    zip_geo_projected = zip_geo.to_crs(NYC_AREA_CRS)
    manhattan_boundary = load_manhattan_boundary(zone_lookup_path)
    zip_geo_projected["manhattan_overlap_area_km2"] = (
        zip_geo_projected.intersection(manhattan_boundary).area / 1_000_000
    )
    zip_geo_projected = zip_geo_projected.loc[
        zip_geo_projected["manhattan_overlap_area_km2"].gt(0)
    ].copy()
    if zip_geo_projected.empty:
        raise ValueError("No MODZCTA ZIP geometries overlap Manhattan.")

    zip_geo_projected["zip_area_km2"] = zip_geo_projected.geometry.area / 1_000_000
    centroids = zip_geo_projected.geometry.centroid.to_crs("EPSG:4326")
    zip_geo_projected["centroid_lon"] = centroids.x
    zip_geo_projected["centroid_lat"] = centroids.y
    return zip_geo_projected.to_crs("EPSG:4326").sort_values("zip_code").reset_index(drop=True)


def build_zip_alias_map(zip_geo: pd.DataFrame) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for row in zip_geo[["zip_code", "alias_zip_codes"]].itertuples(index=False):
        canonical_zip_code = str(row.zip_code)
        alias_zip_codes = str(row.alias_zip_codes).split("|")
        for alias_zip_code in alias_zip_codes:
            if alias_zip_code:
                alias_map[alias_zip_code] = canonical_zip_code
    return alias_map


def canonicalize_zip_hourly(
    zip_hourly: pd.DataFrame,
    zip_geo: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alias_map = build_zip_alias_map(zip_geo)
    canonicalized = zip_hourly.copy()
    canonicalized["raw_zip_code"] = canonicalized["zip_code"].astype(str)
    canonicalized["zip_code"] = (
        canonicalized["raw_zip_code"].map(alias_map).fillna(canonicalized["raw_zip_code"])
    )
    alias_rollup = (
        canonicalized.loc[
            canonicalized["raw_zip_code"] != canonicalized["zip_code"],
            ["raw_zip_code", "zip_code", "ems_incident_count"],
        ]
        .groupby(["raw_zip_code", "zip_code"], as_index=False, observed=True)[
            "ems_incident_count"
        ]
        .sum()
        .rename(
            columns={
                "zip_code": "canonical_zip_code",
                "ems_incident_count": "rolled_up_ems_incidents",
            },
        )
        .sort_values("rolled_up_ems_incidents", ascending=False)
        .reset_index(drop=True)
    )
    canonicalized = (
        canonicalized.groupby(["timestamp_hour", "zip_code"], as_index=False, observed=True)[
            "ems_incident_count"
        ]
        .sum()
        .sort_values(["timestamp_hour", "zip_code"])
        .reset_index(drop=True)
    )
    return canonicalized, alias_rollup


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
        pd.read_csv(raw_path, usecols=usecols, dtype=str, chunksize=chunksize),
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
            .rename("ems_incident_count")
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
    zip_hourly = (
        pd.concat(zip_hour_frames, ignore_index=True)
        .groupby(["timestamp_hour", "zip_code"], as_index=False, observed=True)[
            "ems_incident_count"
        ]
        .sum()
        .sort_values(["timestamp_hour", "zip_code"])
        .reset_index(drop=True)
    )
    return zip_hourly, citywide_hourly, stats


def filter_to_geographic_manhattan_zips(
    zip_hourly: pd.DataFrame,
    zip_geo: gpd.GeoDataFrame,
    *,
    output_start: pd.Timestamp,
    output_end: pd.Timestamp,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    valid_zip_codes = set(zip_geo["zip_code"].astype(str))
    output_window = zip_hourly.loc[
        zip_hourly["timestamp_hour"].ge(output_start)
        & zip_hourly["timestamp_hour"].le(output_end)
    ].copy()
    observed_output_zips = set(output_window["zip_code"].astype(str))
    selected_zip_codes = sorted(valid_zip_codes.intersection(observed_output_zips))
    if not selected_zip_codes:
        raise ValueError("No observed EMS ZIPs overlap Manhattan MODZCTA geography.")

    unmapped = output_window.loc[~output_window["zip_code"].isin(valid_zip_codes)].copy()
    unmapped_totals = (
        unmapped.groupby("zip_code", as_index=False, observed=True)["ems_incident_count"]
        .sum()
        .rename(columns={"ems_incident_count": "unmapped_ems_incidents"})
        .sort_values("unmapped_ems_incidents", ascending=False)
        .reset_index(drop=True)
    )

    filtered_hourly = zip_hourly.loc[zip_hourly["zip_code"].isin(selected_zip_codes)].copy()
    filtered_geo = zip_geo.loc[zip_geo["zip_code"].isin(selected_zip_codes)].copy()
    return filtered_hourly, filtered_geo.sort_values("zip_code").reset_index(drop=True), unmapped_totals


def build_dense_panel(
    zip_codes: list[str],
    zip_hourly: pd.DataFrame,
    *,
    history_start: pd.Timestamp,
    output_end: pd.Timestamp,
) -> pd.DataFrame:
    full_hours = pd.date_range(history_start, output_end, freq="h")
    full_index = pd.MultiIndex.from_product(
        [full_hours, zip_codes],
        names=["timestamp_hour", "zip_code"],
    )
    panel = full_index.to_frame(index=False)
    panel = panel.merge(zip_hourly, on=["timestamp_hour", "zip_code"], how="left")
    panel["ems_incident_count"] = panel["ems_incident_count"].fillna(0).astype(int)
    return panel.sort_values(["zip_code", "timestamp_hour"]).reset_index(drop=True)


def build_zip_adjacency(zip_geo: gpd.GeoDataFrame) -> pd.DataFrame:
    projected = zip_geo[["zip_code", "geometry"]].to_crs(NYC_AREA_CRS).copy()
    zip_codes = projected["zip_code"].tolist()
    adjacency = pd.DataFrame(0.0, index=zip_codes, columns=zip_codes)
    spatial_index = projected.sindex

    for left_idx, left in projected.iterrows():
        candidate_indices = list(spatial_index.query(left.geometry, predicate="intersects"))
        for right_idx in candidate_indices:
            if right_idx == left_idx:
                continue
            right = projected.iloc[right_idx]
            if left.geometry.intersects(right.geometry):
                adjacency.loc[str(right["zip_code"]), str(left["zip_code"])] = 1.0
    return adjacency


def add_ems_features(
    panel: pd.DataFrame,
    zip_geo: gpd.GeoDataFrame,
    *,
    baseline_weeks: int,
) -> pd.DataFrame:
    if baseline_weeks <= 0:
        raise ValueError("baseline_weeks must be positive.")

    panel = panel.copy()
    by_zip = panel.groupby("zip_code", sort=False)["ems_incident_count"]
    panel["ems_incidents_lag_1"] = by_zip.shift(1)

    baseline_columns: list[str] = []
    for week_idx in range(1, baseline_weeks + 1):
        column = f"_ems_same_hour_week_lag_{week_idx}"
        panel[column] = by_zip.shift(168 * week_idx)
        baseline_columns.append(column)
    panel["zone_hour_baseline"] = panel[baseline_columns].mean(axis=1)
    panel = panel.drop(columns=baseline_columns)

    zip_codes = sorted(panel["zip_code"].unique().tolist())
    full_hours = pd.Index(panel["timestamp_hour"].drop_duplicates()).sort_values()
    adjacency = build_zip_adjacency(zip_geo)
    adjacency = adjacency.reindex(index=zip_codes, columns=zip_codes, fill_value=0.0)
    degrees = adjacency.sum(axis=0).replace(0.0, 1.0)
    ems_wide = (
        panel.pivot(index="timestamp_hour", columns="zip_code", values="ems_incident_count")
        .reindex(index=full_hours, columns=zip_codes)
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
        "zip_code",
        "neighbor_ems_incidents_lag_1_mean",
    ]

    panel = panel.merge(neighbor_features, on=["timestamp_hour", "zip_code"], how="left")
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
    citywide["citywide_ems_incidents"] = citywide["citywide_ems_incidents"].fillna(0).astype(int)
    citywide["citywide_ems_incidents_lag_1"] = citywide["citywide_ems_incidents"].shift(1)
    return panel.merge(
        citywide[["timestamp_hour", "citywide_ems_incidents_lag_1"]],
        on="timestamp_hour",
        how="left",
    )


def haversine_distance_matrix_km(
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> np.ndarray:
    latitudes_rad = np.radians(latitudes_deg)
    longitudes_rad = np.radians(longitudes_deg)
    delta_lat = latitudes_rad[:, None] - latitudes_rad[None, :]
    delta_lon = longitudes_rad[:, None] - longitudes_rad[None, :]
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(latitudes_rad)[:, None]
        * np.cos(latitudes_rad)[None, :]
        * np.sin(delta_lon / 2.0) ** 2
    )
    central_angle = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    distances_km = EARTH_RADIUS_KM * central_angle
    np.fill_diagonal(distances_km, 0.0)
    return distances_km


def write_zip_distance_matrix(zip_geo: gpd.GeoDataFrame, output_dir: Path) -> tuple[Path, Path]:
    zip_geo = zip_geo.sort_values("zip_code").reset_index(drop=True)
    distances_km = haversine_distance_matrix_km(
        latitudes_deg=zip_geo["centroid_lat"].to_numpy(dtype=float),
        longitudes_deg=zip_geo["centroid_lon"].to_numpy(dtype=float),
    )
    zip_codes = zip_geo["zip_code"].astype(str).tolist()
    distance_matrix = pd.DataFrame(distances_km, index=zip_codes, columns=zip_codes)
    distance_matrix.index.name = "zip_code"

    csv_path = output_dir / "ems_zip_centroid_distance_matrix_km.csv"
    parquet_path = output_dir / "ems_zip_centroid_distance_matrix_km.parquet"
    distance_matrix.reset_index().to_csv(csv_path, index=False)
    distance_matrix.to_parquet(parquet_path, index=True)
    return csv_path, parquet_path


def finalize_panel(
    panel: pd.DataFrame,
    weather: pd.DataFrame,
    zip_geo: gpd.GeoDataFrame,
    *,
    output_start: pd.Timestamp,
    output_end: pd.Timestamp,
) -> pd.DataFrame:
    panel = panel.merge(weather, on="timestamp_hour", how="left")
    static_features = zip_geo[
        [
            "zip_code",
            "modzcta",
            "alias_zip_codes",
            "alias_zip_count",
            "centroid_lon",
            "centroid_lat",
            "zip_area_km2",
            "manhattan_overlap_area_km2",
        ]
    ].copy()
    panel = panel.merge(static_features, on="zip_code", how="left")
    panel = panel.loc[
        panel["timestamp_hour"].ge(output_start) & panel["timestamp_hour"].le(output_end)
    ].copy()

    required_feature_columns = [
        "ems_incidents_lag_1",
        "neighbor_ems_incidents_lag_1_mean",
        "temp_c",
        "precip_mm",
        "zone_hour_baseline",
        "citywide_ems_incidents_lag_1",
        "hour",
        "month",
        "day_of_week",
    ]
    final_panel = panel.loc[panel[required_feature_columns].notna().all(axis=1)].copy()
    final_panel["ems_incident_count"] = final_panel["ems_incident_count"].astype(int)
    final_panel["citywide_ems_incidents_lag_1"] = final_panel[
        "citywide_ems_incidents_lag_1"
    ].astype(int)

    ordered_columns = [
        "timestamp_hour",
        "zip_code",
        "modzcta",
        "alias_zip_codes",
        "alias_zip_count",
        "centroid_lon",
        "centroid_lat",
        "zip_area_km2",
        "manhattan_overlap_area_km2",
        "hour",
        "month",
        "day_of_week",
        "ems_incident_count",
        "ems_incidents_lag_1",
        "neighbor_ems_incidents_lag_1_mean",
        "temp_c",
        "precip_mm",
        "zone_hour_baseline",
        "citywide_ems_incidents_lag_1",
    ]
    return final_panel[ordered_columns].sort_values(
        ["timestamp_hour", "zip_code"],
    ).reset_index(drop=True)


def write_model_columns(final_panel: pd.DataFrame, output_dir: Path) -> Path:
    model_columns = [
        "hour",
        "month",
        "day_of_week",
        "ems_incidents_lag_1",
        "neighbor_ems_incidents_lag_1_mean",
        "temp_c",
        "precip_mm",
        "zone_hour_baseline",
        "citywide_ems_incidents_lag_1",
        "ems_incident_count",
    ]
    model_path = output_dir / "ems_zip_hour_features_2025_manhattan_model_cols.csv"
    final_panel[model_columns].to_csv(model_path, index=False)
    return model_path


def main() -> None:
    args = parse_args()
    output_start = pd.Timestamp(args.start)
    weather = load_weather(args.weather_path)
    weather_max_hour = pd.Timestamp(weather["timestamp_hour"].max())
    requested_end = pd.Timestamp(args.end) if args.end else weather_max_hour
    if requested_end < output_start:
        raise ValueError("Output end must be greater than or equal to output start.")
    history_start = output_start - pd.Timedelta(weeks=args.baseline_weeks)

    print(
        f"Building EMS ZIP-hour panel from {output_start} with "
        f"{args.baseline_weeks} baseline weeks.",
        flush=True,
    )
    excluded_zip_codes = normalize_zip_codes(args.exclude_zip_codes)
    zip_geo = build_zip_geography(
        args.modzcta_path,
        args.zone_lookup_path,
        excluded_zip_codes=excluded_zip_codes,
    )
    zip_hourly, citywide_hourly, scan_stats = aggregate_ems_csv(
        args.raw_path,
        history_start=history_start,
        read_end=requested_end,
        chunksize=args.chunksize,
    )
    zip_hourly, alias_rollup_totals = canonicalize_zip_hourly(zip_hourly, zip_geo)
    if excluded_zip_codes:
        zip_hourly = zip_hourly.loc[
            ~zip_hourly["zip_code"].astype(str).isin(excluded_zip_codes)
        ].copy()
    ems_max_hour = pd.Timestamp(citywide_hourly["timestamp_hour"].max())
    output_end = min(requested_end, weather_max_hour, ems_max_hour)
    zip_hourly, zip_geo, unmapped_zip_totals = filter_to_geographic_manhattan_zips(
        zip_hourly,
        zip_geo,
        output_start=output_start,
        output_end=output_end,
    )
    zip_codes = sorted(zip_geo["zip_code"].astype(str).tolist())

    dense_panel = build_dense_panel(
        zip_codes,
        zip_hourly,
        history_start=history_start,
        output_end=output_end,
    )
    dense_panel = add_ems_features(
        dense_panel,
        zip_geo,
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
        weather,
        zip_geo,
        output_start=output_start,
        output_end=output_end,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "ems_zip_hour_features_2025_manhattan.csv"
    zip_geo_path = output_dir / "ems_zip_geography_manhattan.csv"
    unmapped_path = output_dir / "ems_zip_unmapped_totals.csv"
    alias_rollup_path = output_dir / "ems_zip_alias_rollup_totals.csv"
    summary_path = output_dir / "ems_zip_hour_features_2025_manhattan_summary.json"
    final_panel.to_csv(feature_path, index=False)
    model_path = write_model_columns(final_panel, output_dir)
    distance_csv_path, distance_parquet_path = write_zip_distance_matrix(zip_geo, output_dir)
    zip_geo.drop(columns="geometry").to_csv(zip_geo_path, index=False)
    unmapped_zip_totals.to_csv(unmapped_path, index=False)
    alias_rollup_totals.to_csv(alias_rollup_path, index=False)

    mapped_incidents = int(
        zip_hourly.loc[
            zip_hourly["timestamp_hour"].ge(output_start)
            & zip_hourly["timestamp_hour"].le(output_end),
            "ems_incident_count",
        ].sum(),
    )
    unmapped_incidents = int(unmapped_zip_totals["unmapped_ems_incidents"].sum())
    raw_manhattan_zip_incidents = mapped_incidents + unmapped_incidents
    summary = {
        **scan_stats,
        "baseline_weeks": float(args.baseline_weeks),
        "zip_count": float(len(zip_codes)),
        "canonical_zip_count": float(len(zip_codes)),
        "alias_zip_rollup_count": float(len(alias_rollup_totals)),
        "alias_zip_rolled_up_incidents": float(
            alias_rollup_totals["rolled_up_ems_incidents"].sum()
            if not alias_rollup_totals.empty
            else 0.0
        ),
        "output_hour_count": float(final_panel["timestamp_hour"].nunique()),
        "output_row_count": float(len(final_panel)),
        "output_start": str(final_panel["timestamp_hour"].min()),
        "output_end": str(final_panel["timestamp_hour"].max()),
        "history_start": str(history_start),
        "ems_raw_max_hour_in_window": str(ems_max_hour),
        "weather_min_hour": str(weather["timestamp_hour"].min()),
        "weather_max_hour": str(weather_max_hour),
        "mapped_manhattan_zip_incidents_in_output_window": float(mapped_incidents),
        "unmapped_manhattan_zip_incidents_in_output_window": float(unmapped_incidents),
        "mapped_share_of_manhattan_zip_incidents": (
            mapped_incidents / raw_manhattan_zip_incidents
            if raw_manhattan_zip_incidents
            else np.nan
        ),
        "source_raw_path": str(args.raw_path),
        "source_weather_path": str(args.weather_path),
        "source_modzcta_path": str(args.modzcta_path),
        "source_manhattan_boundary_path": str(args.zone_lookup_path),
    }
    pd.Series(summary).to_json(summary_path, indent=2)

    print(f"Wrote EMS ZIP feature table to {feature_path}")
    print(f"Wrote modeling-only columns to {model_path}")
    print(f"Wrote ZIP centroid distance matrix to {distance_csv_path}")
    print(f"Wrote parquet distance matrix to {distance_parquet_path}")
    print(f"Wrote ZIP geography table to {zip_geo_path}")
    print(f"Wrote unmapped ZIP totals to {unmapped_path}")
    print(f"Wrote alias ZIP rollup totals to {alias_rollup_path}")
    print(f"Wrote validation summary to {summary_path}")
    print(
        f"Final rows: {len(final_panel):,}; "
        f"hours: {final_panel['timestamp_hour'].nunique():,}; "
        f"ZIPs: {final_panel['zip_code'].nunique():,}",
    )


if __name__ == "__main__":
    main()
