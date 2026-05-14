from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


DEFAULT_LONG_FEATURES_PATH = Path("data/ems_data/processed/ems_zip_hour_features_2025_manhattan.csv")
DEFAULT_OUTPUT_DIR = Path("data/ems_data/processed")
DEFAULT_EXCLUDED_ZIP_CODES = ("10468",)
SHARED_FEATURE_COLUMNS = (
    "hour",
    "month",
    "day_of_week",
    "temp_c",
    "precip_mm",
    "citywide_ems_incidents_lag_1",
)
ZONE_FEATURE_COLUMNS = (
    "ems_incidents_lag_1",
    "neighbor_ems_incidents_lag_1_mean",
    "zone_hour_baseline",
)
TARGET_COLUMN = "ems_incident_count"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build wide hour-level EMS ZIP tables for multi-output regression. "
            "Each row is one hour and each output column is one ZIP."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--long-features-path", type=Path, default=DEFAULT_LONG_FEATURES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--exclude-zip-codes",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_ZIP_CODES),
        help="ZIP codes to remove from the wide EMS model tables.",
    )
    return parser.parse_args()


def normalize_zip_codes(zip_codes: object) -> set[str]:
    if zip_codes is None:
        return set()
    if isinstance(zip_codes, str):
        return {zip_codes}
    if not isinstance(zip_codes, Iterable):
        return {str(zip_codes)}
    normalized: set[str] = set()
    for zip_code in zip_codes:
        normalized.add(str(zip_code))
    return normalized


def load_long_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing EMS ZIP long feature table: {path}")

    frame = pd.read_csv(path, dtype={"zip_code": str, "modzcta": str})
    frame["timestamp_hour"] = pd.to_datetime(frame["timestamp_hour"])
    frame["zip_code"] = frame["zip_code"].astype(str)
    frame = frame.sort_values(["timestamp_hour", "zip_code"]).reset_index(drop=True)
    return frame


def filter_excluded_zip_codes(
    frame: pd.DataFrame,
    excluded_zip_codes: object = DEFAULT_EXCLUDED_ZIP_CODES,
) -> pd.DataFrame:
    excluded = normalize_zip_codes(excluded_zip_codes)
    if not excluded:
        return frame
    return frame.loc[~frame["zip_code"].isin(excluded)].reset_index(drop=True)


def validate_long_features(frame: pd.DataFrame) -> tuple[list[str], pd.Index]:
    required_columns = {
        "timestamp_hour",
        "zip_code",
        *SHARED_FEATURE_COLUMNS,
        *ZONE_FEATURE_COLUMNS,
        TARGET_COLUMN,
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Long feature table is missing columns: {sorted(missing_columns)}")

    duplicate_count = int(frame.duplicated(["timestamp_hour", "zip_code"]).sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate timestamp/ZIP rows.")

    zip_codes = sorted(frame["zip_code"].unique().tolist())
    timestamps = pd.Index(frame["timestamp_hour"].drop_duplicates()).sort_values()
    expected_rows = len(zip_codes) * len(timestamps)
    if len(frame) != expected_rows:
        raise ValueError(
            f"Long feature table is not dense: found {len(frame):,} rows, "
            f"expected {expected_rows:,} rows for {len(zip_codes)} ZIPs and "
            f"{len(timestamps)} hours.",
        )

    shared_nunique = frame.groupby("timestamp_hour", observed=True)[
        list(SHARED_FEATURE_COLUMNS)
    ].nunique(dropna=False)
    inconsistent_shared = shared_nunique.gt(1).any(axis=1)
    if inconsistent_shared.any():
        first_bad_hour = shared_nunique.index[inconsistent_shared][0]
        raise ValueError(f"Shared features vary across ZIPs at {first_bad_hour}.")

    feature_columns = [*SHARED_FEATURE_COLUMNS, *ZONE_FEATURE_COLUMNS]
    null_columns = frame[[*feature_columns, TARGET_COLUMN]].isna().sum()
    null_columns = null_columns[null_columns.gt(0)]
    if not null_columns.empty:
        raise ValueError(f"Long feature table has nulls: {null_columns.to_dict()}")

    return zip_codes, timestamps


def build_shared_features(frame: pd.DataFrame, timestamps: pd.Index) -> pd.DataFrame:
    shared = (
        frame.drop_duplicates("timestamp_hour")
        .loc[:, ["timestamp_hour", *SHARED_FEATURE_COLUMNS]]
        .sort_values("timestamp_hour")
        .reset_index(drop=True)
    )
    shared = shared.set_index("timestamp_hour").reindex(timestamps).reset_index()
    return shared


def pivot_zone_column(
    frame: pd.DataFrame,
    *,
    value_column: str,
    output_prefix: str,
    zip_codes: list[str],
    timestamps: pd.Index,
) -> pd.DataFrame:
    wide = frame.pivot(index="timestamp_hour", columns="zip_code", values=value_column)
    wide = wide.reindex(index=timestamps, columns=zip_codes)
    wide.columns = [f"{output_prefix}_zip_{zip_code}" for zip_code in wide.columns]
    return wide.reset_index(drop=True)


def build_wide_tables(
    frame: pd.DataFrame,
    *,
    zip_codes: list[str],
    timestamps: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    shared = build_shared_features(frame, timestamps)

    wide_feature_parts = [shared]
    feature_columns = list(SHARED_FEATURE_COLUMNS)
    for value_column in ZONE_FEATURE_COLUMNS:
        wide_part = pivot_zone_column(
            frame,
            value_column=value_column,
            output_prefix=value_column,
            zip_codes=zip_codes,
            timestamps=timestamps,
        )
        wide_feature_parts.append(wide_part)
        feature_columns.extend(wide_part.columns.tolist())

    x_wide = pd.concat(wide_feature_parts, axis=1)
    x_wide = x_wide.loc[:, ~x_wide.columns.duplicated()]

    y_wide = pivot_zone_column(
        frame,
        value_column=TARGET_COLUMN,
        output_prefix=f"target_{TARGET_COLUMN}",
        zip_codes=zip_codes,
        timestamps=timestamps,
    )
    y_wide.insert(0, "timestamp_hour", shared["timestamp_hour"])
    target_columns = [column for column in y_wide.columns if column != "timestamp_hour"]
    for column in target_columns:
        y_wide[column] = y_wide[column].astype(int)

    combined = x_wide.merge(y_wide, on="timestamp_hour", how="inner")
    columns = {
        "feature_columns": feature_columns,
        "target_columns": target_columns,
    }
    return combined, x_wide, y_wide, columns


def write_zone_order(
    frame: pd.DataFrame,
    *,
    zip_codes: list[str],
    target_columns: list[str],
    output_dir: Path,
) -> Path:
    static_columns = [
        "zip_code",
        "modzcta",
        "alias_zip_codes",
        "alias_zip_count",
        "centroid_lon",
        "centroid_lat",
        "zip_area_km2",
        "manhattan_overlap_area_km2",
    ]
    available_static_columns = [column for column in static_columns if column in frame.columns]
    static = (
        frame.drop_duplicates("zip_code")
        .set_index("zip_code")
        .reindex(zip_codes)
        .reset_index()[available_static_columns]
    )
    static["target_column"] = target_columns
    static["output_index"] = range(len(static))

    zone_order_path = output_dir / "ems_zip_wide_zone_order.csv"
    static.to_csv(zone_order_path, index=False)
    return zone_order_path


def main() -> None:
    args = parse_args()
    frame = load_long_features(args.long_features_path)
    excluded_zip_codes = normalize_zip_codes(args.exclude_zip_codes)
    frame = filter_excluded_zip_codes(frame, excluded_zip_codes)
    zip_codes, timestamps = validate_long_features(frame)
    combined, x_wide, y_wide, columns = build_wide_tables(
        frame,
        zip_codes=zip_codes,
        timestamps=timestamps,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "ems_zip_hour_features_2025_manhattan_wide.csv"
    x_path = output_dir / "ems_zip_hour_features_2025_manhattan_wide_X.csv"
    y_path = output_dir / "ems_zip_hour_features_2025_manhattan_wide_y.csv"
    metadata_path = output_dir / "ems_zip_hour_features_2025_manhattan_wide_metadata.json"

    combined.to_csv(combined_path, index=False)
    x_wide.to_csv(x_path, index=False)
    y_wide.to_csv(y_path, index=False)
    zone_order_path = write_zone_order(
        frame,
        zip_codes=zip_codes,
        target_columns=columns["target_columns"],
        output_dir=output_dir,
    )

    metadata = {
        "source_long_features_path": str(args.long_features_path),
        "wide_combined_path": str(combined_path),
        "wide_x_path": str(x_path),
        "wide_y_path": str(y_path),
        "zone_order_path": str(zone_order_path),
        "row_count": int(len(combined)),
        "zip_count": int(len(zip_codes)),
        "feature_count_excluding_timestamp": int(len(columns["feature_columns"])),
        "target_count": int(len(columns["target_columns"])),
        "timestamp_min": str(combined["timestamp_hour"].min()),
        "timestamp_max": str(combined["timestamp_hour"].max()),
        "zip_codes": zip_codes,
        "shared_feature_columns": list(SHARED_FEATURE_COLUMNS),
        "zone_feature_columns": list(ZONE_FEATURE_COLUMNS),
        "feature_columns": columns["feature_columns"],
        "target_columns": columns["target_columns"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote combined wide table to {combined_path}")
    print(f"Wrote wide X table to {x_path}")
    print(f"Wrote wide y table to {y_path}")
    print(f"Wrote ZIP output order to {zone_order_path}")
    print(f"Wrote metadata to {metadata_path}")
    print(
        f"Rows: {len(combined):,}; ZIP outputs: {len(zip_codes):,}; "
        f"features excluding timestamp: {len(columns['feature_columns']):,}",
    )


if __name__ == "__main__":
    main()
