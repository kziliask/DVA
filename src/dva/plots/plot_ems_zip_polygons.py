from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import pandas as pd

from dva.data_scripts.ems.build_ems_zip_hour_features import (
    DEFAULT_MODZCTA_PATH,
    DEFAULT_ZONE_LOOKUP_PATH,
    NYC_AREA_CRS,
    build_zip_geography,
    load_manhattan_boundary,
)


matplotlib.use("Agg")

DEFAULT_PROCESSED_GEOGRAPHY_PATH = Path(
    "data/ems_data/processed/ems_zip_geography_manhattan.csv"
)
DEFAULT_OUTDIR = Path("results/ems_zipcode_polygons")
DEFAULT_MAP_FILENAME = "ems_manhattan_zip_polygons.png"
DEFAULT_GEOJSON_FILENAME = "ems_manhattan_zip_polygons.geojson"

# Categorical palette only: colors do not encode demand, area, or distance.
ADJACENCY_PALETTE = (
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#b07aa1",
    "#76b7b2",
    "#edc948",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
    "#8cd17d",
    "#b6992d",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot EMS Manhattan ZIP polygons with adjacency-aware categorical colors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--modzcta-path", type=Path, default=DEFAULT_MODZCTA_PATH)
    parser.add_argument("--zone-lookup-path", type=Path, default=DEFAULT_ZONE_LOOKUP_PATH)
    parser.add_argument(
        "--processed-geography-path",
        type=Path,
        default=DEFAULT_PROCESSED_GEOGRAPHY_PATH,
        help="Processed EMS ZIP geography CSV; controls the model ZIP universe.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser


def load_clipped_model_zip_geography(
    *,
    modzcta_path: Path,
    zone_lookup_path: Path,
    processed_geography_path: Path,
) -> gpd.GeoDataFrame:
    processed_geography = pd.read_csv(processed_geography_path, dtype={"zip_code": str})
    model_zip_codes = set(processed_geography["zip_code"].astype(str))

    zip_geo = build_zip_geography(modzcta_path, zone_lookup_path)
    zip_geo = zip_geo.loc[zip_geo["zip_code"].astype(str).isin(model_zip_codes)].copy()
    if zip_geo.empty:
        raise ValueError("No ZIP geometries match the processed EMS geography universe.")

    manhattan_boundary = load_manhattan_boundary(zone_lookup_path)
    clipped = zip_geo.to_crs(NYC_AREA_CRS).copy()
    clipped["geometry"] = clipped.geometry.intersection(manhattan_boundary)
    clipped["clipped_area_km2"] = clipped.geometry.area / 1_000_000
    clipped = clipped.loc[~clipped.geometry.is_empty].copy()
    return clipped.sort_values("zip_code").reset_index(drop=True)


def build_adjacency(geography: gpd.GeoDataFrame) -> dict[str, set[str]]:
    zip_codes = geography["zip_code"].astype(str).tolist()
    geometries = list(geography.geometry)
    adjacency = {zip_code: set() for zip_code in zip_codes}

    for left_idx, left_zip in enumerate(zip_codes):
        left_geometry = geometries[left_idx]
        for right_idx in range(left_idx + 1, len(zip_codes)):
            right_zip = zip_codes[right_idx]
            right_geometry = geometries[right_idx]
            if left_geometry.touches(right_geometry) or left_geometry.intersects(right_geometry):
                adjacency[left_zip].add(right_zip)
                adjacency[right_zip].add(left_zip)

    return adjacency


def color_adjacency_graph(adjacency: dict[str, set[str]]) -> dict[str, str]:
    color_indices: dict[str, int] = {}
    ordered_zip_codes = sorted(
        adjacency,
        key=lambda zip_code: (-len(adjacency[zip_code]), zip_code),
    )
    for zip_code in ordered_zip_codes:
        used_color_indices = {
            color_indices[neighbor]
            for neighbor in adjacency[zip_code]
            if neighbor in color_indices
        }
        for color_idx, _ in enumerate(ADJACENCY_PALETTE):
            if color_idx not in used_color_indices:
                color_indices[zip_code] = color_idx
                break
        else:
            raise ValueError(
                "Adjacency palette is too small for the EMS ZIP neighborhood graph."
            )

    return {
        zip_code: ADJACENCY_PALETTE[color_indices[zip_code]]
        for zip_code in sorted(adjacency)
    }


def validate_coloring(
    adjacency: dict[str, set[str]],
    zip_colors: dict[str, str],
) -> None:
    conflicts = [
        (zip_code, neighbor)
        for zip_code, neighbors in adjacency.items()
        for neighbor in neighbors
        if zip_code < neighbor and zip_colors[zip_code] == zip_colors[neighbor]
    ]
    if conflicts:
        conflict_preview = ", ".join(f"{left}/{right}" for left, right in conflicts[:5])
        raise ValueError(f"Adjacent ZIP color conflict(s): {conflict_preview}")


def write_zip_polygon_plot(
    geography: gpd.GeoDataFrame,
    *,
    outdir: Path,
) -> tuple[Path, Path]:
    adjacency = build_adjacency(geography)
    zip_colors = color_adjacency_graph(adjacency)
    validate_coloring(adjacency, zip_colors)

    plot_geo = geography.copy()
    plot_geo["map_color"] = plot_geo["zip_code"].astype(str).map(zip_colors)
    plot_geo_wgs84 = plot_geo.to_crs("EPSG:4326")

    outdir.mkdir(parents=True, exist_ok=True)
    geojson_path = outdir / DEFAULT_GEOJSON_FILENAME
    png_path = outdir / DEFAULT_MAP_FILENAME
    plot_geo_wgs84.to_file(geojson_path, driver="GeoJSON")

    fig, ax = plt.subplots(figsize=(9, 12))
    plot_geo_wgs84.plot(
        ax=ax,
        color=plot_geo_wgs84["map_color"],
        edgecolor="#1f1f1f",
        linewidth=0.75,
        alpha=0.88,
    )
    plot_geo_wgs84.boundary.plot(ax=ax, color="white", linewidth=0.3, alpha=0.95)

    label_points = plot_geo.geometry.representative_point().to_crs("EPSG:4326")
    for zip_code, point in zip(plot_geo["zip_code"].astype(str), label_points):
        text = ax.text(
            point.x,
            point.y,
            zip_code,
            ha="center",
            va="center",
            fontsize=6.5,
            color="#111111",
        )
        text.set_path_effects(
            [path_effects.withStroke(linewidth=2.0, foreground="white")]
        )

    ax.set_title(
        f"EMS Manhattan ZIP polygons ({len(plot_geo_wgs84)} ZIPs, adjacency-colored)"
    )
    ax.set_axis_off()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return png_path, geojson_path


def main() -> None:
    args = build_parser().parse_args()
    geography = load_clipped_model_zip_geography(
        modzcta_path=args.modzcta_path,
        zone_lookup_path=args.zone_lookup_path,
        processed_geography_path=args.processed_geography_path,
    )
    png_path, geojson_path = write_zip_polygon_plot(geography, outdir=args.outdir)
    print(f"Wrote adjacency-colored ZIP map to {png_path}")
    print(f"Wrote map GeoJSON to {geojson_path}")


if __name__ == "__main__":
    main()
