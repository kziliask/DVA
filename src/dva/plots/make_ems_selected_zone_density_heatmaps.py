from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cmcrameri  # noqa: F401 - registers cmc.* colormaps with Matplotlib.
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MANIFEST_PATH = Path("results/ems_shap_exhaustive_comparison/manifest.csv")
DEFAULT_GEOJSON_PATH = Path("results/ems_zipcode_polygons/ems_manhattan_zip_polygons.geojson")
DEFAULT_OUTDIR = Path("data/plots/ems_shap_exhaustive_comparison/geospatial")
DEFAULT_SOLUTION_TYPE = "full_model"
DEFAULT_CMAP = "cmc.lajolla"
METHOD_ORDER = ("naive", "greedy", "lp_relaxation", "exact")
METHOD_LABELS = {
    "naive": "Naive",
    "greedy": "Greedy",
    "lp_relaxation": "LP relaxation",
    "exact": "Exact",
}
SOLVER_TO_METHOD = {
    "naive_greedy": "naive",
    "greedy_max_cover": "greedy",
    "gurobi_lp_relaxation": "lp_relaxation",
    "gurobi": "exact",
}
PANEL_EDGE_COLOR = "#363636"
MISSING_COLOR = "#eeeeee"


@dataclass(frozen=True, slots=True)
class HeatmapOutputs:
    by_radius_plot: Path
    by_facility_budget_plot: Path
    density_csv: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create EMS geospatial heatmaps of selected staging-zone density by method."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="EMS exhaustive comparison manifest with results_dir entries.",
    )
    parser.add_argument(
        "--geojson",
        type=Path,
        default=DEFAULT_GEOJSON_PATH,
        help="Manhattan EMS ZIP polygon GeoJSON.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--solution-type",
        default=DEFAULT_SOLUTION_TYPE,
        help="Coverage solution type to summarize from coverage_solutions.csv.",
    )
    parser.add_argument(
        "--cmap",
        default=DEFAULT_CMAP,
        help="Registered Matplotlib colormap name; cmcrameri maps use the cmc.* prefix.",
    )
    parser.add_argument(
        "--include-zip-labels",
        action="store_true",
        help="Annotate each polygon with its ZIP code.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = write_ems_selected_zone_density_heatmaps(
        manifest_path=args.manifest,
        geojson_path=args.geojson,
        outdir=args.outdir,
        solution_type=args.solution_type,
        cmap_name=args.cmap,
        include_zip_labels=args.include_zip_labels,
    )
    print(f"Wrote EMS selected-zone heatmap by radius to {outputs.by_radius_plot}")
    print(
        "Wrote EMS selected-zone heatmap by facility budget to "
        f"{outputs.by_facility_budget_plot}"
    )
    print(f"Wrote EMS selected-zone density table to {outputs.density_csv}")


def write_ems_selected_zone_density_heatmaps(
    *,
    manifest_path: Path,
    geojson_path: Path,
    outdir: Path,
    solution_type: str = DEFAULT_SOLUTION_TYPE,
    cmap_name: str = DEFAULT_CMAP,
    include_zip_labels: bool = False,
) -> HeatmapOutputs:
    outdir.mkdir(parents=True, exist_ok=True)
    geography = _load_geography(geojson_path)
    selections = load_selected_zone_records(manifest_path, solution_type=solution_type)
    if selections.empty:
        raise ValueError(
            f"No selected zone records found for solution_type={solution_type!r}."
        )

    by_radius = compute_selection_density(
        selections,
        group_columns=("method", "coverage_radius_km"),
        zip_codes=tuple(geography["zip_code"].astype(str)),
        grid_type="radius",
    )
    by_facility_budget = compute_selection_density(
        selections,
        group_columns=("method", "facility_budget"),
        zip_codes=tuple(geography["zip_code"].astype(str)),
        grid_type="facility_budget",
    )
    density = pd.concat([by_radius, by_facility_budget], ignore_index=True)
    density_csv = outdir / "ems_selected_zone_density.csv"
    density.to_csv(density_csv, index=False)

    by_radius_plot = outdir / "ems_selected_zone_density_by_radius.png"
    plot_density_grid(
        geography=geography,
        density=by_radius,
        row_column="coverage_radius_km",
        row_label_format=lambda value: f"{float(value):g} km",
        output_path=by_radius_plot,
        title="Selected EMS staging-zone density by coverage radius",
        cmap_name=cmap_name,
        include_zip_labels=include_zip_labels,
    )
    by_facility_budget_plot = outdir / "ems_selected_zone_density_by_facility_budget.png"
    plot_density_grid(
        geography=geography,
        density=by_facility_budget,
        row_column="facility_budget",
        row_label_format=lambda value: f"{int(value)} staging areas",
        output_path=by_facility_budget_plot,
        title="Selected EMS staging-zone density by number of staging areas",
        cmap_name=cmap_name,
        include_zip_labels=include_zip_labels,
    )
    return HeatmapOutputs(
        by_radius_plot=by_radius_plot,
        by_facility_budget_plot=by_facility_budget_plot,
        density_csv=density_csv,
    )


def load_selected_zone_records(
    manifest_path: Path,
    *,
    solution_type: str,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required_manifest_columns = {
        "setting_id",
        "results_dir",
        "coverage_solver",
        "coverage_radius_km",
        "facility_budget",
    }
    missing_manifest_columns = required_manifest_columns - set(manifest.columns)
    if missing_manifest_columns:
        raise KeyError(
            "Manifest is missing required columns: "
            + ", ".join(sorted(missing_manifest_columns))
        )

    rows: list[dict[str, Any]] = []
    for manifest_row in manifest.to_dict(orient="records"):
        setting_id = str(manifest_row["setting_id"])
        results_dir = _resolve_manifest_path(
            manifest_path,
            Path(str(manifest_row["results_dir"])),
        )
        coverage_solutions_path = results_dir / "coverage_solutions.csv"
        if not coverage_solutions_path.exists():
            raise FileNotFoundError(
                f"Missing coverage_solutions.csv for {setting_id}: "
                f"{coverage_solutions_path}"
            )

        coverage_solutions = pd.read_csv(
            coverage_solutions_path,
            dtype={"timestamp_hour": str},
        )
        required_solution_columns = {
            "timestamp_hour",
            "solution_type",
            "selected_facility_zip_codes",
        }
        missing_solution_columns = required_solution_columns - set(coverage_solutions.columns)
        if missing_solution_columns:
            raise KeyError(
                f"{coverage_solutions_path} is missing required columns: "
                + ", ".join(sorted(missing_solution_columns))
            )

        solution_frame = coverage_solutions.loc[
            coverage_solutions["solution_type"].eq(solution_type)
        ].copy()
        if solution_frame.empty:
            continue
        method = _resolve_method(manifest_row)
        for solution_row in solution_frame.to_dict(orient="records"):
            selected_zip_codes = _parse_zip_code_list(
                solution_row["selected_facility_zip_codes"]
            )
            for selected_zip_code in selected_zip_codes:
                rows.append(
                    {
                        "setting_id": setting_id,
                        "method": method,
                        "coverage_solver": str(manifest_row["coverage_solver"]),
                        "coverage_radius_km": float(manifest_row["coverage_radius_km"]),
                        "facility_budget": int(manifest_row["facility_budget"]),
                        "timestamp_hour": str(solution_row["timestamp_hour"]),
                        "zip_code": selected_zip_code,
                    }
                )

    return pd.DataFrame(rows)


def compute_selection_density(
    selected_records: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    zip_codes: Sequence[str],
    grid_type: str,
) -> pd.DataFrame:
    required_columns = {*group_columns, "timestamp_hour", "setting_id", "zip_code"}
    missing_columns = required_columns - set(selected_records.columns)
    if missing_columns:
        raise KeyError(
            "Selected records are missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    opportunities = (
        selected_records.loc[:, [*group_columns, "setting_id", "timestamp_hour"]]
        .drop_duplicates()
        .groupby(list(group_columns), dropna=False)
        .size()
        .rename("opportunity_count")
        .reset_index()
    )
    selection_counts = (
        selected_records.groupby([*group_columns, "zip_code"], dropna=False)
        .size()
        .rename("selection_count")
        .reset_index()
    )

    rows = []
    for opportunity_row in opportunities.to_dict(orient="records"):
        opportunity_count = int(opportunity_row["opportunity_count"])
        group_filter = pd.Series(True, index=selection_counts.index)
        for column in group_columns:
            group_filter &= selection_counts[column].eq(opportunity_row[column])
        group_counts = selection_counts.loc[group_filter].set_index("zip_code")
        for zip_code in zip_codes:
            selection_count = (
                int(group_counts.loc[zip_code, "selection_count"])
                if zip_code in group_counts.index
                else 0
            )
            row = {
                column: opportunity_row[column]
                for column in group_columns
            }
            row.update(
                {
                    "grid_type": grid_type,
                    "zip_code": str(zip_code),
                    "selection_count": selection_count,
                    "opportunity_count": opportunity_count,
                    "selected_rate": (
                        float(selection_count / opportunity_count)
                        if opportunity_count
                        else np.nan
                    ),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def plot_density_grid(
    *,
    geography: gpd.GeoDataFrame,
    density: pd.DataFrame,
    row_column: str,
    row_label_format,
    output_path: Path,
    title: str,
    cmap_name: str,
    include_zip_labels: bool,
) -> None:
    methods = [method for method in METHOD_ORDER if method in set(density["method"])]
    extra_methods = sorted(set(density["method"]) - set(methods))
    methods.extend(extra_methods)
    row_values = sorted(density[row_column].dropna().unique())
    if not methods or not row_values:
        raise ValueError("Need at least one method and one row value to plot density grid.")

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(MISSING_COLOR)
    norm = Normalize(vmin=0.0, vmax=float(density["selected_rate"].max()))

    plot_geography = geography.to_crs("EPSG:4326").copy()
    label_points = (
        geography.geometry.representative_point().to_crs("EPSG:4326")
        if include_zip_labels
        else None
    )

    fig, axes = plt.subplots(
        len(row_values),
        len(methods),
        figsize=(4.2 * len(methods), 4.8 * len(row_values)),
        squeeze=False,
    )
    for row_idx, row_value in enumerate(row_values):
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            panel_density = density.loc[
                density["method"].eq(method) & density[row_column].eq(row_value),
                ["zip_code", "selected_rate", "selection_count", "opportunity_count"],
            ]
            panel_geo = plot_geography.merge(panel_density, on="zip_code", how="left")
            panel_geo.plot(
                ax=ax,
                column="selected_rate",
                cmap=cmap,
                norm=norm,
                edgecolor=PANEL_EDGE_COLOR,
                linewidth=0.45,
                missing_kwds={"color": MISSING_COLOR},
            )
            panel_geo.boundary.plot(ax=ax, color="white", linewidth=0.18, alpha=0.9)
            if include_zip_labels and label_points is not None:
                _add_zip_labels(ax, plot_geography, label_points)
            if row_idx == 0:
                ax.set_title(METHOD_LABELS.get(method, method), fontsize=12, pad=8)
            if col_idx == 0:
                ax.text(
                    -0.08,
                    0.5,
                    row_label_format(row_value),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=11,
                    fontweight="bold",
                )
            ax.set_axis_off()
            ax.set_aspect("equal")

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=axes.ravel().tolist(),
        fraction=0.026,
        pad=0.018,
    )
    colorbar.set_label("Selection frequency across explained hours/settings")
    fig.suptitle(title, fontsize=15, y=0.995)
    fig.savefig(output_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def _load_geography(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing EMS ZIP GeoJSON: {path}")
    geography = gpd.read_file(path)
    if "zip_code" not in geography.columns:
        raise KeyError("GeoJSON must include a zip_code column.")
    geography = geography.copy()
    geography["zip_code"] = geography["zip_code"].astype(str)
    return geography.sort_values("zip_code").reset_index(drop=True)


def _resolve_manifest_path(manifest_path: Path, path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return path


def _resolve_method(manifest_row: dict[str, Any]) -> str:
    if "coverage_solver_label" in manifest_row and pd.notna(
        manifest_row["coverage_solver_label"]
    ):
        return str(manifest_row["coverage_solver_label"])
    solver = str(manifest_row["coverage_solver"])
    return SOLVER_TO_METHOD.get(solver, solver)


def _parse_zip_code_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(zip_code) for zip_code in value)
    if pd.isna(value):
        return ()
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list of ZIP codes, got {value!r}.")
    return tuple(str(zip_code) for zip_code in parsed)


def _add_zip_labels(
    ax: plt.Axes,
    geography: gpd.GeoDataFrame,
    label_points: gpd.GeoSeries,
) -> None:
    for zip_code, point in zip(geography["zip_code"].astype(str), label_points):
        ax.text(
            point.x,
            point.y,
            zip_code,
            ha="center",
            va="center",
            fontsize=4.3,
            color="#171717",
            alpha=0.78,
        )


if __name__ == "__main__":
    main()
