from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cmcrameri  # noqa: F401 - registers cmc.* colormaps with Matplotlib.
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from dva.plots.make_ems_selected_zone_density_heatmaps import (
    DEFAULT_CMAP,
    DEFAULT_GEOJSON_PATH,
    MISSING_COLOR,
    PANEL_EDGE_COLOR,
    _add_zip_labels,
    _load_geography,
    _parse_zip_code_list,
    _resolve_manifest_path,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MANIFEST_PATH = Path("results/ems_cvar_alpha_experiment/run_manifest.csv")
DEFAULT_OUTDIR = Path("data/plots/ems_cvar_alpha_experiment/geospatial")
DEFAULT_NORMAL_SOLUTION_TYPE = "full_model"
DEFAULT_CVAR_SOLUTION_TYPE = "cvar_full_model"
DEFAULT_COLORBAR_MAX = 1.0


@dataclass(frozen=True, slots=True)
class CvarAlphaHeatmapOutputs:
    density_csv: Path
    plot_paths: tuple[Path, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create one EMS selected-zone density heatmap for the normal run and "
            "each CVaR alpha run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="EMS CVaR alpha experiment manifest with outdir entries.",
    )
    parser.add_argument(
        "--geojson",
        type=Path,
        default=DEFAULT_GEOJSON_PATH,
        help="Manhattan EMS ZIP polygon GeoJSON.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--normal-solution-type",
        default=DEFAULT_NORMAL_SOLUTION_TYPE,
        help="Coverage solution type to summarize for the normal run.",
    )
    parser.add_argument(
        "--cvar-solution-type",
        default=DEFAULT_CVAR_SOLUTION_TYPE,
        help="Coverage solution type to summarize for CVaR alpha runs.",
    )
    parser.add_argument(
        "--cmap",
        default=DEFAULT_CMAP,
        help="Registered Matplotlib colormap name; cmcrameri maps use the cmc.* prefix.",
    )
    parser.add_argument(
        "--colorbar-max",
        type=float,
        default=DEFAULT_COLORBAR_MAX,
        help="Upper bound for the shared selection-frequency color scale.",
    )
    parser.add_argument(
        "--include-zip-labels",
        action="store_true",
        help="Annotate each polygon with its ZIP code.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = write_ems_cvar_alpha_density_heatmaps(
        manifest_path=args.manifest,
        geojson_path=args.geojson,
        outdir=args.outdir,
        normal_solution_type=args.normal_solution_type,
        cvar_solution_type=args.cvar_solution_type,
        cmap_name=args.cmap,
        colorbar_max=args.colorbar_max,
        include_zip_labels=args.include_zip_labels,
    )
    print(f"Wrote EMS CVaR alpha density table to {outputs.density_csv}")
    for plot_path in outputs.plot_paths:
        print(f"Wrote EMS CVaR alpha density heatmap to {plot_path}")


def write_ems_cvar_alpha_density_heatmaps(
    *,
    manifest_path: Path,
    geojson_path: Path,
    outdir: Path,
    normal_solution_type: str = DEFAULT_NORMAL_SOLUTION_TYPE,
    cvar_solution_type: str = DEFAULT_CVAR_SOLUTION_TYPE,
    cmap_name: str = DEFAULT_CMAP,
    colorbar_max: float = DEFAULT_COLORBAR_MAX,
    include_zip_labels: bool = False,
) -> CvarAlphaHeatmapOutputs:
    if colorbar_max <= 0.0:
        raise ValueError("colorbar_max must be positive.")

    outdir.mkdir(parents=True, exist_ok=True)
    geography = _load_geography(geojson_path)
    run_specs = load_cvar_alpha_run_specs(
        manifest_path,
        normal_solution_type=normal_solution_type,
        cvar_solution_type=cvar_solution_type,
    )
    selected_records = load_cvar_alpha_selected_zone_records(
        manifest_path,
        run_specs=run_specs,
    )
    density = compute_cvar_alpha_selection_density(
        selected_records,
        run_specs=run_specs,
        zip_codes=tuple(geography["zip_code"].astype(str)),
    )

    density_csv = outdir / "ems_cvar_alpha_selected_zone_density.csv"
    density.to_csv(density_csv, index=False)

    plot_paths: list[Path] = []
    for spec in run_specs:
        plot_density_map(
            geography=geography,
            density=density.loc[density["run_label"].eq(spec["run_label"])],
            title=str(spec["plot_title"]),
            output_path=outdir / f"ems_selected_zone_density_{spec['plot_slug']}.png",
            cmap_name=cmap_name,
            colorbar_max=colorbar_max,
            include_zip_labels=include_zip_labels,
        )
        plot_paths.append(outdir / f"ems_selected_zone_density_{spec['plot_slug']}.png")

    return CvarAlphaHeatmapOutputs(
        density_csv=density_csv,
        plot_paths=tuple(plot_paths),
    )


def load_cvar_alpha_run_specs(
    manifest_path: Path,
    *,
    normal_solution_type: str,
    cvar_solution_type: str,
) -> list[dict[str, Any]]:
    manifest = pd.read_csv(manifest_path)
    required_columns = {"run_label", "run_kind", "outdir", "cvar_alpha"}
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise KeyError(
            "Manifest is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    specs: list[dict[str, Any]] = []
    normal_rows = manifest.loc[manifest["run_kind"].eq("normal")].copy()
    if normal_rows.empty:
        raise ValueError("Manifest does not contain a normal run.")
    if len(normal_rows) > 1:
        raise ValueError("Expected exactly one normal run in the manifest.")

    normal_row = normal_rows.iloc[0]
    specs.append(
        _build_run_spec(
            manifest_path,
            normal_row,
            solution_type=normal_solution_type,
            plot_title="Normal selected-zone density",
            plot_slug="normal",
            sort_key=-1.0,
        )
    )

    cvar_rows = manifest.loc[manifest["run_kind"].eq("cvar")].copy()
    if cvar_rows.empty:
        raise ValueError("Manifest does not contain any CVaR alpha runs.")
    cvar_rows = cvar_rows.sort_values("cvar_alpha", na_position="last")
    for _, cvar_row in cvar_rows.iterrows():
        alpha = float(cvar_row["cvar_alpha"])
        alpha_label = f"{alpha:.2f}"
        alpha_slug = alpha_label.replace(".", "p")
        specs.append(
            _build_run_spec(
                manifest_path,
                cvar_row,
                solution_type=cvar_solution_type,
                plot_title=f"CVaR alpha {alpha_label} selected-zone density",
                plot_slug=f"cvar_alpha_{alpha_slug}",
                sort_key=alpha,
            )
        )

    return specs


def load_cvar_alpha_selected_zone_records(
    manifest_path: Path,
    *,
    run_specs: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in run_specs:
        coverage_solutions_path = Path(spec["results_dir"]) / "coverage_solutions.csv"
        if not coverage_solutions_path.exists():
            raise FileNotFoundError(
                f"Missing coverage_solutions.csv for {spec['run_label']}: "
                f"{coverage_solutions_path}"
            )

        coverage_solutions = pd.read_csv(
            coverage_solutions_path,
            dtype={"timestamp_hour": str},
        )
        required_columns = {
            "timestamp_hour",
            "solution_type",
            "selected_facility_zip_codes",
        }
        missing_columns = required_columns - set(coverage_solutions.columns)
        if missing_columns:
            raise KeyError(
                f"{coverage_solutions_path} is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        solution_frame = coverage_solutions.loc[
            coverage_solutions["solution_type"].eq(spec["solution_type"])
        ].copy()
        if solution_frame.empty:
            raise ValueError(
                f"No {spec['solution_type']!r} rows found for {spec['run_label']} "
                f"in {coverage_solutions_path}."
            )

        for solution_row in solution_frame.to_dict(orient="records"):
            selected_zip_codes = _parse_zip_code_list(
                solution_row["selected_facility_zip_codes"]
            )
            for selected_zip_code in selected_zip_codes:
                rows.append(
                    {
                        "run_label": spec["run_label"],
                        "run_kind": spec["run_kind"],
                        "cvar_alpha": spec["cvar_alpha"],
                        "solution_type": spec["solution_type"],
                        "timestamp_hour": str(solution_row["timestamp_hour"]),
                        "zip_code": selected_zip_code,
                    }
                )

    if not rows:
        raise ValueError(f"No selected zone records found from {manifest_path}.")
    return pd.DataFrame(rows)


def compute_cvar_alpha_selection_density(
    selected_records: pd.DataFrame,
    *,
    run_specs: list[dict[str, Any]],
    zip_codes: tuple[str, ...],
) -> pd.DataFrame:
    required_columns = {"run_label", "timestamp_hour", "zip_code"}
    missing_columns = required_columns - set(selected_records.columns)
    if missing_columns:
        raise KeyError(
            "Selected records are missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    opportunities = (
        selected_records.loc[:, ["run_label", "timestamp_hour"]]
        .drop_duplicates()
        .groupby("run_label", dropna=False)
        .size()
        .rename("opportunity_count")
        .to_dict()
    )
    selection_counts = (
        selected_records.groupby(["run_label", "zip_code"], dropna=False)
        .size()
        .rename("selection_count")
        .reset_index()
    )

    rows: list[dict[str, Any]] = []
    for spec in run_specs:
        run_label = str(spec["run_label"])
        opportunity_count = int(opportunities.get(run_label, 0))
        run_counts = selection_counts.loc[
            selection_counts["run_label"].eq(run_label)
        ].set_index("zip_code")
        for zip_code in zip_codes:
            selection_count = (
                int(run_counts.loc[zip_code, "selection_count"])
                if zip_code in run_counts.index
                else 0
            )
            rows.append(
                {
                    "run_label": run_label,
                    "run_kind": spec["run_kind"],
                    "cvar_alpha": spec["cvar_alpha"],
                    "solution_type": spec["solution_type"],
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

    return pd.DataFrame(rows)


def plot_density_map(
    *,
    geography: gpd.GeoDataFrame,
    density: pd.DataFrame,
    title: str,
    output_path: Path,
    cmap_name: str,
    colorbar_max: float,
    include_zip_labels: bool,
) -> None:
    if density.empty:
        raise ValueError(f"No density rows available for {title!r}.")

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(MISSING_COLOR)
    norm = Normalize(vmin=0.0, vmax=colorbar_max)

    plot_geography = geography.to_crs("EPSG:4326").copy()
    label_points = (
        geography.geometry.representative_point().to_crs("EPSG:4326")
        if include_zip_labels
        else None
    )
    panel_geo = plot_geography.merge(
        density.loc[
            :,
            ["zip_code", "selected_rate", "selection_count", "opportunity_count"],
        ],
        on="zip_code",
        how="left",
    )

    fig, ax = plt.subplots(figsize=(6.4, 8.6))
    panel_geo.plot(
        ax=ax,
        column="selected_rate",
        cmap=cmap,
        norm=norm,
        edgecolor=PANEL_EDGE_COLOR,
        linewidth=0.5,
        missing_kwds={"color": MISSING_COLOR},
    )
    panel_geo.boundary.plot(ax=ax, color="white", linewidth=0.2, alpha=0.9)
    if include_zip_labels and label_points is not None:
        _add_zip_labels(ax, plot_geography, label_points)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_axis_off()
    ax.set_aspect("equal")

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.044, pad=0.025)
    colorbar.set_label("Selection frequency across explained hours")
    fig.savefig(output_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def _build_run_spec(
    manifest_path: Path,
    manifest_row: pd.Series,
    *,
    solution_type: str,
    plot_title: str,
    plot_slug: str,
    sort_key: float,
) -> dict[str, Any]:
    return {
        "run_label": str(manifest_row["run_label"]),
        "run_kind": str(manifest_row["run_kind"]),
        "cvar_alpha": (
            float(manifest_row["cvar_alpha"])
            if pd.notna(manifest_row["cvar_alpha"])
            else np.nan
        ),
        "solution_type": solution_type,
        "results_dir": _resolve_manifest_path(
            manifest_path,
            Path(str(manifest_row["outdir"])),
        ),
        "plot_title": plot_title,
        "plot_slug": plot_slug,
        "sort_key": sort_key,
    }


if __name__ == "__main__":
    main()
