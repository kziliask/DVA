from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cmcrameri  # noqa: F401 - registers cmc.* colormaps with Matplotlib.
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import Normalize


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RESULT_ROOT = Path("results/ems/experiment_a_infodva")
DEFAULT_OUTDIR = Path("data/plots/ems_infodva_design_summary")
DEFAULT_CMAP = "YlGnBu"
DEFAULT_STACKED_BAR_RADIUS_KM = 1.0
DEFAULT_COVERAGE_RADII_KM = (1.0, 2.0, 3.0)
DEFAULT_FACILITY_BUDGETS = (3, 5, 8)
HEATMAP_COLORBAR_LABEL = "Mean total absolute InfoDVA"
BAR_Y_AXIS_LABEL = "Mean total absolute InfoDVA"

FEATURES = (
    ("zone_hour_baseline", "Zone-hour baseline"),
    ("hour", "Hour"),
    ("neighbor_ems_incidents_lag_1_mean", "Neighbor lag-1 mean"),
    ("citywide_ems_incidents_lag_1", "Citywide lag-1 EMS"),
    ("ems_incidents_lag_1", "ZIP lag-1 EMS"),
    ("temp_c", "Temperature"),
    ("day_of_week", "Day of week"),
    ("precip_mm", "Precipitation"),
)
FEATURE_LABELS = {feature: label for feature, label in FEATURES}
FEATURE_COLORS = {
    "zone_hour_baseline": "#2c62a8",
    "hour": "#43b7c4",
    "neighbor_ems_incidents_lag_1_mean": "#7fcab7",
    "citywide_ems_incidents_lag_1": "#c7e8b4",
    "ems_incidents_lag_1": "#edf8a6",
    "temp_c": "#a9bdd9",
    "day_of_week": "#bdbdbd",
    "precip_mm": "#fee08b",
}
MODE_COLUMN_NAMES = {
    "pre": "ante_decision_mean_abs_shap",
    "post": "decision_mean_abs_shap",
}
MODE_LABELS = {
    "pre": "Pre-InfoDVA",
    "post": "Post-InfoDVA",
}
FONT_CANDIDATES = (
    "Latin Computer Roman",
    "Latin Modern Roman",
    "Computer Modern Roman",
    "CMU Serif",
    "DejaVu Serif",
)


@dataclass(frozen=True, slots=True)
class InfoDVADesignSummaryOutputs:
    feature_summary_csv: Path
    total_summary_csv: Path
    heatmap_pngs: tuple[Path, ...]
    heatmap_pdfs: tuple[Path, ...]
    stacked_bar_pngs: tuple[Path, ...]
    stacked_bar_pdfs: tuple[Path, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create EMS InfoDVA design heatmaps and stacked feature bars averaged "
            "across model runs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Root containing xgb_*/manifest.csv and xgb_*/models/xgb_*/runs outputs.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory where plot PNG/PDF files and summary CSVs are written.",
    )
    parser.add_argument(
        "--coverage-radii-km",
        type=float,
        nargs="+",
        default=DEFAULT_COVERAGE_RADII_KM,
        help="Coverage-radius grid to validate and show on heatmaps.",
    )
    parser.add_argument(
        "--facility-budgets",
        type=int,
        nargs="+",
        default=DEFAULT_FACILITY_BUDGETS,
        help="Facility-budget grid to validate and show on heatmaps and bars.",
    )
    parser.add_argument(
        "--stacked-bar-radius-km",
        type=float,
        default=DEFAULT_STACKED_BAR_RADIUS_KM,
        help="Coverage radius used for the stacked feature contribution bars.",
    )
    parser.add_argument(
        "--cmap",
        default=DEFAULT_CMAP,
        help="Registered Matplotlib colormap name for heatmaps.",
    )
    parser.add_argument(
        "--common-scale",
        action="store_true",
        help=(
            "Use one colorbar limit for pre/post heatmaps and one y-axis limit "
            "for pre/post stacked bars. By default each plot scales independently."
        ),
    )
    parser.add_argument(
        "--heatmap-vmax",
        type=float,
        default=None,
        help="Optional heatmap colorbar maximum. Overrides --common-scale for heatmaps.",
    )
    parser.add_argument(
        "--bar-ymax",
        type=float,
        default=None,
        help="Optional stacked-bar y-axis maximum. Overrides --common-scale for bars.",
    )
    parser.add_argument(
        "--allow-incomplete-grid",
        action="store_true",
        help="Average available model/design cells instead of requiring a complete grid.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = write_ems_infodva_design_summary_plots(
        result_root=args.result_root,
        outdir=args.outdir,
        coverage_radii_km=tuple(args.coverage_radii_km),
        facility_budgets=tuple(args.facility_budgets),
        stacked_bar_radius_km=args.stacked_bar_radius_km,
        cmap_name=args.cmap,
        common_scale=args.common_scale,
        heatmap_vmax=args.heatmap_vmax,
        bar_ymax=args.bar_ymax,
        allow_incomplete_grid=args.allow_incomplete_grid,
    )
    print(f"Wrote EMS InfoDVA feature summary to {outputs.feature_summary_csv}")
    print(f"Wrote EMS InfoDVA total summary to {outputs.total_summary_csv}")
    for path in (*outputs.heatmap_pngs, *outputs.heatmap_pdfs):
        print(f"Wrote EMS InfoDVA design heatmap to {path}")
    for path in (*outputs.stacked_bar_pngs, *outputs.stacked_bar_pdfs):
        print(f"Wrote EMS InfoDVA stacked bars to {path}")


def write_ems_infodva_design_summary_plots(
    *,
    result_root: Path,
    outdir: Path,
    coverage_radii_km: Sequence[float] = DEFAULT_COVERAGE_RADII_KM,
    facility_budgets: Sequence[int] = DEFAULT_FACILITY_BUDGETS,
    stacked_bar_radius_km: float = DEFAULT_STACKED_BAR_RADIUS_KM,
    cmap_name: str = DEFAULT_CMAP,
    common_scale: bool = False,
    heatmap_vmax: float | None = None,
    bar_ymax: float | None = None,
    allow_incomplete_grid: bool = False,
) -> InfoDVADesignSummaryOutputs:
    outdir.mkdir(parents=True, exist_ok=True)
    coverage_radii = tuple(float(radius) for radius in coverage_radii_km)
    budgets = tuple(int(budget) for budget in facility_budgets)

    records = load_info_dva_feature_records(
        result_root,
        coverage_radii_km=coverage_radii,
        facility_budgets=budgets,
        allow_incomplete_grid=allow_incomplete_grid,
    )
    feature_summary = aggregate_feature_attributions(records)
    total_summary = aggregate_total_attributions(records)

    feature_summary_csv = outdir / "ems_infodva_design_feature_mean_abs_shap.csv"
    total_summary_csv = outdir / "ems_infodva_design_total_mean_abs_shap.csv"
    feature_summary.to_csv(feature_summary_csv, index=False)
    total_summary.to_csv(total_summary_csv, index=False)

    apply_plot_style()
    mode_heatmap_vmax = _mode_vmax_by_key(
        total_summary,
        mode_column="value_mode",
        value_column="mean_total_abs_shap",
        explicit_vmax=heatmap_vmax,
        common_scale=common_scale,
    )
    mode_bar_ymax = _mode_vmax_by_key(
        total_summary.loc[
            np.isclose(total_summary["coverage_radius_km"], stacked_bar_radius_km)
        ],
        mode_column="value_mode",
        value_column="mean_total_abs_shap",
        explicit_vmax=bar_ymax,
        common_scale=common_scale,
        headroom=1.10,
    )

    heatmap_pngs: list[Path] = []
    heatmap_pdfs: list[Path] = []
    stacked_bar_pngs: list[Path] = []
    stacked_bar_pdfs: list[Path] = []
    for mode in MODE_COLUMN_NAMES:
        heatmap_stem = f"ems_infodva_{mode}_mean_abs_design_heatmap"
        heatmap_paths = (
            outdir / f"{heatmap_stem}.png",
            outdir / f"{heatmap_stem}.pdf",
        )
        plot_design_heatmap(
            total_summary=total_summary,
            value_mode=mode,
            coverage_radii_km=coverage_radii,
            facility_budgets=budgets,
            output_paths=heatmap_paths,
            cmap_name=cmap_name,
            vmax=mode_heatmap_vmax[mode],
        )
        heatmap_pngs.append(heatmap_paths[0])
        heatmap_pdfs.append(heatmap_paths[1])

        radius_label = _radius_label(stacked_bar_radius_km)
        bar_stem = f"ems_infodva_{mode}_mean_abs_feature_bars_tau{radius_label}"
        bar_paths = (
            outdir / f"{bar_stem}.png",
            outdir / f"{bar_stem}.pdf",
        )
        plot_feature_stacked_bars(
            feature_summary=feature_summary,
            value_mode=mode,
            coverage_radius_km=stacked_bar_radius_km,
            facility_budgets=budgets,
            output_paths=bar_paths,
            ymax=mode_bar_ymax[mode],
        )
        stacked_bar_pngs.append(bar_paths[0])
        stacked_bar_pdfs.append(bar_paths[1])

    return InfoDVADesignSummaryOutputs(
        feature_summary_csv=feature_summary_csv,
        total_summary_csv=total_summary_csv,
        heatmap_pngs=tuple(heatmap_pngs),
        heatmap_pdfs=tuple(heatmap_pdfs),
        stacked_bar_pngs=tuple(stacked_bar_pngs),
        stacked_bar_pdfs=tuple(stacked_bar_pdfs),
    )


def load_info_dva_feature_records(
    result_root: Path,
    *,
    coverage_radii_km: Sequence[float],
    facility_budgets: Sequence[int],
    allow_incomplete_grid: bool = False,
) -> pd.DataFrame:
    manifest_paths = sorted(result_root.glob("xgb_*/manifest.csv"))
    if not manifest_paths:
        raise FileNotFoundError(
            f"No manifests found under {result_root!s} with pattern xgb_*/manifest.csv."
        )

    expected_grid = {
        (float(radius), int(budget))
        for radius in coverage_radii_km
        for budget in facility_budgets
    }
    records: list[dict[str, object]] = []
    grid_by_model: dict[str, set[tuple[float, int]]] = {}
    for manifest_path in manifest_paths:
        manifest = pd.read_csv(manifest_path)
        _require_columns(
            manifest,
            {
                "setting_id",
                "model_id",
                "results_dir",
                "coverage_radius_km",
                "facility_budget",
            },
            source=manifest_path,
        )
        for manifest_row in manifest.to_dict(orient="records"):
            model_id = str(manifest_row["model_id"])
            coverage_radius_km = float(manifest_row["coverage_radius_km"])
            facility_budget = int(manifest_row["facility_budget"])
            if (coverage_radius_km, facility_budget) not in expected_grid:
                continue

            results_dir = _resolve_manifest_path(
                manifest_path,
                Path(str(manifest_row["results_dir"])),
            )
            summary_path = results_dir / "summary_shap.csv"
            if not summary_path.exists():
                raise FileNotFoundError(
                    f"Missing summary_shap.csv for {manifest_row['setting_id']}: "
                    f"{summary_path}"
                )
            summary = pd.read_csv(summary_path)
            _require_columns(
                summary,
                {"feature", *MODE_COLUMN_NAMES.values()},
                source=summary_path,
            )
            feature_values = summary.set_index("feature")
            missing_features = [feature for feature, _ in FEATURES if feature not in feature_values.index]
            if missing_features:
                raise KeyError(
                    f"{summary_path} is missing feature rows: "
                    + ", ".join(missing_features)
                )

            grid_by_model.setdefault(model_id, set()).add(
                (coverage_radius_km, facility_budget)
            )
            for value_mode, column in MODE_COLUMN_NAMES.items():
                for feature, feature_label in FEATURES:
                    records.append(
                        {
                            "value_mode": value_mode,
                            "value_mode_label": MODE_LABELS[value_mode],
                            "model_id": model_id,
                            "setting_id": str(manifest_row["setting_id"]),
                            "coverage_radius_km": coverage_radius_km,
                            "facility_budget": facility_budget,
                            "feature": feature,
                            "feature_label": feature_label,
                            "mean_abs_shap": float(feature_values.at[feature, column]),
                            "results_dir": str(results_dir),
                        }
                    )

    if not allow_incomplete_grid:
        incomplete_models = {
            model_id: sorted(expected_grid - model_grid)
            for model_id, model_grid in sorted(grid_by_model.items())
            if model_grid != expected_grid
        }
        if incomplete_models:
            details = "; ".join(
                f"{model_id}: {missing}" for model_id, missing in incomplete_models.items()
            )
            raise ValueError(f"InfoDVA design grid is incomplete: {details}")

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No InfoDVA feature records matched the requested design grid.")
    return frame


def aggregate_feature_attributions(records: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        records,
        {
            "value_mode",
            "value_mode_label",
            "model_id",
            "coverage_radius_km",
            "facility_budget",
            "feature",
            "feature_label",
            "mean_abs_shap",
        },
        source="records",
    )
    grouped = (
        records.groupby(
            [
                "value_mode",
                "value_mode_label",
                "coverage_radius_km",
                "facility_budget",
                "feature",
                "feature_label",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            std_abs_shap=("mean_abs_shap", "std"),
            n_models=("model_id", "nunique"),
        )
        .sort_values(
            [
                "value_mode",
                "coverage_radius_km",
                "facility_budget",
                "feature",
            ],
            key=_sort_feature_column,
        )
        .reset_index(drop=True)
    )
    grouped["std_abs_shap"] = grouped["std_abs_shap"].fillna(0.0)
    return grouped


def aggregate_total_attributions(records: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        records,
        {
            "value_mode",
            "value_mode_label",
            "model_id",
            "coverage_radius_km",
            "facility_budget",
            "mean_abs_shap",
        },
        source="records",
    )
    model_totals = (
        records.groupby(
            [
                "value_mode",
                "value_mode_label",
                "model_id",
                "coverage_radius_km",
                "facility_budget",
            ],
            as_index=False,
            sort=False,
        )["mean_abs_shap"]
        .sum()
        .rename(columns={"mean_abs_shap": "total_abs_shap"})
    )
    totals = (
        model_totals.groupby(
            [
                "value_mode",
                "value_mode_label",
                "coverage_radius_km",
                "facility_budget",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            mean_total_abs_shap=("total_abs_shap", "mean"),
            std_total_abs_shap=("total_abs_shap", "std"),
            n_models=("model_id", "nunique"),
        )
        .sort_values(["value_mode", "coverage_radius_km", "facility_budget"])
        .reset_index(drop=True)
    )
    totals["std_total_abs_shap"] = totals["std_total_abs_shap"].fillna(0.0)
    return totals


def design_total_matrix(
    total_summary: pd.DataFrame,
    *,
    value_mode: str,
    coverage_radii_km: Sequence[float],
    facility_budgets: Sequence[int],
) -> pd.DataFrame:
    mode_frame = total_summary.loc[total_summary["value_mode"].eq(value_mode)]
    matrix = (
        mode_frame.pivot(
            index="coverage_radius_km",
            columns="facility_budget",
            values="mean_total_abs_shap",
        )
        .reindex(index=coverage_radii_km, columns=facility_budgets)
        .astype(float)
    )
    if matrix.isna().any().any():
        missing = [
            (float(radius), int(budget))
            for radius in matrix.index
            for budget in matrix.columns
            if pd.isna(matrix.loc[radius, budget])
        ]
        raise ValueError(
            f"Missing total summary values for {value_mode}: {missing}"
        )
    return matrix


def feature_bar_matrix(
    feature_summary: pd.DataFrame,
    *,
    value_mode: str,
    coverage_radius_km: float,
    facility_budgets: Sequence[int],
) -> pd.DataFrame:
    mode_frame = feature_summary.loc[
        feature_summary["value_mode"].eq(value_mode)
        & np.isclose(feature_summary["coverage_radius_km"], coverage_radius_km)
    ]
    matrix = (
        mode_frame.pivot(
            index="feature",
            columns="facility_budget",
            values="mean_abs_shap",
        )
        .reindex(index=[feature for feature, _ in FEATURES], columns=facility_budgets)
        .astype(float)
    )
    if matrix.isna().any().any():
        missing = [
            (feature, int(budget))
            for feature in matrix.index
            for budget in matrix.columns
            if pd.isna(matrix.loc[feature, budget])
        ]
        raise ValueError(
            f"Missing feature summary values for {value_mode} at "
            f"coverage_radius_km={coverage_radius_km:g}: {missing}"
        )
    return matrix


def register_latin_modern_fonts() -> None:
    font_dirs = (
        Path.home() / "Library" / "Fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
    )
    for font_dir in font_dirs:
        if not font_dir.exists():
            continue
        for font_path in font_dir.glob("lmroman*.otf"):
            font_manager.fontManager.addfont(font_path)


def choose_serif_font() -> str:
    register_latin_modern_fonts()
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in FONT_CANDIDATES:
        if candidate in available:
            return candidate
    return "DejaVu Serif"


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [choose_serif_font()],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 320,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_design_heatmap(
    *,
    total_summary: pd.DataFrame,
    value_mode: str,
    coverage_radii_km: Sequence[float],
    facility_budgets: Sequence[int],
    output_paths: tuple[Path, Path],
    cmap_name: str,
    vmax: float | None = None,
) -> None:
    matrix = design_total_matrix(
        total_summary,
        value_mode=value_mode,
        coverage_radii_km=coverage_radii_km,
        facility_budgets=facility_budgets,
    )
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Heatmap values for {value_mode} contain non-finite values.")

    color_vmax = float(vmax) if vmax is not None else float(np.nanmax(values))
    if color_vmax <= 0:
        color_vmax = 1.0

    fig, ax = plt.subplots(figsize=(6.3, 5.6))
    image = ax.imshow(
        values,
        cmap=plt.get_cmap(cmap_name),
        norm=Normalize(vmin=0.0, vmax=color_vmax),
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xticks(np.arange(len(facility_budgets)))
    ax.set_xticklabels([str(int(budget)) for budget in facility_budgets])
    ax.set_yticks(np.arange(len(coverage_radii_km)))
    ax.set_yticklabels([f"{float(radius):g}" for radius in coverage_radii_km])
    ax.set_xticks(np.arange(-0.5, len(facility_budgets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(coverage_radii_km), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel(r"Facility budget $p$", labelpad=10)
    ax.set_ylabel(r"Coverage radius $\tau$ (km)", labelpad=12)

    colorbar = fig.colorbar(image, ax=ax, pad=0.035, fraction=0.052)
    colorbar.set_label(HEATMAP_COLORBAR_LABEL, rotation=90, labelpad=16)
    colorbar.outline.set_linewidth(1.2)

    fig.subplots_adjust(left=0.14, right=0.88, top=0.97, bottom=0.14)
    _save_figure(fig, output_paths)


def plot_feature_stacked_bars(
    *,
    feature_summary: pd.DataFrame,
    value_mode: str,
    coverage_radius_km: float,
    facility_budgets: Sequence[int],
    output_paths: tuple[Path, Path],
    ymax: float | None = None,
) -> None:
    matrix = feature_bar_matrix(
        feature_summary,
        value_mode=value_mode,
        coverage_radius_km=coverage_radius_km,
        facility_budgets=facility_budgets,
    )
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            f"Stacked bar values for {value_mode} contain non-finite values."
        )

    x = np.arange(len(facility_budgets))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bottoms = np.zeros(len(facility_budgets), dtype=float)
    for feature, _ in FEATURES:
        heights = matrix.loc[feature].to_numpy(dtype=float)
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            width=0.58,
            color=FEATURE_COLORS[feature],
            edgecolor="white",
            linewidth=0.8,
            label=FEATURE_LABELS[feature],
        )
        bottoms += heights

    y_limit = float(ymax) if ymax is not None else float(bottoms.max()) * 1.10
    if y_limit <= 0:
        y_limit = 1.0
    ax.set_ylim(0.0, y_limit)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(budget)) for budget in facility_budgets])
    ax.set_xlabel(r"Facility budget $p$")
    ax.set_ylabel(BAR_Y_AXIS_LABEL)
    ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.11, right=0.76, top=0.96, bottom=0.16)
    _save_figure(fig, output_paths)


def _save_figure(fig: plt.Figure, output_paths: tuple[Path, Path]) -> None:
    for output_path in output_paths:
        save_kwargs = {"bbox_inches": "tight"}
        if output_path.suffix.lower() == ".png":
            save_kwargs["dpi"] = 320
        fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def _mode_vmax_by_key(
    frame: pd.DataFrame,
    *,
    mode_column: str,
    value_column: str,
    explicit_vmax: float | None,
    common_scale: bool,
    headroom: float = 1.0,
) -> dict[str, float | None]:
    if explicit_vmax is not None:
        return {mode: explicit_vmax for mode in MODE_COLUMN_NAMES}
    if frame.empty:
        return {mode: None for mode in MODE_COLUMN_NAMES}
    if common_scale:
        value = float(frame[value_column].max()) * headroom
        return {mode: value for mode in MODE_COLUMN_NAMES}
    return {
        mode: float(mode_frame[value_column].max()) * headroom
        for mode, mode_frame in frame.groupby(mode_column, sort=False)
    }


def _resolve_manifest_path(manifest_path: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    candidates = [Path.cwd(), *manifest_path.resolve().parents]
    for base in candidates:
        candidate = base / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def _require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    *,
    source: Path | str,
) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise KeyError(f"{source} is missing columns: {', '.join(sorted(missing))}")


def _sort_feature_column(column: pd.Series) -> pd.Series:
    if column.name != "feature":
        return column
    order = {feature: idx for idx, (feature, _) in enumerate(FEATURES)}
    return column.map(order).fillna(len(order))


def _radius_label(radius_km: float) -> str:
    return f"{float(radius_km):g}".replace(".", "p")


if __name__ == "__main__":
    main()
