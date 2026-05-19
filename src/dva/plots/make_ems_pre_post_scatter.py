from __future__ import annotations

import argparse
from pathlib import Path

import cmcrameri  # noqa: F401 - registers cmc.* colormaps with Matplotlib.
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RESULT_ROOT = Path("results/ems/experiment_a_infodva")
DEFAULT_FEATURE_PATH = Path(
    "data/ems_data/processed/ems_zip_hour_features_2025_manhattan_wide_X.csv"
)
DEFAULT_OUTDIR = Path("data/plots/ems_pre_post_scatter")
BEESWARM_CMAP = plt.get_cmap("cmc.vik")
FEATURE_VALUE_CLIP_PERCENTILES = (5, 95)
MISSING_FEATURE_COLOR = "#7f7f7f"
POINT_EDGE_COLOR = "#242424"
ANTEPOS_QUADRANT_COLOR = "#d9a0a0"
ANTEPOS_QUADRANT_EDGE_COLOR = "#9f2f2f"

FEATURES = (
    ("hour", "Hour"),
    ("day_of_week", "Day of week"),
    ("temp_c", "Temperature"),
    ("precip_mm", "Precipitation"),
    ("citywide_ems_incidents_lag_1", "Citywide EMS lag 1"),
    ("ems_incidents_lag_1", "ZIP EMS lag 1"),
    ("neighbor_ems_incidents_lag_1_mean", "Neighbor EMS lag 1 mean"),
    ("zone_hour_baseline", "Zone-hour baseline"),
)
MODE_COLUMN_PREFIXES = {
    "predictive": "predictive_shap",
    "ante": "ante_decision_shap",
    "post": "decision_shap",
}
MODE_AXIS_LABELS = {
    "predictive": "Predictive SHAP value",
    "ante": "Ante-DVA value",
    "post": "Post-DVA value",
}
MODE_SHORT_LABELS = {
    "predictive": "Predictive",
    "ante": "Ante",
    "post": "Post",
}
FONT_CANDIDATES = (
    "Latin Computer Roman",
    "Latin Modern Roman",
    "Computer Modern Roman",
    "CMU Serif",
    "DejaVu Serif",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot paired EMS attribution values for all eight feature groups, "
            "comparing selectable attribution modes such as ante and post."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help=(
            "Root containing xgb_*/models/xgb_*/runs/*/hourly_shap.csv files."
        ),
    )
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
        help="EMS wide feature CSV containing timestamp_hour and feature columns.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory where PNG and PDF outputs are written.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help=(
            "Filename stem for the PNG and PDF outputs. When omitted, a stem is "
            "derived from --x-mode, --y-mode, and --plot-kind."
        ),
    )
    parser.add_argument(
        "--x-mode",
        choices=tuple(MODE_COLUMN_PREFIXES),
        default="ante",
        help="Attribution mode shown on the x-axis.",
    )
    parser.add_argument(
        "--y-mode",
        choices=tuple(MODE_COLUMN_PREFIXES),
        default="post",
        help="Attribution mode shown on the y-axis.",
    )
    parser.add_argument(
        "--zone-feature-aggregation",
        choices=("mean", "sum", "median"),
        default="mean",
        help=(
            "How ZIP-specific EMS feature columns are collapsed into one feature "
            "value for point coloring."
        ),
    )
    parser.add_argument(
        "--axis-percentile",
        type=float,
        default=99.0,
        help="Percentile of absolute x/y values used for symmetric axes.",
    )
    parser.add_argument(
        "--plot-kind",
        choices=("scatter", "density"),
        default="scatter",
        help="Draw raw scatter points or a 2D histogram density heatmap.",
    )
    parser.add_argument(
        "--density-bins",
        type=int,
        default=50,
        help="Number of bins per axis for --plot-kind density.",
    )
    parser.add_argument(
        "--density-cmap",
        default="cmc.batlow",
        help="Sequential colormap for --plot-kind density.",
    )
    parser.add_argument(
        "--scatter-color-mode",
        choices=("feature_value", "uniform"),
        default="feature_value",
        help="Color scatter points by feature value or use one color.",
    )
    parser.add_argument(
        "--uniform-color",
        default="#011959",
        help="Point color used when --scatter-color-mode uniform.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Point transparency.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=8.0,
        help="Scatter marker size.",
    )
    parser.add_argument(
        "--outline-width",
        type=float,
        default=0.0,
        help="Dark gray point outline width.",
    )
    return parser.parse_args()


def choose_serif_font() -> str:
    register_latin_modern_fonts()
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in FONT_CANDIDATES:
        if candidate in available:
            return candidate
    return "DejaVu Serif"


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


def hourly_shap_paths(result_root: Path) -> list[Path]:
    pattern = "xgb_*/models/xgb_*/runs/*/hourly_shap.csv"
    paths = sorted(result_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No hourly_shap.csv files found under {result_root!s} "
            f"with pattern {pattern!r}."
        )
    return paths


def mode_column(mode: str, feature: str) -> str:
    if mode not in MODE_COLUMN_PREFIXES:
        raise ValueError(f"Unknown attribution mode: {mode}")
    return f"{MODE_COLUMN_PREFIXES[mode]}_{feature}"


def feature_source_columns(columns: pd.Index, feature: str) -> list[str]:
    if feature in columns:
        return [feature]
    prefix = f"{feature}_zip_"
    matches = [column for column in columns if str(column).startswith(prefix)]
    if matches:
        return matches
    raise KeyError(
        f"Feature path is missing {feature!r} and columns with prefix {prefix!r}."
    )


def aggregate_zone_feature(
    frame: pd.DataFrame,
    columns: list[str],
    aggregation: str,
) -> pd.Series:
    if len(columns) == 1:
        return frame[columns[0]]
    if aggregation == "mean":
        return frame[columns].mean(axis=1)
    if aggregation == "sum":
        return frame[columns].sum(axis=1)
    if aggregation == "median":
        return frame[columns].median(axis=1)
    raise ValueError(f"Unknown zone feature aggregation: {aggregation}")


def load_feature_values(feature_path: Path, zone_feature_aggregation: str) -> pd.DataFrame:
    header = pd.read_csv(feature_path, nrows=0)
    source_by_feature = {
        feature: feature_source_columns(header.columns, feature)
        for feature, _ in FEATURES
    }
    usecols = ["timestamp_hour"]
    for columns in source_by_feature.values():
        usecols.extend(column for column in columns if column not in usecols)

    raw = pd.read_csv(feature_path, usecols=usecols)
    raw["timestamp_hour"] = pd.to_datetime(raw["timestamp_hour"])
    feature_values = pd.DataFrame({"timestamp_hour": raw["timestamp_hour"]})
    for feature, _ in FEATURES:
        feature_values[feature] = aggregate_zone_feature(
            raw,
            source_by_feature[feature],
            zone_feature_aggregation,
        )
    return feature_values


def normalize_feature_values(values: np.ndarray) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    normalized = np.full(values_array.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(values_array)
    if not finite_mask.any():
        return normalized

    finite_values = values_array[finite_mask]
    vmin, vmax = np.nanpercentile(finite_values, FEATURE_VALUE_CLIP_PERCENTILES)
    if np.isclose(vmin, vmax):
        vmin = np.nanmin(finite_values)
        vmax = np.nanmax(finite_values)
    if np.isclose(vmin, vmax):
        normalized[finite_mask] = 0.5
        return normalized

    clipped = np.clip(finite_values, vmin, vmax)
    normalized[finite_mask] = (clipped - vmin) / (vmax - vmin)
    return normalized


def load_feature_frame(
    paths: list[Path],
    feature_values: pd.DataFrame,
    *,
    x_mode: str,
    y_mode: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    usecols = ["timestamp_hour"]
    for feature, _ in FEATURES:
        usecols.extend([mode_column(x_mode, feature), mode_column(y_mode, feature)])
    usecols = list(dict.fromkeys(usecols))

    for path in paths:
        model_id = path.parents[2].name
        setting_id = path.parent.name
        hourly = pd.read_csv(path, usecols=usecols)
        hourly["timestamp_hour"] = pd.to_datetime(hourly["timestamp_hour"])
        hourly = hourly.merge(
            feature_values,
            on="timestamp_hour",
            how="left",
            validate="many_to_one",
        )
        for feature, label in FEATURES:
            rows.append(
                pd.DataFrame(
                    {
                        "model_id": model_id,
                        "setting_id": setting_id,
                        "timestamp_hour": hourly["timestamp_hour"],
                        "feature_key": feature,
                        "feature": label,
                        "feature_value": hourly[feature].to_numpy(dtype=float),
                        "feature_value_normalized": normalize_feature_values(
                            hourly[feature].to_numpy(dtype=float)
                        ),
                        "x_value": hourly[mode_column(x_mode, feature)].to_numpy(
                            dtype=float
                        ),
                        "y_value": hourly[mode_column(y_mode, feature)].to_numpy(
                            dtype=float
                        ),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def percentile_limit(x: np.ndarray, y: np.ndarray, percentile: float) -> float:
    vals = np.concatenate([x, y])
    lim = float(np.nanpercentile(np.abs(vals), percentile))
    if not np.isfinite(lim) or lim <= 0.0:
        return 1.0
    return lim


def positive_negative_rate(feature_frame: pd.DataFrame) -> float:
    return float(
        feature_frame["x_value"].gt(0).mul(feature_frame["y_value"].lt(0)).mean()
    )


def sorted_features_by_positive_negative_rate(
    frame: pd.DataFrame,
) -> list[tuple[str, str]]:
    rate_by_label = (
        frame.assign(
            x_positive_y_negative=frame["x_value"].gt(0)
            & frame["y_value"].lt(0)
        )
        .groupby("feature", sort=False)["x_positive_y_negative"]
        .mean()
        .to_dict()
    )
    return sorted(
        FEATURES,
        key=lambda feature: rate_by_label.get(feature[1], float("-inf")),
        reverse=True,
    )


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [choose_serif_font()],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.labelsize": 10,
            "figure.dpi": 320,
            "savefig.dpi": 320,
        }
    )


def create_panel_grid() -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(13.0, 7.2),
        constrained_layout=True,
        subplot_kw={"box_aspect": 1},
    )
    return fig, axes, axes.ravel()


def style_panel(
    ax: plt.Axes,
    *,
    label: str,
    rate: float,
    lim: float,
    x_mode: str,
    y_mode: str,
) -> None:
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_box_aspect(1)
    ax.add_patch(
        Rectangle(
            (0.0, -lim),
            lim,
            lim,
            facecolor=ANTEPOS_QUADRANT_COLOR,
            edgecolor="none",
            alpha=0.24,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.0, -lim),
            lim,
            lim,
            facecolor="none",
            edgecolor=ANTEPOS_QUADRANT_EDGE_COLOR,
            linewidth=1.25,
            zorder=4,
        )
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.plot([0.0, lim], [0.0, 0.0], color=ANTEPOS_QUADRANT_EDGE_COLOR, linewidth=1.6, zorder=5)
    ax.plot([0.0, 0.0], [-lim, 0.0], color=ANTEPOS_QUADRANT_EDGE_COLOR, linewidth=1.6, zorder=5)
    ax.plot([-lim, lim], [-lim, lim], "--", color="gray", linewidth=1)
    ax.grid(True, color="#e9e9e9", linewidth=0.65)
    ax.set_title(
        f"{label}\n{MODE_SHORT_LABELS[x_mode]}+/{MODE_SHORT_LABELS[y_mode]}- {rate:.1%}",
        fontsize=10,
        pad=5,
    )


def label_grid_axes(axes: np.ndarray, *, x_mode: str, y_mode: str) -> None:
    for row in range(axes.shape[0]):
        axes[row, 0].set_ylabel(MODE_AXIS_LABELS[y_mode])
    for col in range(axes.shape[1]):
        axes[-1, col].set_xlabel(MODE_AXIS_LABELS[x_mode])


def make_plot(
    frame: pd.DataFrame,
    *,
    x_mode: str,
    y_mode: str,
    axis_percentile: float,
    alpha: float,
    point_size: float,
    outline_width: float,
    scatter_color_mode: str,
    uniform_color: str,
) -> plt.Figure:
    apply_plot_style()

    color_norm = Normalize(vmin=0.0, vmax=1.0)
    ordered_features = sorted_features_by_positive_negative_rate(frame)
    fig, axes, flat_axes = create_panel_grid()
    for ax, (_, label) in zip(flat_axes, ordered_features, strict=True):
        feature_frame = frame.loc[frame["feature"] == label]
        x = feature_frame["x_value"].to_numpy(dtype=float)
        y = feature_frame["y_value"].to_numpy(dtype=float)
        color_values = feature_frame["feature_value_normalized"].to_numpy(dtype=float)
        lim = percentile_limit(x, y, axis_percentile)
        rate = positive_negative_rate(feature_frame)

        style_panel(
            ax,
            label=label,
            rate=rate,
            lim=lim,
            x_mode=x_mode,
            y_mode=y_mode,
        )
        if scatter_color_mode == "uniform":
            ax.scatter(
                x,
                y,
                color=uniform_color,
                alpha=alpha,
                s=point_size,
                edgecolors=POINT_EDGE_COLOR if outline_width > 0.0 else "none",
                linewidths=outline_width,
                rasterized=True,
                zorder=2,
            )
        else:
            finite_color = np.isfinite(color_values)
            if finite_color.any():
                ax.scatter(
                    x[finite_color],
                    y[finite_color],
                    c=color_values[finite_color],
                    cmap=BEESWARM_CMAP,
                    norm=color_norm,
                    alpha=alpha,
                    s=point_size,
                    edgecolors=POINT_EDGE_COLOR if outline_width > 0.0 else "none",
                    linewidths=outline_width,
                    rasterized=True,
                    zorder=2,
                )
            if (~finite_color).any():
                ax.scatter(
                    x[~finite_color],
                    y[~finite_color],
                    alpha=alpha,
                    s=point_size,
                    color=MISSING_FEATURE_COLOR,
                    edgecolors=POINT_EDGE_COLOR if outline_width > 0.0 else "none",
                    linewidths=outline_width,
                    rasterized=True,
                    zorder=2,
                )

    label_grid_axes(axes, x_mode=x_mode, y_mode=y_mode)

    if scatter_color_mode == "feature_value":
        scalar_mappable = plt.cm.ScalarMappable(cmap=BEESWARM_CMAP, norm=color_norm)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, ax=flat_axes, pad=0.02)
        colorbar.set_ticks([0.0, 1.0])
        colorbar.set_ticklabels(["Low", "High"])
        colorbar.set_label("Relative Feature Value")
    return fig


def make_density_plot(
    frame: pd.DataFrame,
    *,
    x_mode: str,
    y_mode: str,
    axis_percentile: float,
    density_bins: int,
    density_cmap: str,
) -> plt.Figure:
    if density_bins <= 0:
        raise ValueError("density_bins must be positive.")

    apply_plot_style()
    ordered_features = sorted_features_by_positive_negative_rate(frame)
    panel_data: list[dict[str, object]] = []
    max_count = 0.0

    for _, label in ordered_features:
        feature_frame = frame.loc[frame["feature"] == label]
        x = feature_frame["x_value"].to_numpy(dtype=float)
        y = feature_frame["y_value"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        lim = percentile_limit(x, y, axis_percentile)
        counts, x_edges, y_edges = np.histogram2d(
            x,
            y,
            bins=density_bins,
            range=[[-lim, lim], [-lim, lim]],
        )
        density_percent = counts / float(len(x)) * 100.0
        max_count = max(max_count, float(np.nanmax(density_percent)))
        panel_data.append(
            {
                "label": label,
                "rate": positive_negative_rate(feature_frame),
                "lim": lim,
                "density_percent": density_percent.T,
                "x_edges": x_edges,
                "y_edges": y_edges,
            }
        )

    density_norm = Normalize(vmin=0.0, vmax=max_count if max_count > 0.0 else 1.0)
    density_map = plt.get_cmap(density_cmap).copy()
    density_map.set_bad((1.0, 1.0, 1.0, 0.0))
    fig, axes, flat_axes = create_panel_grid()

    for ax, panel in zip(flat_axes, panel_data, strict=True):
        lim = float(panel["lim"])
        style_panel(
            ax,
            label=str(panel["label"]),
            rate=float(panel["rate"]),
            lim=lim,
            x_mode=x_mode,
            y_mode=y_mode,
        )
        density_percent = np.asarray(panel["density_percent"], dtype=float)
        masked_density = np.ma.masked_where(density_percent <= 0.0, density_percent)
        ax.pcolormesh(
            np.asarray(panel["x_edges"], dtype=float),
            np.asarray(panel["y_edges"], dtype=float),
            masked_density,
            cmap=density_map,
            norm=density_norm,
            shading="auto",
            rasterized=True,
            zorder=1,
        )

    label_grid_axes(axes, x_mode=x_mode, y_mode=y_mode)
    scalar_mappable = plt.cm.ScalarMappable(cmap=density_map, norm=density_norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=flat_axes, pad=0.02)
    colorbar.set_label(f"Samples per {density_bins}x{density_bins} bin (%)")
    return fig


def default_output_stem(x_mode: str, y_mode: str, plot_kind: str) -> str:
    plot_descriptor = "density" if plot_kind == "density" else "scatter"
    return f"ems_{x_mode}_{y_mode}_{plot_descriptor}_all_features_by_feature_value"


def save_outputs(fig: plt.Figure, outdir: Path, output_stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{output_stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(outdir / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = hourly_shap_paths(args.result_root)
    feature_values = load_feature_values(
        args.feature_path,
        args.zone_feature_aggregation,
    )
    frame = load_feature_frame(
        paths,
        feature_values,
        x_mode=args.x_mode,
        y_mode=args.y_mode,
    )
    if args.plot_kind == "density":
        fig = make_density_plot(
            frame,
            x_mode=args.x_mode,
            y_mode=args.y_mode,
            axis_percentile=args.axis_percentile,
            density_bins=args.density_bins,
            density_cmap=args.density_cmap,
        )
    else:
        fig = make_plot(
            frame,
            x_mode=args.x_mode,
            y_mode=args.y_mode,
            axis_percentile=args.axis_percentile,
            alpha=args.alpha,
            point_size=args.point_size,
            outline_width=args.outline_width,
            scatter_color_mode=args.scatter_color_mode,
            uniform_color=args.uniform_color,
        )

    output_stem = args.output_stem or default_output_stem(
        args.x_mode,
        args.y_mode,
        args.plot_kind,
    )
    save_outputs(fig, args.outdir, output_stem)
    print(f"Wrote {args.outdir / (output_stem + '.png')}")
    print(f"Wrote {args.outdir / (output_stem + '.pdf')}")


if __name__ == "__main__":
    main()
