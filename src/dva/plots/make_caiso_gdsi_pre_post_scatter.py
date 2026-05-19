from __future__ import annotations

import argparse
import json
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


DEFAULT_RESULT_ROOT = Path("results/caiso/gdsi")
DEFAULT_OUTDIR = Path("data/plots/caiso_gdsi_pre_post_scatter")
DEFAULT_EVALUATION_LABEL = "validation_baseline"
DEFAULT_OUTPUT_STEM = "validation_pre_post_scatter_all_features_by_prepost_feature_value"
BEESWARM_CMAP = plt.get_cmap("cmc.vik")
FEATURE_VALUE_CLIP_PERCENTILES = (5, 95)
MISSING_FEATURE_COLOR = "#7f7f7f"
POINT_EDGE_COLOR = "#242424"
ANTEPOS_QUADRANT_COLOR = "#d9a0a0"
ANTEPOS_QUADRANT_EDGE_COLOR = "#9f2f2f"

FEATURES = (
    ("min_temp_c", "Min temp"),
    ("max_temp_c", "Max temp"),
    ("mean_temp_c", "Mean temp"),
    ("mean_humidity", "Mean humidity"),
    ("mean_wind_speed", "Mean wind speed"),
    ("mean_solar_irradiance", "Mean solar irradiance"),
    ("max_solar_irradiance", "Max solar irradiance"),
    ("day_of_week", "Day of week"),
)
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
            "Plot paired pre-DVA and post-DVA attribution values for selected "
            "CAISO GDSI features."
        )
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Root containing xgb_*/models/xgb_*/<evaluation-label>/daily_shap.csv.",
    )
    parser.add_argument(
        "--evaluation-label",
        default=DEFAULT_EVALUATION_LABEL,
        help="Evaluation subdirectory to read, e.g. test_baseline.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Optional CAISO dataset CSV containing date and feature columns. "
            "When omitted, the path is read from the first run_metadata.json."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory where PNG and PDF outputs are written.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Filename stem for the PNG and PDF outputs.",
    )
    parser.add_argument(
        "--axis-percentile",
        type=float,
        default=99.0,
        help="Percentile of absolute pre/post values used for symmetric axes.",
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
        help="Color scatter points by beeswarm-style feature value or use one color.",
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


def daily_shap_paths(result_root: Path, evaluation_label: str) -> list[Path]:
    pattern = f"xgb_*/models/xgb_*/{evaluation_label}/daily_shap.csv"
    paths = sorted(result_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No daily_shap.csv files found under {result_root!s} with pattern {pattern!r}."
        )
    return paths


def metadata_path(daily_shap_path: Path) -> Path:
    return daily_shap_path.with_name("run_metadata.json")


def resolve_dataset_path(paths: list[Path], dataset_path: Path | None) -> Path:
    if dataset_path is not None:
        return dataset_path
    path = metadata_path(paths[0])
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if "dataset_path" not in metadata:
        raise KeyError(f"{path} does not contain a dataset_path entry.")
    return Path(str(metadata["dataset_path"]))


def load_feature_values(dataset_path: Path) -> pd.DataFrame:
    usecols = ["date", *[feature for feature, _ in FEATURES]]
    feature_values = pd.read_csv(dataset_path, usecols=usecols)
    feature_values["date"] = pd.to_datetime(feature_values["date"])
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


def load_feature_frame(paths: list[Path], feature_values: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    usecols = ["date"]
    for feature, _ in FEATURES:
        usecols.extend([f"ead_decision_shap_{feature}", f"decision_shap_{feature}"])

    for path in paths:
        model_id = path.parents[3].name
        daily = pd.read_csv(path, usecols=usecols)
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.merge(feature_values, on="date", how="left", validate="many_to_one")
        for feature, label in FEATURES:
            rows.append(
                pd.DataFrame(
                    {
                        "model_id": model_id,
                        "date": daily["date"],
                        "feature": label,
                        "feature_value": daily[feature].to_numpy(dtype=float),
                        "feature_value_normalized": normalize_feature_values(
                            daily[feature].to_numpy(dtype=float)
                        ),
                        "pre_dva": daily[f"ead_decision_shap_{feature}"].to_numpy(
                            dtype=float
                        ),
                        "post_dva": daily[f"decision_shap_{feature}"].to_numpy(
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


def sorted_features_by_ante_post_rate(frame: pd.DataFrame) -> list[tuple[str, str]]:
    rate_by_label = (
        frame.assign(ante_post_negative=frame["pre_dva"].gt(0) & frame["post_dva"].lt(0))
        .groupby("feature", sort=False)["ante_post_negative"]
        .mean()
        .to_dict()
    )
    return sorted(
        FEATURES,
        key=lambda feature: rate_by_label.get(feature[1], float("-inf")),
        reverse=True,
    )


def ante_post_rate(feature_frame: pd.DataFrame) -> float:
    return float(feature_frame["pre_dva"].gt(0).mul(feature_frame["post_dva"].lt(0)).mean())


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
    ax.set_title(f"{label}\nPre+/Post- {rate:.1%}", fontsize=10, pad=5)


def label_grid_axes(axes: np.ndarray) -> None:
    for row in range(axes.shape[0]):
        axes[row, 0].set_ylabel("Post-DVA value")
    for col in range(axes.shape[1]):
        axes[-1, col].set_xlabel("Pre-DVA value")


def make_plot(
    frame: pd.DataFrame,
    *,
    axis_percentile: float,
    alpha: float,
    point_size: float,
    outline_width: float,
    scatter_color_mode: str,
    uniform_color: str,
) -> plt.Figure:
    apply_plot_style()

    color_norm = Normalize(vmin=0.0, vmax=1.0)
    ordered_features = sorted_features_by_ante_post_rate(frame)
    fig, axes, flat_axes = create_panel_grid()
    for ax, (_, label) in zip(flat_axes, ordered_features, strict=True):
        feature_frame = frame.loc[frame["feature"] == label]
        x = feature_frame["pre_dva"].to_numpy(dtype=float)
        y = feature_frame["post_dva"].to_numpy(dtype=float)
        color_values = feature_frame["feature_value_normalized"].to_numpy(dtype=float)
        lim = percentile_limit(x, y, axis_percentile)
        rate = ante_post_rate(feature_frame)

        style_panel(ax, label=label, rate=rate, lim=lim)
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

    label_grid_axes(axes)

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
    axis_percentile: float,
    density_bins: int,
    density_cmap: str,
) -> plt.Figure:
    if density_bins <= 0:
        raise ValueError("density_bins must be positive.")

    apply_plot_style()
    ordered_features = sorted_features_by_ante_post_rate(frame)
    panel_data: list[dict[str, object]] = []
    max_count = 0.0

    for _, label in ordered_features:
        feature_frame = frame.loc[frame["feature"] == label]
        x = feature_frame["pre_dva"].to_numpy(dtype=float)
        y = feature_frame["post_dva"].to_numpy(dtype=float)
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
                "rate": ante_post_rate(feature_frame),
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

    label_grid_axes(axes)
    scalar_mappable = plt.cm.ScalarMappable(cmap=density_map, norm=density_norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=flat_axes, pad=0.02)
    colorbar.set_label(f"Samples per {density_bins}x{density_bins} bin (%)")
    return fig


def save_outputs(fig: plt.Figure, outdir: Path, output_stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{output_stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(outdir / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = daily_shap_paths(args.result_root, args.evaluation_label)
    dataset_path = resolve_dataset_path(paths, args.dataset_path)
    feature_values = load_feature_values(dataset_path)
    frame = load_feature_frame(paths, feature_values)
    if args.plot_kind == "density":
        fig = make_density_plot(
            frame,
            axis_percentile=args.axis_percentile,
            density_bins=args.density_bins,
            density_cmap=args.density_cmap,
        )
    else:
        fig = make_plot(
            frame,
            axis_percentile=args.axis_percentile,
            alpha=args.alpha,
            point_size=args.point_size,
            outline_width=args.outline_width,
            scatter_color_mode=args.scatter_color_mode,
            uniform_color=args.uniform_color,
        )
    save_outputs(fig, args.outdir, args.output_stem)
    print(f"Wrote {args.outdir / (args.output_stem + '.png')}")
    print(f"Wrote {args.outdir / (args.output_stem + '.pdf')}")


if __name__ == "__main__":
    main()
