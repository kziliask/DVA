from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from dva.plots.make_ems_joint_dvi_interaction_heatmaps import (
    FEATURES,
    GRID_COLOR,
    INTERACTION_CMAP,
    PERCENT_SCALE,
    TEXT_COLOR,
    TEXT_COLOR_ON_DARK,
    apply_plot_style,
    round_percent_columns,
)
from dva.plots.make_ems_solver_joint_dvi_interaction_heatmaps import (
    DEFAULT_RESULT_ROOT,
    load_interaction_frame,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTDIR = Path("data/plots/ems_solver_cross_dvi_summary")
DEFAULT_OUTPUT_PREFIX = "ems_solver_cross_dvi"
SUMMARY_FLOAT_FORMAT = "%.6f"
COMPARISON_ORDER = ("exact_vs_greedy", "exact_vs_naive")
VALUE_MODE_ORDER = ("ante", "post")
SCENARIO_ORDER = tuple(
    (comparison, value_mode)
    for comparison in COMPARISON_ORDER
    for value_mode in VALUE_MODE_ORDER
)
COMPARISON_LABELS = {
    "exact_vs_greedy": "Exact -> Greedy",
    "exact_vs_naive": "Exact -> Naive",
}
HEATMAP_COLUMN_LABELS = {
    ("exact_vs_greedy", "ante"): "Greedy\npre",
    ("exact_vs_greedy", "post"): "Greedy\npost",
    ("exact_vs_naive", "ante"): "Naive\npre",
    ("exact_vs_naive", "post"): "Naive\npost",
}
FEATURE_LABELS = dict(FEATURES)
MODE_COLORS = {
    "ante": INTERACTION_CMAP(0.68),
    "post": INTERACTION_CMAP(0.88),
}
MODE_LABELS = {
    "ante": "pre",
    "post": "post",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate concise EMS solver Cross-DVI summary figures: aggregate "
            "magnitude bars and a feature-by-solver signed interaction heatmap."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help=(
            "Root containing xgb_*/exact_vs_<solver>_<value-mode>/"
            "dvi_interactions.csv."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory where figure PNG/PDF files and CSV summaries are written.",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix used for output filenames.",
    )
    parser.add_argument(
        "--value-format",
        default=".3f",
        help="Format specifier used for bar and heatmap annotations.",
    )
    parser.add_argument(
        "--heatmap-vmax",
        type=float,
        default=None,
        help=(
            "Optional absolute signed color limit in percentage points for the "
            "feature heatmap. When omitted, the observed maximum absolute signed "
            "mean is used."
        ),
    )
    return parser


def load_solver_cross_dvi(result_root: Path) -> pd.DataFrame:
    frame = load_interaction_frame(result_root)
    frame = frame.loc[frame["design_parameter"].eq("solver")].copy()
    if frame.empty:
        raise ValueError(f"No feature-solver Cross-DVI rows found under {result_root!s}.")
    frame["decision_interaction_value"] = pd.to_numeric(
        frame["decision_interaction_value"],
        errors="raise",
    )
    frame["solver_comparison"] = pd.Categorical(
        frame["solver_comparison"],
        categories=COMPARISON_ORDER,
        ordered=True,
    )
    frame["value_mode"] = pd.Categorical(
        frame["value_mode"],
        categories=VALUE_MODE_ORDER,
        ordered=True,
    )
    return frame.sort_values(["solver_comparison", "value_mode", "feature"])


def _agg_cross_dvi(grouped: pd.core.groupby.SeriesGroupBy) -> pd.DataFrame:
    return grouped.agg(
        n="size",
        signed_mean="mean",
        signed_median="median",
        mean_abs=lambda series: series.abs().mean(),
        median_abs=lambda series: series.abs().median(),
        p95_abs=lambda series: series.abs().quantile(0.95),
        negative_count=lambda series: (series < -1e-12).sum(),
        zero_count=lambda series: (series.abs() <= 1e-12).sum(),
        positive_count=lambda series: (series > 1e-12).sum(),
    ).reset_index()


def aggregate_overall(frame: pd.DataFrame) -> pd.DataFrame:
    summary = _agg_cross_dvi(
        frame.groupby(["solver_comparison", "value_mode"], observed=False)[
            "decision_interaction_value"
        ]
    )
    summary["positive_share"] = summary["positive_count"] / summary["n"]
    summary["negative_share"] = summary["negative_count"] / summary["n"]
    return _add_scaled_columns(summary)


def aggregate_by_feature(frame: pd.DataFrame) -> pd.DataFrame:
    summary = _agg_cross_dvi(
        frame.groupby(["solver_comparison", "value_mode", "feature"], observed=False)[
            "decision_interaction_value"
        ]
    )
    summary["positive_share"] = summary["positive_count"] / summary["n"]
    summary["negative_share"] = summary["negative_count"] / summary["n"]
    return _add_scaled_columns(summary)


def _add_scaled_columns(frame: pd.DataFrame) -> pd.DataFrame:
    scaled = frame.copy()
    for column in ("signed_mean", "signed_median", "mean_abs", "median_abs", "p95_abs"):
        scaled[f"{column}_pct"] = scaled[column] * PERCENT_SCALE
    return scaled


def _ordered_overall(summary: pd.DataFrame) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        SCENARIO_ORDER,
        names=["solver_comparison", "value_mode"],
    )
    return summary.set_index(["solver_comparison", "value_mode"]).reindex(index).reset_index()


def _feature_order(feature_summary: pd.DataFrame) -> list[str]:
    naive_post = feature_summary.loc[
        feature_summary["solver_comparison"].eq("exact_vs_naive")
        & feature_summary["value_mode"].eq("post")
    ]
    observed = (
        naive_post.sort_values("mean_abs", ascending=False)["feature"].astype(str).tolist()
    )
    extras = [feature for feature, _ in FEATURES if feature not in observed]
    return observed + extras


def _format_percent(value: float, value_format: str) -> str:
    if not np.isfinite(value):
        return "NA"
    formatted = format(float(value), value_format)
    if formatted.startswith("-0") and float(formatted) == 0.0:
        formatted = formatted[1:]
    return formatted


def _format_signed_percent(value: float, value_format: str) -> str:
    if not np.isfinite(value):
        return "NA"
    formatted = format(float(value), f"+{value_format}")
    if formatted.startswith("-0") and float(formatted) == 0.0:
        formatted = "+" + formatted[1:]
    return formatted


def write_summary_csvs(
    *,
    overall_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    outdir: Path,
    output_prefix: str,
) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    overall_path = outdir / f"{output_prefix}_magnitude_summary.csv"
    feature_path = outdir / f"{output_prefix}_feature_summary.csv"
    percent_columns = tuple(
        column
        for column in overall_summary.columns
        if column.endswith("_pct")
    )
    round_percent_columns(overall_summary, percent_columns).to_csv(
        overall_path,
        index=False,
        float_format=SUMMARY_FLOAT_FORMAT,
    )
    round_percent_columns(feature_summary, percent_columns).to_csv(
        feature_path,
        index=False,
        float_format=SUMMARY_FLOAT_FORMAT,
    )
    return overall_path, feature_path


def plot_magnitude_summary(
    overall_summary: pd.DataFrame,
    *,
    outdir: Path,
    output_prefix: str,
    value_format: str,
) -> tuple[Path, Path]:
    summary = _ordered_overall(overall_summary)
    x_centers = np.arange(len(COMPARISON_ORDER), dtype=float)
    offsets = {"ante": -0.18, "post": 0.18}
    width = 0.30

    fig, ax = plt.subplots(figsize=(4.7, 2.85), constrained_layout=True)
    for value_mode in VALUE_MODE_ORDER:
        scenario = summary.loc[summary["value_mode"].eq(value_mode)]
        x = np.array(
            [
                x_centers[COMPARISON_ORDER.index(str(comparison))]
                + offsets[value_mode]
                for comparison in scenario["solver_comparison"]
            ],
            dtype=float,
        )
        heights = scenario["mean_abs_pct"].to_numpy(dtype=float)
        upper = scenario["p95_abs_pct"].to_numpy(dtype=float) - heights
        ax.bar(
            x,
            heights,
            width=width,
            color=MODE_COLORS[value_mode],
            edgecolor=TEXT_COLOR,
            linewidth=0.7,
            label=f"{MODE_LABELS[value_mode]} mean |Cross-DVI|",
            zorder=3,
        )
        ax.errorbar(
            x,
            heights,
            yerr=np.vstack([np.zeros_like(upper), upper]),
            fmt="none",
            ecolor=TEXT_COLOR,
            elinewidth=0.8,
            capsize=2.3,
            capthick=0.8,
            zorder=4,
        )
        ax.scatter(
            x,
            scenario["signed_mean_pct"].to_numpy(dtype=float),
            marker="D",
            s=22,
            facecolor="white",
            edgecolor=TEXT_COLOR,
            linewidth=0.8,
            zorder=5,
        )
        for xpos, height in zip(x, heights, strict=True):
            ax.text(
                xpos - width * 0.14,
                height + 0.035,
                _format_percent(height, value_format),
                ha="right",
                va="bottom",
                fontsize=8,
                color=TEXT_COLOR,
            )

    max_p95 = float(summary["p95_abs_pct"].max())
    min_signed = min(0.0, float(summary["signed_mean_pct"].min()))
    ax.axhline(0.0, color=TEXT_COLOR, linewidth=0.7, zorder=2)
    ax.set_ylim(min_signed - 0.12, max_p95 * 1.15)
    ax.set_ylabel("Mean |Cross-DVI| (% points)")
    ax.set_xticks(x_centers, [COMPARISON_LABELS[comparison] for comparison in COMPARISON_ORDER])
    ax.set_xlabel("Solver switch")
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="white",
            markeredgecolor=TEXT_COLOR,
            markersize=5,
            label="Signed mean",
        )
    )
    labels.append("Signed mean")
    handles.append(
        Line2D(
            [0],
            [0],
            color=TEXT_COLOR,
            linewidth=0.8,
            marker="_",
            markersize=7,
            label="95th pct |Cross-DVI|",
        )
    )
    labels.append("95th pct |Cross-DVI|")
    ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=False,
        fontsize=8,
        handlelength=1.2,
    )

    output_stem = outdir / f"{output_prefix}_magnitude"
    return _save_figure(fig, output_stem)


def plot_feature_heatmap(
    feature_summary: pd.DataFrame,
    *,
    outdir: Path,
    output_prefix: str,
    value_format: str,
    heatmap_vmax: float | None,
) -> tuple[Path, Path]:
    feature_order = _feature_order(feature_summary)
    row_labels = [FEATURE_LABELS.get(feature, feature) for feature in feature_order]
    column_labels = [HEATMAP_COLUMN_LABELS[scenario] for scenario in SCENARIO_ORDER]

    signed_matrix = np.full((len(feature_order), len(SCENARIO_ORDER)), np.nan)
    abs_matrix = np.full_like(signed_matrix, np.nan)
    summary_index = feature_summary.set_index(
        ["feature", "solver_comparison", "value_mode"]
    )
    for row_idx, feature in enumerate(feature_order):
        for col_idx, scenario in enumerate(SCENARIO_ORDER):
            key = (feature, *scenario)
            if key not in summary_index.index:
                continue
            row = summary_index.loc[key]
            signed_matrix[row_idx, col_idx] = float(row["signed_mean_pct"])
            abs_matrix[row_idx, col_idx] = float(row["mean_abs_pct"])

    observed_abs = np.abs(signed_matrix[np.isfinite(signed_matrix)])
    if heatmap_vmax is not None:
        if heatmap_vmax <= 0.0:
            raise ValueError("--heatmap-vmax must be positive when provided.")
        limit = heatmap_vmax
    elif observed_abs.size:
        limit = max(float(observed_abs.max()), 0.1)
    else:
        limit = 1.0

    cmap = INTERACTION_CMAP.copy()
    cmap.set_bad("#f2f2f2")
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, ax = plt.subplots(figsize=(5.15, 4.0), constrained_layout=True)
    image = ax.imshow(
        np.ma.masked_invalid(signed_matrix),
        cmap=cmap,
        norm=norm,
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(column_labels)), column_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_xticks(np.arange(-0.5, len(column_labels), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1.0), minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("")
    ax.set_ylabel("EMS feature")
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_idx in range(signed_matrix.shape[0]):
        for col_idx in range(signed_matrix.shape[1]):
            signed_value = signed_matrix[row_idx, col_idx]
            abs_value = abs_matrix[row_idx, col_idx]
            if not np.isfinite(signed_value) or not np.isfinite(abs_value):
                ax.text(
                    col_idx,
                    row_idx,
                    "NA",
                    ha="center",
                    va="center",
                    color=TEXT_COLOR,
                    fontsize=8,
                )
                continue
            text_color = (
                TEXT_COLOR_ON_DARK
                if abs(signed_value) / limit > 0.55
                else TEXT_COLOR
            )
            ax.text(
                col_idx,
                row_idx,
                _format_signed_percent(signed_value, value_format),
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.052, pad=0.035)
    colorbar.set_label("Signed mean Cross-DVI (% points)")

    output_stem = outdir / f"{output_prefix}_feature_heatmap"
    return _save_figure(fig, output_stem)


def _save_figure(fig: plt.Figure, output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def write_figures(
    *,
    result_root: Path = DEFAULT_RESULT_ROOT,
    outdir: Path = DEFAULT_OUTDIR,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
    value_format: str = ".3f",
    heatmap_vmax: float | None = None,
) -> tuple[Path, ...]:
    apply_plot_style()
    frame = load_solver_cross_dvi(result_root)
    overall_summary = aggregate_overall(frame)
    feature_summary = aggregate_by_feature(frame)

    output_paths: list[Path] = []
    output_paths.extend(
        write_summary_csvs(
            overall_summary=overall_summary,
            feature_summary=feature_summary,
            outdir=outdir,
            output_prefix=output_prefix,
        )
    )
    output_paths.extend(
        plot_magnitude_summary(
            overall_summary,
            outdir=outdir,
            output_prefix=output_prefix,
            value_format=value_format,
        )
    )
    output_paths.extend(
        plot_feature_heatmap(
            feature_summary,
            outdir=outdir,
            output_prefix=output_prefix,
            value_format=value_format,
            heatmap_vmax=heatmap_vmax,
        )
    )
    return tuple(output_paths)


def main() -> None:
    args = build_parser().parse_args()
    output_paths = write_figures(
        result_root=args.result_root,
        outdir=args.outdir,
        output_prefix=args.output_prefix,
        value_format=args.value_format,
        heatmap_vmax=args.heatmap_vmax,
    )
    for output_path in output_paths:
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
