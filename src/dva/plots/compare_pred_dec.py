from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmcrameri  # noqa: F401 - registers cmc.* colormaps with Matplotlib.
from matplotlib import font_manager
from matplotlib.colors import Normalize


DEFAULT_DAILY_SHAP_PATH = Path("results/ems_exact_shap_case_study_full/hourly_shap.csv")
DEFAULT_OUTDIR = Path("data/plots/compare_pred_dec_ems")
DEFAULT_UPSET_TOP_N = 20
DEFAULT_WATERFALL_TOP_N = 12

BEESWARM_CMAP = plt.get_cmap("cmc.vik")
FEATURE_VALUE_CLIP_PERCENTILES = (5, 95)
MISSING_FEATURE_COLOR = "#7f7f7f"
POINT_EDGE_COLOR = "#242424"
WATERFALL_REFERENCE_COLOR = "#8a8a8a"
WATERFALL_POSITIVE_COLOR = "#ff2b6d"
WATERFALL_NEGATIVE_COLOR = "#1598ed"
WATERFALL_OTHER_LABEL = "Other features"
METHOD_LABELS = {
    "predictive": "Predictive",
    "decision": "Decision",
    "ead_decision": "EAD-SHAP",
}
CANONICAL_FEATURE_LABELS = {
    "min_temp_c": "Min Temperature",
    "max_temp_c": "Max Temperature",
    "mean_temp_c": "Mean Temperature",
    "mean_humidity": "Mean Humidity",
    "mean_wind_speed": "Mean Wind Speed",
    "mean_solar_irradiance": "Mean Solar Irradiance",
    "max_solar_irradiance": "Max Solar Irradiance",
    "day_of_week": "Day of the Week",
    "throughput_penalty": "Throughput Penalty",
    "efficiency": "Efficiency",
    "energy_capacity": "Energy Capacity",
}


def apply_plot_style() -> None:
    register_latin_modern_fonts()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Latin Modern Roman",
                "Latin Modern Roman 10",
                "CMU Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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


def save_plot_figure(fig: plt.Figure, path: Path, *, dpi: int = 200) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def canonical_feature_label(feature_name: str) -> str:
    if feature_name in CANONICAL_FEATURE_LABELS:
        return CANONICAL_FEATURE_LABELS[feature_name]
    return feature_name.replace("_", " ").title()


def canonical_shap_column_label(column_name: str, method_prefix: str) -> str:
    return canonical_feature_label(column_name.replace(f"{method_prefix}_shap_", ""))


def method_label(method_prefix: str) -> str:
    return METHOD_LABELS.get(method_prefix, method_prefix.replace("_", " ").title())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create predictive-vs-decision SHAP comparison plots.",
    )
    parser.add_argument(
        "--shap-csv",
        "--daily-shap",
        dest="daily_shap",
        type=Path,
        default=DEFAULT_DAILY_SHAP_PATH,
        help="Path to a SHAP CSV, e.g. daily_shap.csv or EMS hourly_shap.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory where plots and summaries will be written.",
    )
    parser.add_argument(
        "--upset",
        action="store_true",
        help=(
            "Also create upset-style plots from interaction CSVs in the same "
            "results directory."
        ),
    )
    parser.add_argument(
        "--upset-top-n",
        type=int,
        default=DEFAULT_UPSET_TOP_N,
        help="Maximum number of interaction subsets to show in each upset plot.",
    )
    parser.add_argument(
        "--feature-focus",
        action="store_true",
        help=(
            "Restrict the regular SHAP plots to dataset features only. For upset "
            "plots, keep only mixed feature-parameter interactions."
        ),
    )
    parser.add_argument(
        "--exclude-parameter-only",
        action="store_true",
        help="For upset plots, drop subsets composed only of parameter players.",
    )
    parser.add_argument(
        "--latin-modern-font",
        action="store_true",
        help="Use Latin Modern Roman for all generated plots.",
    )
    parser.add_argument(
        "--waterfall-date",
        default=None,
        help=(
            "Optional date or timestamp to plot as a single-row predictive and "
            "decision SHAP waterfall, e.g. 2025-12-12."
        ),
    )
    parser.add_argument(
        "--waterfall-top-n",
        type=int,
        default=DEFAULT_WATERFALL_TOP_N,
        help=(
            "Maximum number of individual feature contributions to show in the "
            "single-row waterfall before aggregating the rest."
        ),
    )
    return parser


def _load_run_metadata(results_path: Path) -> dict[str, object]:
    run_metadata_path = results_path.with_name("run_metadata.json")
    with run_metadata_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _resolve_feature_groups(value: object) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    feature_groups: dict[str, list[str]] = {}
    for group_name, group_columns in value.items():
        if isinstance(group_columns, list):
            feature_groups[str(group_name)] = [str(column) for column in group_columns]
    return feature_groups


def _resolve_time_column(frame: pd.DataFrame) -> str:
    for column_name in ("date", "timestamp_hour"):
        if column_name in frame.columns:
            return column_name
    raise KeyError("SHAP CSV must include either a 'date' or 'timestamp_hour' column.")


def _read_shap_frame(shap_path: Path) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(shap_path)
    time_column = _resolve_time_column(frame)
    frame[time_column] = pd.to_datetime(frame[time_column])
    return frame, time_column


def _time_axis_label(time_column: str) -> str:
    return "Hour" if time_column == "timestamp_hour" else "Date"


def _time_grain_label(time_column: str) -> str:
    return "Hourly" if time_column == "timestamp_hour" else "Daily"


def _path_from_metadata(value: object) -> Path:
    path = Path(str(value))
    return path


def _available_usecols(path: Path, requested_columns: Sequence[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns
    header_set = set(header)
    unique_requested_columns = dict.fromkeys(requested_columns)
    return [column for column in unique_requested_columns if column in header_set]


def _load_grouped_feature_values(
    *,
    x_path: Path,
    time_column: str,
    feature_names: Sequence[str],
    feature_groups: dict[str, list[str]],
) -> pd.DataFrame:
    requested_columns = [time_column]
    for feature_name in feature_names:
        requested_columns.extend(feature_groups.get(feature_name, [feature_name]))
    usecols = _available_usecols(x_path, requested_columns)
    if time_column not in usecols:
        raise KeyError(f"{x_path} does not include required time column '{time_column}'.")

    x_frame = pd.read_csv(x_path, usecols=usecols)
    x_frame[time_column] = pd.to_datetime(x_frame[time_column])
    feature_context = pd.DataFrame({time_column: x_frame[time_column]})

    for feature_name in feature_names:
        if feature_name in x_frame.columns:
            feature_context[feature_name] = pd.to_numeric(
                x_frame[feature_name],
                errors="coerce",
            )
            continue

        group_columns = [
            column_name
            for column_name in feature_groups.get(feature_name, [])
            if column_name in x_frame.columns
        ]
        if group_columns:
            feature_context[feature_name] = x_frame.loc[:, group_columns].apply(
                pd.to_numeric,
                errors="coerce",
            ).mean(axis=1)
        else:
            feature_context[feature_name] = np.nan

    return feature_context


def load_feature_value_frame(
    shap_path: Path,
    shap_frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    run_metadata: dict[str, object] | None = None,
    time_column: str | None = None,
) -> pd.DataFrame:
    run_metadata = run_metadata if run_metadata is not None else _load_run_metadata(shap_path)
    time_column = time_column or _resolve_time_column(shap_frame)
    feature_groups = _resolve_feature_groups(run_metadata.get("feature_groups"))
    required_feature_value_names: list[str] = []

    if "x_path" in run_metadata:
        feature_values = _load_grouped_feature_values(
            x_path=_path_from_metadata(run_metadata["x_path"]),
            time_column=time_column,
            feature_names=feature_names,
            feature_groups=feature_groups,
        )
    else:
        dataset_path = _path_from_metadata(run_metadata["dataset_path"])
        dataset_feature_names = _resolve_string_list(run_metadata.get("feature_columns"))
        if not dataset_feature_names:
            dataset_feature_names = list(feature_names)
        required_feature_value_names = dataset_feature_names
        feature_values = pd.read_csv(
            dataset_path,
            usecols=[time_column, *dataset_feature_names],
        )
        feature_values[time_column] = pd.to_datetime(feature_values[time_column])

    merged = shap_frame.merge(
        feature_values,
        on=time_column,
        how="left",
        validate="one_to_one",
    )
    for player_name in feature_names:
        if player_name not in merged.columns:
            # Parameter players and non-tabular grouped players are left uncolored
            # in beeswarm plots rather than failing.
            merged[player_name] = np.nan

    missing_features = [
        name
        for name in required_feature_value_names
        if name in merged.columns and merged[name].isna().all()
    ]
    if missing_features:
        raise ValueError(
            "Missing feature values for beeswarm coloring: "
            + ", ".join(missing_features)
        )
    return merged


def feature_order(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return frame.loc[:, list(columns)].abs().mean().sort_values().index.tolist()


def normalize_feature_values(values: Sequence[float] | np.ndarray) -> np.ndarray:
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


def safe_pearson_correlation(
    left: Sequence[float] | pd.Series,
    right: Sequence[float] | pd.Series,
) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    finite_mask = np.isfinite(left_values) & np.isfinite(right_values)
    if finite_mask.sum() < 2:
        return float("nan")

    left_finite = left_values[finite_mask]
    right_finite = right_values[finite_mask]
    if np.isclose(np.std(left_finite), 0.0) or np.isclose(np.std(right_finite), 0.0):
        return float("nan")
    return float(np.corrcoef(left_finite, right_finite)[0, 1])


def safe_normalized_share(values: pd.Series) -> pd.Series:
    denominator = float(values.sum())
    if np.isclose(denominator, 0.0):
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return values / denominator


def make_beeswarm_like(
    *,
    frame: pd.DataFrame,
    outdir: Path,
    method_prefix: str,
    columns: Sequence[str],
    title: str,
    filename: str,
    order_override: Sequence[str] | None = None,
) -> tuple[Path, list[str]]:
    order = list(order_override) if order_override is not None else feature_order(frame, columns)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    rng = np.random.default_rng(42)
    color_norm = Normalize(vmin=0, vmax=1)

    for index, column in enumerate(order):
        feature_name = column.replace(f"{method_prefix}_shap_", "")
        x_values = frame[column].to_numpy(dtype=float)
        if feature_name in frame.columns:
            color_values = normalize_feature_values(frame[feature_name].to_numpy(dtype=float))
        else:
            color_values = np.full(x_values.shape, np.nan, dtype=float)
        y_values = np.full_like(x_values, index, dtype=float) + rng.uniform(
            -0.22,
            0.22,
            size=len(x_values),
        )
        finite_mask = np.isfinite(color_values)
        if finite_mask.any():
            ax.scatter(
                x_values[finite_mask],
                y_values[finite_mask],
                c=color_values[finite_mask],
                cmap=BEESWARM_CMAP,
                norm=color_norm,
                alpha=0.75,
                s=30,
                edgecolors=POINT_EDGE_COLOR,
                linewidths=0.15,
            )
        if (~finite_mask).any():
            ax.scatter(
                x_values[~finite_mask],
                y_values[~finite_mask],
                color=MISSING_FEATURE_COLOR,
                alpha=0.75,
                s=30,
                edgecolors=POINT_EDGE_COLOR,
                linewidths=0.15,
            )

    scalar_mappable = plt.cm.ScalarMappable(cmap=BEESWARM_CMAP, norm=color_norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02)
    colorbar.set_ticks([0.0, 1.0])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label("Feature value")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [canonical_shap_column_label(column, method_prefix) for column in order]
    )
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    path = outdir / filename
    save_plot_figure(fig, path, dpi=200)
    plt.close(fig)
    return path, order


def _format_value(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:,.0f}"
    if abs_value >= 10:
        return f"{value:,.1f}"
    if abs_value >= 1:
        return f"{value:,.2f}"
    return f"{value:,.3f}"


def _waterfall_timestamp_label(timestamp: pd.Timestamp, time_column: str) -> str:
    if time_column == "date" or timestamp == timestamp.normalize():
        return timestamp.strftime("%Y-%m-%d")
    return timestamp.strftime("%Y-%m-%d %H:%M")


def _waterfall_filename_label(timestamp: pd.Timestamp, time_column: str) -> str:
    label = _waterfall_timestamp_label(timestamp, time_column)
    return label.replace(" ", "_").replace(":", "")


def _target_has_time_component(raw_target: str, target: pd.Timestamp) -> bool:
    if target != target.normalize():
        return True
    return any(token in raw_target for token in ("T", ":", " "))


def select_waterfall_row(
    frame: pd.DataFrame,
    *,
    time_column: str,
    waterfall_date: str,
) -> pd.Series:
    target = pd.Timestamp(waterfall_date)
    target_has_time = _target_has_time_component(waterfall_date, target)
    if target_has_time:
        matches = frame[time_column].eq(target)
    else:
        matches = frame[time_column].dt.normalize().eq(target.normalize())

    match_count = int(matches.sum())
    if match_count == 0:
        available_min = frame[time_column].min()
        available_max = frame[time_column].max()
        raise ValueError(
            f"No row matched waterfall date {waterfall_date!r}. "
            f"Available range is {available_min} to {available_max}."
        )
    if match_count > 1:
        matched_values = (
            frame.loc[matches, time_column]
            .sort_values()
            .dt.strftime("%Y-%m-%d %H:%M:%S")
            .tolist()
        )
        raise ValueError(
            f"Waterfall date {waterfall_date!r} matched {match_count} rows. "
            "Pass a full timestamp instead. Matches: "
            + ", ".join(matched_values[:10])
        )
    return frame.loc[matches].iloc[0]


def _resolve_row_value(
    row: pd.Series,
    candidates: Sequence[str],
    *,
    default: float | None = None,
) -> float:
    for column_name in candidates:
        if column_name in row.index and pd.notna(row[column_name]):
            return float(row[column_name])
    if default is None:
        raise KeyError(
            "Missing required waterfall value column. Tried: "
            + ", ".join(candidates)
        )
    return float(default)


def _waterfall_columns(method_prefix: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if method_prefix == "predictive":
        return ("predictive_baseline_total",), ("predictive_full_total",)
    if method_prefix == "decision":
        return ("decision_baseline_value",), (
            "decision_full_value",
            "decision_full_obj",
        )
    if method_prefix == "ead_decision":
        return ("ead_decision_characteristic_baseline_value",), (
            "ead_decision_characteristic_full_value",
            "ead_decision_value_gain",
        )
    raise ValueError(f"Unsupported waterfall method prefix: {method_prefix}")


def _build_waterfall_contributions(
    *,
    row: pd.Series,
    features: Sequence[str],
    method_prefix: str,
    top_n: int,
    order_override: Sequence[str] | None = None,
) -> tuple[float, float, list[tuple[str, float]]]:
    if top_n <= 0:
        raise ValueError("waterfall_top_n must be strictly positive.")

    shap_values: list[tuple[str, float]] = []
    for feature_name in features:
        column_name = f"{method_prefix}_shap_{feature_name}"
        if column_name in row.index:
            shap_values.append((feature_name, float(row[column_name])))
    if not shap_values:
        raise ValueError(f"No {method_prefix}_shap_* columns available for waterfall.")

    baseline_candidates, full_candidates = _waterfall_columns(method_prefix)
    shap_sum = float(sum(value for _, value in shap_values))
    baseline = _resolve_row_value(row, baseline_candidates, default=0.0)
    full = _resolve_row_value(row, full_candidates, default=baseline + shap_sum)

    if order_override is None:
        ordered = sorted(shap_values, key=lambda item: abs(item[1]), reverse=True)
    else:
        override_rank = {
            feature_name: feature_idx
            for feature_idx, feature_name in enumerate(order_override)
        }
        ordered = sorted(
            shap_values,
            key=lambda item: (
                override_rank.get(item[0], len(override_rank)),
                -abs(item[1]),
            ),
        )
    if len(ordered) > top_n:
        visible = ordered[: max(top_n - 1, 1)]
        hidden = ordered[max(top_n - 1, 1) :]
    else:
        visible = ordered
        hidden = []
    if hidden:
        visible.append(
            (
                f"{len(hidden)} {WATERFALL_OTHER_LABEL.lower()}",
                float(sum(value for _, value in hidden)),
            )
        )

    residual = full - baseline - shap_sum
    tolerance = max(1e-9, 1e-6 * max(1.0, abs(full), abs(baseline), abs(shap_sum)))
    if abs(residual) > tolerance:
        visible.append(("Residual", float(residual)))

    return baseline, full, visible


def _waterfall_cumulative_points(
    *,
    baseline: float,
    contributions: Sequence[tuple[str, float]],
) -> dict[str, tuple[float, float]]:
    cumulative = baseline
    points: dict[str, tuple[float, float]] = {}
    for label, value in reversed(contributions):
        start = cumulative
        end = cumulative + value
        points[label] = (start, end)
        cumulative = end
    return points


def _waterfall_axis_span(
    *,
    baseline: float,
    full: float,
    contribution_points: dict[str, tuple[float, float]],
) -> tuple[float, float, float]:
    values = [baseline, full]
    for start, end in contribution_points.values():
        values.extend((start, end))
    xmin = min(values)
    xmax = max(values)
    span = xmax - xmin
    if np.isclose(span, 0.0):
        span = max(1.0, abs(xmax))
    padding = 0.14 * span
    return xmin - padding, xmax + padding, span


def _format_contribution_value(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{_format_value(value)}"


def _format_feature_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, int, np.floating, np.integer)):
        float_value = float(value)
        if np.isclose(float_value, round(float_value)):
            return f"{float_value:,.0f}"
        return f"{float_value:,.3g}"
    return str(value)


def _waterfall_feature_label(row: pd.Series, feature_name: str) -> str:
    if feature_name in {WATERFALL_OTHER_LABEL, "Residual"}:
        return feature_name
    if feature_name.endswith(WATERFALL_OTHER_LABEL.lower()):
        return feature_name
    label = canonical_feature_label(feature_name)
    if feature_name in row.index and pd.notna(row[feature_name]):
        feature_value = _format_feature_value(row[feature_name])
        if feature_value:
            return f"{feature_value} = {label}"
    return label


def _draw_waterfall_arrow(
    ax: plt.Axes,
    *,
    y_position: float,
    start: float,
    end: float,
    axis_span: float,
    color: str,
    height: float = 0.62,
) -> None:
    dx = end - start
    abs_dx = abs(dx)
    if np.isclose(abs_dx, 0.0):
        ax.plot([start, end], [y_position, y_position], color=color, linewidth=2.0)
        return

    head_length = min(abs_dx * 0.6, max(axis_span * 0.015, abs_dx * 0.25))
    if dx > 0:
        points = [
            (start, y_position - height / 2),
            (end - head_length, y_position - height / 2),
            (end, y_position),
            (end - head_length, y_position + height / 2),
            (start, y_position + height / 2),
        ]
    else:
        points = [
            (start, y_position - height / 2),
            (end + head_length, y_position - height / 2),
            (end, y_position),
            (end + head_length, y_position + height / 2),
            (start, y_position + height / 2),
        ]
    ax.add_patch(plt.Polygon(points, closed=True, color=color, linewidth=0))


def _annotate_contribution(
    ax: plt.Axes,
    *,
    y_position: float,
    start: float,
    end: float,
    value: float,
    axis_span: float,
    color: str,
) -> None:
    label = _format_contribution_value(value)
    width = abs(end - start)
    if width >= 0.12 * axis_span:
        ax.text(
            (start + end) / 2,
            y_position,
            label,
            color="white",
            ha="center",
            va="center",
            fontsize=9,
        )
        return

    offset = 0.012 * axis_span
    if value >= 0:
        x = end + offset
        ha = "left"
    else:
        x = end - offset
        ha = "right"
    ax.text(
        x,
        y_position,
        label,
        color=color,
        ha=ha,
        va="center",
        fontsize=9,
    )


def _draw_waterfall_axis(
    ax: plt.Axes,
    *,
    row: pd.Series,
    baseline: float,
    full: float,
    contributions: Sequence[tuple[str, float]],
    title: str,
    xlabel: str,
) -> None:
    labels = [_waterfall_feature_label(row, name) for name, _ in contributions]
    y_positions = np.arange(len(contributions), dtype=float)
    contribution_points = _waterfall_cumulative_points(
        baseline=baseline,
        contributions=contributions,
    )
    xmin, xmax, axis_span = _waterfall_axis_span(
        baseline=baseline,
        full=full,
        contribution_points=contribution_points,
    )
    ax.set_xlim(xmin, xmax)
    bar_height = 0.62
    for y_position in y_positions:
        ax.axhline(
            y_position,
            color="0.88",
            linewidth=0.7,
            linestyle=(0, (1, 4)),
            zorder=0,
        )

    for y_position, (feature_name, value) in zip(
        y_positions,
        contributions,
        strict=True,
    ):
        start, end = contribution_points[feature_name]
        color = WATERFALL_POSITIVE_COLOR if value >= 0 else WATERFALL_NEGATIVE_COLOR
        _draw_waterfall_arrow(
            ax,
            y_position=y_position,
            start=start,
            end=end,
            axis_span=axis_span,
            color=color,
            height=bar_height,
        )
        _annotate_contribution(
            ax,
            y_position=y_position,
            start=start,
            end=end,
            value=value,
            axis_span=axis_span,
            color=color,
        )
        ax.plot(
            [end, end],
            [y_position - bar_height / 2, y_position + bar_height / 2],
            color="0.72",
            linewidth=0.8,
            linestyle=":",
        )

    ax.axvline(
        full,
        color=WATERFALL_REFERENCE_COLOR,
        linewidth=1.0,
        linestyle=(0, (2, 2)),
    )
    ax.axvline(
        baseline,
        color=WATERFALL_REFERENCE_COLOR,
        linewidth=1.0,
        linestyle=(0, (2, 2)),
    )
    ax.text(
        full,
        1.03,
        f"v(x) = {_format_value(full)}",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=10,
        color="0.2",
    )
    ax.text(
        baseline,
        -0.12,
        f"v(null) = {_format_value(baseline)}",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10,
        color="0.2",
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_ylim(len(contributions) - 0.35, -0.65)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)


def make_single_row_waterfall_plot(
    *,
    frame: pd.DataFrame,
    features: Sequence[str],
    outdir: Path,
    time_column: str,
    waterfall_date: str,
    top_n: int = DEFAULT_WATERFALL_TOP_N,
) -> Path:
    row = select_waterfall_row(
        frame,
        time_column=time_column,
        waterfall_date=waterfall_date,
    )
    timestamp = pd.Timestamp(row[time_column])
    timestamp_label = _waterfall_timestamp_label(timestamp, time_column)
    filename_label = _waterfall_filename_label(timestamp, time_column)

    predictive_baseline, predictive_full, predictive_contributions = (
        _build_waterfall_contributions(
            row=row,
            features=features,
            method_prefix="predictive",
            top_n=top_n,
        )
    )
    predictive_feature_order = [
        feature_name
        for feature_name, _ in predictive_contributions
        if feature_name != "Residual"
        and not feature_name.endswith(WATERFALL_OTHER_LABEL.lower())
    ]
    decision_baseline, decision_full, decision_contributions = (
        _build_waterfall_contributions(
            row=row,
            features=features,
            method_prefix="decision",
            top_n=top_n,
            order_override=predictive_feature_order,
        )
    )
    waterfall_specs = [
        (
            predictive_baseline,
            predictive_full,
            predictive_contributions,
            f"Predictive SHAP waterfall\n{timestamp_label}",
            "Predicted target total",
        ),
        (
            decision_baseline,
            decision_full,
            decision_contributions,
            f"Decision SHAP waterfall\n{timestamp_label}",
            "Decision value",
        ),
    ]
    if any(f"ead_decision_shap_{feature_name}" in row.index for feature_name in features):
        ead_baseline, ead_full, ead_contributions = _build_waterfall_contributions(
            row=row,
            features=features,
            method_prefix="ead_decision",
            top_n=top_n,
            order_override=predictive_feature_order,
        )
        waterfall_specs.append(
            (
                ead_baseline,
                ead_full,
                ead_contributions,
                f"EAD-SHAP waterfall\n{timestamp_label}",
                "Ex ante decision characteristic value",
            )
        )

    row_count = max(
        len(contributions)
        for _, _, contributions, _, _ in waterfall_specs
    ) + 2
    fig_height = max(6.0, 0.42 * row_count + 2.0)
    fig, axes = plt.subplots(
        1,
        len(waterfall_specs),
        figsize=(8.5 * len(waterfall_specs), fig_height),
        squeeze=False,
    )
    for axis, (baseline, full, contributions, title, xlabel) in zip(
        axes[0],
        waterfall_specs,
        strict=True,
    ):
        _draw_waterfall_axis(
            axis,
            row=row,
            baseline=baseline,
            full=full,
            contributions=contributions,
            title=title,
            xlabel=xlabel,
        )
    fig.tight_layout()
    path = outdir / f"waterfall_{filename_label}.png"
    save_plot_figure(fig, path, dpi=220)
    plt.close(fig)
    return path


def make_bar_compare(
    *,
    frame: pd.DataFrame,
    features: Sequence[str],
    predictive_columns: Sequence[str],
    decision_columns: Sequence[str],
    ead_decision_columns: Sequence[str] | None = None,
    outdir: Path,
    filename: str,
) -> tuple[Path, pd.DataFrame]:
    method_specs: list[tuple[str, Sequence[str]]] = [
        ("predictive", predictive_columns),
        ("decision", decision_columns),
    ]
    if ead_decision_columns is not None:
        method_specs.append(("ead_decision", ead_decision_columns))

    compare = pd.DataFrame({"feature": list(features)})
    for method_prefix, columns in method_specs:
        mean_abs_column = f"{method_prefix}_mean_abs_shap"
        share_column = f"{method_prefix}_share"
        compare[mean_abs_column] = frame.loc[:, list(columns)].abs().mean().values
        compare[share_column] = safe_normalized_share(compare[mean_abs_column])
    compare = compare.sort_values("predictive_share")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_positions = np.arange(len(compare))
    bar_height = min(0.7 / len(method_specs), 0.35)
    offset_span = 0.28 if len(method_specs) > 1 else 0.0
    offsets = np.linspace(-offset_span, offset_span, len(method_specs))
    for offset, (method_prefix, _) in zip(offsets, method_specs, strict=True):
        ax.barh(
            y_positions + offset,
            compare[f"{method_prefix}_share"],
            height=bar_height,
            label=method_label(method_prefix),
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(compare["feature"].map(canonical_feature_label))
    ax.set_xlabel("Share of total mean |SHAP|")
    ax.set_ylabel("Feature")
    ax.set_title("Normalized feature importance comparison")
    ax.legend()
    plt.tight_layout()
    path = outdir / filename
    save_plot_figure(fig, path, dpi=200)
    plt.close(fig)
    return path, compare


def make_method_correlation_plot(
    *,
    frame: pd.DataFrame,
    features: Sequence[str],
    outdir: Path,
    filename: str,
    include_ead_decision: bool = False,
) -> tuple[Path, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    for feature_name in features:
        predictive_series = frame[f"predictive_shap_{feature_name}"]
        decision_series = frame[f"decision_shap_{feature_name}"]
        row: dict[str, float | str] = {
            "feature": feature_name,
            "predictive_decision_corr": safe_pearson_correlation(
                predictive_series,
                decision_series,
            ),
        }
        if include_ead_decision:
            ead_decision_series = frame[f"ead_decision_shap_{feature_name}"]
            row["predictive_ead_decision_corr"] = safe_pearson_correlation(
                predictive_series,
                ead_decision_series,
            )
            row["decision_ead_decision_corr"] = safe_pearson_correlation(
                decision_series,
                ead_decision_series,
            )
        rows.append(row)
    correlation_frame = pd.DataFrame(rows).sort_values("predictive_decision_corr")

    fig, ax = plt.subplots(figsize=(10, 6))
    y_positions = np.arange(len(correlation_frame))
    if include_ead_decision:
        specs = [
            ("predictive_decision_corr", "Predictive vs Decision"),
            ("predictive_ead_decision_corr", "Predictive vs EAD-SHAP"),
            ("decision_ead_decision_corr", "Decision vs EAD-SHAP"),
        ]
        bar_height = 0.22
        offsets = np.linspace(-0.25, 0.25, len(specs))
        for offset, (column_name, label) in zip(offsets, specs, strict=True):
            ax.barh(
                y_positions + offset,
                correlation_frame[column_name],
                height=bar_height,
                label=label,
            )
        ax.legend()
    else:
        ax.barh(
            y_positions,
            correlation_frame["predictive_decision_corr"],
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(correlation_frame["feature"].map(canonical_feature_label))
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Feature")
    ax.set_title("Feature-wise SHAP correlation between methods")
    ax.axvline(0, linewidth=1)
    plt.tight_layout()
    path = outdir / filename
    save_plot_figure(fig, path, dpi=200)
    plt.close(fig)
    return path, correlation_frame


def make_rank_agreement_plot(
    *,
    frame: pd.DataFrame,
    predictive_columns: Sequence[str],
    decision_columns: Sequence[str],
    outdir: Path,
    filename: str,
    time_column: str,
    use_cached_metric: bool = True,
) -> Path:
    if use_cached_metric and "abs_rank_spearman" in frame.columns:
        series = frame["abs_rank_spearman"]
        ylabel = "Rank Spearman"
        title = f"{_time_grain_label(time_column)} rank agreement between methods"
    else:
        rankings: list[float] = []
        for _, row in frame.iterrows():
            predictive_ranks = pd.Series(np.abs(row[list(predictive_columns)].to_numpy())).rank()
            decision_ranks = pd.Series(np.abs(row[list(decision_columns)].to_numpy())).rank()
            rankings.append(float(predictive_ranks.corr(decision_ranks, method="spearman")))
        series = pd.Series(rankings, index=frame.index)
        ylabel = "Rank Spearman"
        title = f"{_time_grain_label(time_column)} rank agreement between methods"

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(frame[time_column], series)
    ax.axhline(float(series.mean()), linewidth=1)
    ax.set_xlabel(_time_axis_label(time_column))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = outdir / filename
    save_plot_figure(fig, path, dpi=200)
    plt.close(fig)
    return path


def make_gain_comparison_plot(
    *,
    frame: pd.DataFrame,
    outdir: Path,
    filename: str,
    time_column: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(frame[time_column], frame["predictive_total_gain"], label="Predictive total gain")
    ax.plot(frame[time_column], frame["decision_value_gain"], label="Decision value gain")
    if "ead_decision_value_gain" in frame.columns:
        ax.plot(
            frame[time_column],
            frame["ead_decision_value_gain"],
            label="EAD-SHAP value gain",
        )
    ax.set_xlabel(_time_axis_label(time_column))
    ax.set_ylabel("Gain")
    ax.set_title(f"{_time_grain_label(time_column)} method output gain")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = outdir / filename
    save_plot_figure(fig, path, dpi=200)
    plt.close(fig)
    return path


def make_signed_mean_plot(
    *,
    frame: pd.DataFrame,
    features: Sequence[str],
    outdir: Path,
    filename: str,
    include_ead_decision: bool = False,
) -> tuple[Path, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    for feature_name in features:
        row: dict[str, float | str] = {
            "feature": feature_name,
            "predictive_mean_shap": float(
                frame[f"predictive_shap_{feature_name}"].mean()
            ),
            "decision_mean_shap": float(frame[f"decision_shap_{feature_name}"].mean()),
        }
        if include_ead_decision:
            row["ead_decision_mean_shap"] = float(
                frame[f"ead_decision_shap_{feature_name}"].mean()
            )
        rows.append(row)
    means = pd.DataFrame(rows).sort_values("predictive_mean_shap")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_positions = np.arange(len(means))
    specs = [
        ("predictive_mean_shap", "Predictive"),
        ("decision_mean_shap", "Decision"),
    ]
    if include_ead_decision:
        specs.append(("ead_decision_mean_shap", "EAD-SHAP"))
    bar_height = min(0.7 / len(specs), 0.35)
    offset_span = 0.28 if len(specs) > 1 else 0.0
    offsets = np.linspace(-offset_span, offset_span, len(specs))
    for offset, (column_name, label) in zip(offsets, specs, strict=True):
        ax.barh(
            y_positions + offset,
            means[column_name],
            height=bar_height,
            label=label,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(means["feature"].map(canonical_feature_label))
    ax.set_xlabel("Mean SHAP value")
    ax.set_ylabel("Feature")
    ax.set_title("Average directional contribution by feature")
    ax.axvline(0, linewidth=1)
    ax.legend()
    plt.tight_layout()
    path = outdir / filename
    save_plot_figure(fig, path, dpi=200)
    plt.close(fig)
    return path, means


def make_method_summary(
    *,
    frame: pd.DataFrame,
    features: Sequence[str],
    outdir: Path,
    filename: str,
    include_ead_decision: bool = False,
) -> Path:
    summary_rows: list[dict[str, float | str]] = []
    for feature_name in features:
        predictive_series = frame[f"predictive_shap_{feature_name}"]
        decision_series = frame[f"decision_shap_{feature_name}"]
        row: dict[str, float | str] = {
            "feature": feature_name,
            "predictive_mean_shap": float(predictive_series.mean()),
            "predictive_mean_abs_shap": float(predictive_series.abs().mean()),
            "predictive_std_shap": float(predictive_series.std()),
            "predictive_positive_fraction": float((predictive_series > 0).mean()),
            "decision_mean_shap": float(decision_series.mean()),
            "decision_mean_abs_shap": float(decision_series.abs().mean()),
            "decision_std_shap": float(decision_series.std()),
            "decision_positive_fraction": float((decision_series > 0).mean()),
            "predictive_decision_corr": safe_pearson_correlation(
                predictive_series,
                decision_series,
            ),
        }
        if include_ead_decision:
            ead_decision_series = frame[f"ead_decision_shap_{feature_name}"]
            row.update(
                {
                    "ead_decision_mean_shap": float(ead_decision_series.mean()),
                    "ead_decision_mean_abs_shap": float(
                        ead_decision_series.abs().mean()
                    ),
                    "ead_decision_std_shap": float(ead_decision_series.std()),
                    "ead_decision_positive_fraction": float(
                        (ead_decision_series > 0).mean()
                    ),
                    "predictive_ead_decision_corr": safe_pearson_correlation(
                        predictive_series,
                        ead_decision_series,
                    ),
                    "decision_ead_decision_corr": safe_pearson_correlation(
                        decision_series,
                        ead_decision_series,
                    ),
                }
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        "predictive_mean_abs_shap",
        ascending=False,
    )
    summary_path = outdir / filename
    summary.to_csv(summary_path, index=False)
    return summary_path


def _resolve_upset_player_order(
    interaction_path: Path,
    subset_players: Sequence[Sequence[str]],
) -> list[str]:
    present_players = {
        player_name
        for player_list in subset_players
        for player_name in player_list
    }
    metadata_player_names = _resolve_string_list(
        _load_run_metadata(interaction_path).get("player_names")
    )
    if metadata_player_names:
        return [
            player_name
            for player_name in metadata_player_names
            if player_name in present_players
        ]
    return sorted(present_players)


def make_interaction_upset_plot(
    *,
    interaction_path: Path,
    outdir: Path,
    value_column: str,
    title: str,
    filename: str,
    xlabel: str,
    top_n: int = DEFAULT_UPSET_TOP_N,
    feature_focus: bool = False,
    feature_names: Sequence[str] = (),
    parameter_names: Sequence[str] = (),
    exclude_parameter_only: bool = False,
) -> Path:
    if top_n <= 0:
        raise ValueError("top_n must be strictly positive.")
    if not interaction_path.exists():
        raise FileNotFoundError(
            f"Missing interaction CSV at {interaction_path}. "
            "Run the case study with --interaction-order first."
        )

    frame = pd.read_csv(interaction_path)
    required_columns = {"players", "subset_size", value_column}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise KeyError(
            f"{interaction_path.name} is missing required columns: "
            + ", ".join(missing_columns)
        )
    if frame.empty:
        raise ValueError(f"{interaction_path.name} is empty.")

    frame = frame.copy()
    frame["players"] = frame["players"].astype(str)
    frame["subset_size"] = frame["subset_size"].astype(int)
    frame[value_column] = pd.to_numeric(frame[value_column], errors="raise")
    frame["abs_value"] = frame[value_column].abs()

    plot_frame = frame.copy()
    plot_frame["player_list"] = plot_frame["players"].str.split("|")
    parameter_name_set = set(parameter_names)

    if feature_focus:
        feature_name_set = set(feature_names)
        if not feature_name_set:
            raise ValueError(
                "feature_focus requires feature names in run_metadata.feature_columns."
            )
        if not parameter_name_set:
            raise ValueError(
                "feature_focus requires parameter players, but none were found in run_metadata.player_names."
            )
        plot_frame = plot_frame.loc[
            plot_frame["player_list"].map(
                lambda player_list: (
                    all(player_name in feature_name_set for player_name in player_list)
                    or (
                        any(player_name in feature_name_set for player_name in player_list)
                        and any(player_name in parameter_name_set for player_name in player_list)
                    )
                )
            )
        ].copy()
        if plot_frame.empty:
            raise ValueError(
                "No eligible feature or mixed feature-parameter terms remain after applying --feature-focus."
            )
    elif exclude_parameter_only and parameter_name_set:
        plot_frame = plot_frame.loc[
            ~plot_frame["player_list"].map(
                lambda player_list: all(
                    player_name in parameter_name_set for player_name in player_list
                )
            )
        ].copy()
        if plot_frame.empty:
            raise ValueError(
                "No eligible terms remain after excluding parameter-only subsets."
            )

    grouped = (
        plot_frame.groupby(["players", "subset_size"], as_index=False)
        .agg(
            mean_abs_value=("abs_value", "mean"),
            mean_signed_value=(value_column, "mean"),
            observations=(value_column, "size"),
        )
        .sort_values(
            ["mean_abs_value", "subset_size", "players"],
            ascending=[False, False, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )
    grouped["player_list"] = grouped["players"].str.split("|")

    player_order = _resolve_upset_player_order(
        interaction_path,
        grouped["player_list"].tolist(),
    )
    if not player_order:
        raise ValueError(
            f"No player memberships found in {interaction_path.name}."
        )

    membership = np.zeros((len(player_order), len(grouped)), dtype=bool)
    player_position = {
        player_name: position
        for position, player_name in enumerate(player_order)
    }
    for column_idx, player_list in enumerate(grouped["player_list"]):
        for player_name in player_list:
            membership[player_position[player_name], column_idx] = True

    figure_width = max(10.0, 0.6 * len(grouped) + 4.0)
    figure_height = max(6.0, 0.45 * len(player_order) + 4.0)
    fig = plt.figure(figsize=(figure_width, figure_height))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.2], hspace=0.05)
    ax_bar = fig.add_subplot(grid[0])
    ax_matrix = fig.add_subplot(grid[1], sharex=ax_bar)

    x_positions = np.arange(len(grouped))
    subset_colors = plt.get_cmap("Blues")(
        np.linspace(0.45, 0.9, max(2, int(grouped["subset_size"].nunique())))
    )
    color_map = {
        subset_size: subset_colors[index]
        for index, subset_size in enumerate(sorted(grouped["subset_size"].unique()))
    }
    ax_bar.bar(
        x_positions,
        grouped["mean_abs_value"],
        color=[color_map[int(size)] for size in grouped["subset_size"]],
        width=0.8,
    )
    ax_bar.set_ylabel(f"Mean |{value_column}|")
    ax_bar.set_title(title)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(axis="x", labelbottom=False)

    for column_idx in x_positions:
        active_rows = np.flatnonzero(membership[:, column_idx])
        if active_rows.size > 1:
            ax_matrix.plot(
                [column_idx, column_idx],
                [active_rows.min(), active_rows.max()],
                color="0.35",
                linewidth=1.4,
                zorder=1,
            )
        ax_matrix.scatter(
            np.full(len(player_order), column_idx, dtype=float),
            np.arange(len(player_order)),
            s=22,
            color="0.85",
            zorder=2,
        )
        ax_matrix.scatter(
            np.full(active_rows.size, column_idx, dtype=float),
            active_rows,
            s=48,
            color="0.15",
            zorder=3,
        )

    ax_matrix.set_yticks(np.arange(len(player_order)))
    ax_matrix.set_yticklabels([canonical_feature_label(player) for player in player_order])
    ax_matrix.invert_yaxis()
    ax_matrix.set_xlabel(xlabel)
    ax_matrix.set_ylabel("Players")
    ax_matrix.set_xticks(x_positions)
    ax_matrix.set_xticklabels([""] * len(grouped))
    ax_matrix.spines["top"].set_visible(False)
    ax_matrix.spines["right"].set_visible(False)
    ax_matrix.spines["bottom"].set_visible(False)

    fig.tight_layout()
    path = outdir / filename
    save_plot_figure(fig, path, dpi=200)
    plt.close(fig)
    return path


def make_shapley_taylor_upset_plot(
    *,
    shapley_taylor_path: Path,
    outdir: Path,
    value_column: str,
    title: str,
    filename: str,
    top_n: int = DEFAULT_UPSET_TOP_N,
    feature_focus: bool = False,
    feature_names: Sequence[str] = (),
    parameter_names: Sequence[str] = (),
    exclude_parameter_only: bool = False,
) -> Path:
    return make_interaction_upset_plot(
        interaction_path=shapley_taylor_path,
        outdir=outdir,
        value_column=value_column,
        title=title,
        filename=filename,
        xlabel="Top Shapley-Taylor subsets",
        top_n=top_n,
        feature_focus=feature_focus,
        feature_names=feature_names,
        parameter_names=parameter_names,
        exclude_parameter_only=exclude_parameter_only,
    )


def create_comparison_plots(
    *,
    daily_shap_path: Path,
    outdir: Path,
    upset: bool = False,
    upset_top_n: int = DEFAULT_UPSET_TOP_N,
    feature_focus: bool = False,
    exclude_parameter_only: bool = False,
    waterfall_date: str | None = None,
    waterfall_top_n: int = DEFAULT_WATERFALL_TOP_N,
    latin_modern_font: bool = False,
) -> list[Path]:
    if latin_modern_font:
        apply_plot_style()
    outdir.mkdir(parents=True, exist_ok=True)

    daily_shap_frame, time_column = _read_shap_frame(daily_shap_path)
    run_metadata = _load_run_metadata(daily_shap_path)
    all_predictive_columns = [
        column_name
        for column_name in daily_shap_frame.columns
        if column_name.startswith("predictive_shap_")
    ]
    all_ead_decision_columns = [
        column_name
        for column_name in daily_shap_frame.columns
        if column_name.startswith("ead_decision_shap_")
    ]
    all_players = [
        column_name.replace("predictive_shap_", "")
        for column_name in all_predictive_columns
    ]
    metadata_feature_names = _resolve_string_list(run_metadata.get("feature_columns"))
    metadata_player_names = _resolve_string_list(run_metadata.get("player_names"))
    metadata_feature_groups = _resolve_feature_groups(run_metadata.get("feature_groups"))
    grouped_feature_names = list(metadata_feature_groups)
    focus_feature_names = grouped_feature_names or metadata_feature_names
    if feature_focus and focus_feature_names:
        features = [
            feature_name
            for feature_name in focus_feature_names
            if f"predictive_shap_{feature_name}" in daily_shap_frame.columns
            and f"decision_shap_{feature_name}" in daily_shap_frame.columns
        ]
    else:
        features = all_players
    if not features:
        raise ValueError("No SHAP feature columns available for plotting.")
    predictive_columns = [f"predictive_shap_{feature_name}" for feature_name in features]
    decision_columns = [f"decision_shap_{feature_name}" for feature_name in features]
    ead_decision_columns = [
        f"ead_decision_shap_{feature_name}"
        for feature_name in features
        if f"ead_decision_shap_{feature_name}" in daily_shap_frame.columns
    ]
    has_ead_decision = bool(all_ead_decision_columns) and len(
        ead_decision_columns
    ) == len(features)
    regret_predictive_columns = [
        f"regret_predictive_shap_{feature_name}"
        for feature_name in features
        if f"regret_predictive_shap_{feature_name}" in daily_shap_frame.columns
    ]
    parameter_names = [
        player_name
        for player_name in metadata_player_names
        if player_name not in set(focus_feature_names)
    ]

    merged_frame = load_feature_value_frame(
        daily_shap_path,
        daily_shap_frame,
        features,
        run_metadata=run_metadata,
        time_column=time_column,
    )

    created_paths: list[Path] = []
    predictive_beeswarm, predictive_order = make_beeswarm_like(
        frame=merged_frame,
        outdir=outdir,
        method_prefix="predictive",
        columns=predictive_columns,
        title="Predictive SHAP beeswarm",
        filename="predictive_beeswarm.png",
    )
    created_paths.append(predictive_beeswarm)
    if regret_predictive_columns:
        regret_predictive_order = [
            column_name.replace("predictive_shap_", "regret_predictive_shap_")
            for column_name in predictive_order
            if column_name.replace("predictive_shap_", "regret_predictive_shap_")
            in regret_predictive_columns
        ]
        regret_predictive_beeswarm, _ = make_beeswarm_like(
            frame=merged_frame,
            outdir=outdir,
            method_prefix="regret_predictive",
            columns=regret_predictive_columns,
            title="Regret predictive SHAP beeswarm",
            filename="regret_predictive_beeswarm.png",
            order_override=regret_predictive_order,
        )
        created_paths.append(regret_predictive_beeswarm)

    decision_order = [
        column_name.replace("predictive_shap_", "decision_shap_")
        for column_name in predictive_order
    ]
    decision_beeswarm, _ = make_beeswarm_like(
        frame=merged_frame,
        outdir=outdir,
        method_prefix="decision",
        columns=decision_columns,
        title="Decision SHAP beeswarm",
        filename="decision_beeswarm.png",
        order_override=decision_order,
    )
    created_paths.append(decision_beeswarm)
    if has_ead_decision:
        ead_decision_order = [
            column_name.replace("predictive_shap_", "ead_decision_shap_")
            for column_name in predictive_order
        ]
        ead_decision_beeswarm, _ = make_beeswarm_like(
            frame=merged_frame,
            outdir=outdir,
            method_prefix="ead_decision",
            columns=ead_decision_columns,
            title="EAD-SHAP beeswarm",
            filename="ead_decision_beeswarm.png",
            order_override=ead_decision_order,
        )
        created_paths.append(ead_decision_beeswarm)

    normalized_bar, _ = make_bar_compare(
        frame=merged_frame,
        features=features,
        predictive_columns=predictive_columns,
        decision_columns=decision_columns,
        ead_decision_columns=ead_decision_columns if has_ead_decision else None,
        outdir=outdir,
        filename="normalized_importance_comparison.png",
    )
    created_paths.append(normalized_bar)

    method_correlation, _ = make_method_correlation_plot(
        frame=merged_frame,
        features=features,
        outdir=outdir,
        filename="featurewise_method_correlation.png",
        include_ead_decision=has_ead_decision,
    )
    created_paths.append(method_correlation)

    created_paths.append(
        make_rank_agreement_plot(
            frame=merged_frame,
            predictive_columns=predictive_columns,
            decision_columns=decision_columns,
            outdir=outdir,
            filename="daily_rank_agreement.png",
            time_column=time_column,
            use_cached_metric=not feature_focus,
        )
    )
    if has_ead_decision:
        created_paths.append(
            make_rank_agreement_plot(
                frame=merged_frame,
                predictive_columns=predictive_columns,
                decision_columns=ead_decision_columns,
                outdir=outdir,
                filename="daily_rank_agreement_predictive_ead.png",
                time_column=time_column,
                use_cached_metric=False,
            )
        )
        created_paths.append(
            make_rank_agreement_plot(
                frame=merged_frame,
                predictive_columns=decision_columns,
                decision_columns=ead_decision_columns,
                outdir=outdir,
                filename="daily_rank_agreement_decision_ead.png",
                time_column=time_column,
                use_cached_metric=False,
            )
        )
    created_paths.append(
        make_gain_comparison_plot(
            frame=merged_frame,
            outdir=outdir,
            filename="gain_over_time.png",
            time_column=time_column,
        )
    )

    signed_mean_plot, _ = make_signed_mean_plot(
        frame=merged_frame,
        features=features,
        outdir=outdir,
        filename="signed_mean_shap_comparison.png",
        include_ead_decision=has_ead_decision,
    )
    created_paths.append(signed_mean_plot)

    created_paths.append(
        make_method_summary(
            frame=merged_frame,
            features=features,
            outdir=outdir,
            filename="shap_method_summary.csv",
            include_ead_decision=has_ead_decision,
        )
    )

    if waterfall_date is not None:
        created_paths.append(
            make_single_row_waterfall_plot(
                frame=merged_frame,
                features=features,
                outdir=outdir,
                time_column=time_column,
                waterfall_date=waterfall_date,
                top_n=waterfall_top_n,
            )
        )

    if upset:
        results_dir = daily_shap_path.parent
        interaction_method = str(run_metadata.get("interaction_method", "shapley_taylor"))
        interaction_label = {
            "faith_shap": "Faith-SHAP",
            "shapley_taylor": "Shapley-Taylor",
        }.get(interaction_method, interaction_method)
        generic_decision_path = results_dir / "daily_interaction_decision.csv"
        generic_predictive_path = results_dir / "daily_interaction_predictive.csv"
        if generic_decision_path.exists() and generic_predictive_path.exists():
            decision_interaction_path = generic_decision_path
            predictive_interaction_path = generic_predictive_path
            decision_value_column = "decision_interaction_value"
            predictive_value_column = "predictive_interaction_value"
        else:
            decision_interaction_path = results_dir / "daily_shapley_taylor_decision.csv"
            predictive_interaction_path = results_dir / "daily_shapley_taylor_predictive.csv"
            decision_value_column = "decision_shapley_taylor"
            predictive_value_column = "predictive_shapley_taylor"

        created_paths.append(
            make_interaction_upset_plot(
                interaction_path=decision_interaction_path,
                outdir=outdir,
                value_column=decision_value_column,
                title=f"Decision {interaction_label} upset plot",
                filename=f"decision_{interaction_method}_upset.png",
                xlabel=f"Top {interaction_label} subsets",
                top_n=upset_top_n,
                feature_focus=feature_focus,
                feature_names=focus_feature_names,
                parameter_names=parameter_names,
                exclude_parameter_only=exclude_parameter_only,
            )
        )
        created_paths.append(
            make_interaction_upset_plot(
                interaction_path=predictive_interaction_path,
                outdir=outdir,
                value_column=predictive_value_column,
                title=f"Predictive {interaction_label} upset plot",
                filename=f"predictive_{interaction_method}_upset.png",
                xlabel=f"Top {interaction_label} subsets",
                top_n=upset_top_n,
                feature_focus=feature_focus,
                feature_names=focus_feature_names,
                parameter_names=parameter_names,
                exclude_parameter_only=exclude_parameter_only,
            )
        )

    return created_paths


def main() -> None:
    args = build_parser().parse_args()
    created_paths = create_comparison_plots(
        daily_shap_path=args.daily_shap,
        outdir=args.outdir,
        upset=args.upset,
        upset_top_n=args.upset_top_n,
        feature_focus=args.feature_focus,
        exclude_parameter_only=args.exclude_parameter_only,
        waterfall_date=args.waterfall_date,
        waterfall_top_n=args.waterfall_top_n,
        latin_modern_font=args.latin_modern_font,
    )
    print("Created files:")
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()
