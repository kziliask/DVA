from __future__ import annotations

import argparse
from pathlib import Path

import cmcrameri  # noqa: F401 - registers cmc.* colormaps with Matplotlib.
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RESULT_ROOT = Path("results/caiso/joint_dvi")
DEFAULT_OUTDIR = Path("data/plots/caiso_joint_dvi_interaction_heatmaps")
DEFAULT_OUTPUT_PREFIX = "caiso_joint_dvi_signed_interaction_heatmap"
INTERACTION_CMAP = plt.get_cmap("cmc.vik")
BATTERY_INDIVIDUAL_POSITIVE_COLOR = INTERACTION_CMAP(0.92)
BATTERY_INDIVIDUAL_NEGATIVE_COLOR = INTERACTION_CMAP(0.08)
BATTERY_INDIVIDUAL_ZERO_COLOR = "#e7e7e7"
MISSING_COLOR = "#f2f2f2"
GRID_COLOR = "#ffffff"
SEPARATOR_COLOR = "#565656"
TEXT_COLOR = "#202020"
TEXT_COLOR_ON_DARK = "#ffffff"
INDIVIDUAL_VALUE_KEY = "__individual_value__"
INDIVIDUAL_VALUE_LABEL = "Individual"

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
CAISO_INFO_PLAYERS = tuple(feature for feature, _ in FEATURES)
BATTERY_PARAMETERS = (
    ("energy_capacity", "Energy capacity"),
    ("efficiency", "Efficiency"),
)
DESIGN_PLAYER_NAMES = {parameter for parameter, _ in BATTERY_PARAMETERS}
PLAYER_ORDER = tuple(feature for feature, _ in FEATURES) + tuple(
    parameter for parameter, _ in BATTERY_PARAMETERS
)
SCENARIOS = (
    ("top_left", "conservative", "ante", "Pre (ante)-DVI: Conservative"),
    ("top_right", "optimistic", "ante", "Pre (ante)-DVI: Optimistic"),
    ("bottom_left", "conservative", "post", "Post-DVI: Conservative"),
    ("bottom_right", "optimistic", "post", "Post-DVI: Optimistic"),
)
FONT_CANDIDATES = (
    "Latin Computer Roman",
    "Latin Modern Roman",
    "Computer Modern Roman",
    "CMU Serif",
    "DejaVu Serif",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot signed CAISO JointDVI feature-battery interaction heatmaps "
            "for conservative/optimistic and ante/post scenarios."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Root containing xgb_*/<baseline>_<value-mode>/dvi_interactions.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory where heatmap PNG/PDF files and the summary CSV are written.",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix used for each heatmap filename.",
    )
    parser.add_argument(
        "--aggregation",
        choices=("mean", "median"),
        default="mean",
        help="How to aggregate signed daily interaction values across dates and runs.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help=(
            "Optional absolute color limit. When omitted, the maximum absolute "
            "aggregated interaction across all four heatmaps is used."
        ),
    )
    parser.add_argument(
        "--value-format",
        default=".2f",
        help="Format specifier used for cell annotations.",
    )
    return parser


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
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 320,
            "savefig.dpi": 320,
        }
    )


def interaction_paths(result_root: Path, baseline: str, value_mode: str) -> list[Path]:
    pattern = f"xgb_*/{baseline}_{value_mode}/dvi_interactions.csv"
    paths = sorted(result_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No dvi_interactions.csv files found under {result_root!s} "
            f"with pattern {pattern!r}."
        )
    return paths


def _value_mode_prefix(value_mode: str) -> str:
    if value_mode == "ante":
        return "ead_decision"
    if value_mode == "post":
        return "decision"
    raise ValueError(f"Unknown value_mode: {value_mode}")


def _load_individual_values_from_daily_frame(
    path: Path,
    *,
    value_mode: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise KeyError(f"{path} is missing required column: date")

    if path.name == "daily_dva.csv":
        column_by_player = {player: f"dva_{player}" for player in PLAYER_ORDER}
    else:
        prefix = _value_mode_prefix(value_mode)
        column_by_player = {
            player: f"{prefix}_shap_{player}" for player in PLAYER_ORDER
        }

    missing = set(column_by_player.values()) - set(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {', '.join(sorted(missing))}")

    rows: list[pd.DataFrame] = []
    for player, column in column_by_player.items():
        player_frame = frame.loc[:, ["date", column]].copy()
        player_frame["player"] = player
        player_frame["individual_value"] = pd.to_numeric(
            player_frame[column],
            errors="raise",
        )
        rows.append(player_frame.loc[:, ["date", "player", "individual_value"]])
    return pd.concat(rows, ignore_index=True)


def _load_individual_values_for_run(
    run_dir: Path,
    *,
    value_mode: str,
) -> pd.DataFrame:
    joint_path = run_dir / "joint_dva.csv"
    if joint_path.exists():
        frame = pd.read_csv(joint_path)
        required_columns = {"date", "player", "dva_value"}
        missing = required_columns - set(frame.columns)
        if missing:
            raise KeyError(
                f"{joint_path} is missing columns: {', '.join(sorted(missing))}"
            )
        if "value_mode" in frame.columns:
            frame = frame.loc[frame["value_mode"].astype(str).eq(value_mode)].copy()
        frame = frame.loc[frame["player"].isin(PLAYER_ORDER)].copy()
        frame["individual_value"] = pd.to_numeric(frame["dva_value"], errors="raise")
        return frame.loc[:, ["date", "player", "individual_value"]]

    daily_dva_path = run_dir / "daily_dva.csv"
    if daily_dva_path.exists():
        return _load_individual_values_from_daily_frame(
            daily_dva_path,
            value_mode=value_mode,
        )

    daily_shap_path = run_dir / "daily_shap.csv"
    if daily_shap_path.exists():
        return _load_individual_values_from_daily_frame(
            daily_shap_path,
            value_mode=value_mode,
        )

    raise FileNotFoundError(
        f"Could not find joint_dva.csv, daily_dva.csv, or daily_shap.csv in {run_dir}"
    )


def load_scenario_individual_frame(
    paths: list[Path],
    *,
    baseline: str,
    value_mode: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for interaction_path in paths:
        frame = _load_individual_values_for_run(
            interaction_path.parent,
            value_mode=value_mode,
        )
        frame["date"] = pd.to_datetime(frame["date"])
        frame["model_id"] = interaction_path.parent.parent.name
        frame["baseline"] = baseline
        frame["value_mode"] = value_mode
        frames.append(
            frame.loc[
                :,
                [
                    "baseline",
                    "value_mode",
                    "model_id",
                    "date",
                    "player",
                    "individual_value",
                ],
            ]
        )
    return pd.concat(frames, ignore_index=True)


def load_scenario_frame(
    paths: list[Path],
    *,
    baseline: str,
    value_mode: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required_columns = {
        "date",
        "players",
        "decision_interaction_value",
        "interaction_type",
    }
    info_players = set(CAISO_INFO_PLAYERS)
    design_players = set(DESIGN_PLAYER_NAMES)

    for path in paths:
        frame = pd.read_csv(path)
        missing = required_columns - set(frame.columns)
        if missing:
            raise KeyError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        if "subset_size" in frame.columns:
            frame = frame.loc[frame["subset_size"].eq(2)].copy()
        frame = frame.loc[frame["interaction_type"].eq("Cross-DVI")].copy()

        players = frame["players"].astype(str).str.split("|", n=1, expand=True)
        if players.shape[1] != 2:
            raise ValueError(f"Expected pairwise players in {path}.")
        frame["player_a"] = players[0]
        frame["player_b"] = players[1]

        a_is_feature = frame["player_a"].isin(info_players)
        b_is_feature = frame["player_b"].isin(info_players)
        a_is_design = frame["player_a"].isin(design_players)
        b_is_design = frame["player_b"].isin(design_players)
        valid_cross = (a_is_feature & b_is_design) | (a_is_design & b_is_feature)
        if not valid_cross.all():
            invalid = sorted(frame.loc[~valid_cross, "players"].dropna().unique())
            raise ValueError(
                f"Found Cross-DVI rows that are not feature-parameter pairs in {path}: "
                + ", ".join(invalid[:8])
            )

        frame["feature"] = np.where(a_is_feature, frame["player_a"], frame["player_b"])
        frame["battery_parameter"] = np.where(
            a_is_design,
            frame["player_a"],
            frame["player_b"],
        )
        frame["date"] = pd.to_datetime(frame["date"])
        frame["model_id"] = path.parent.parent.name
        frame["baseline"] = baseline
        frame["value_mode"] = value_mode
        frames.append(
            frame.loc[
                :,
                [
                    "baseline",
                    "value_mode",
                    "model_id",
                    "date",
                    "feature",
                    "battery_parameter",
                    "decision_interaction_value",
                ],
            ]
        )

    return pd.concat(frames, ignore_index=True)


def load_interaction_frame(result_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, baseline, value_mode, _ in SCENARIOS:
        frames.append(
            load_scenario_frame(
                interaction_paths(result_root, baseline, value_mode),
                baseline=baseline,
                value_mode=value_mode,
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_individual_value_frame(result_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, baseline, value_mode, _ in SCENARIOS:
        frames.append(
            load_scenario_individual_frame(
                interaction_paths(result_root, baseline, value_mode),
                baseline=baseline,
                value_mode=value_mode,
            )
        )
    return pd.concat(frames, ignore_index=True)


def aggregate_interactions(frame: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    grouped = frame.groupby(
        ["baseline", "value_mode", "feature", "battery_parameter"],
        sort=False,
    )["decision_interaction_value"]
    if aggregation == "mean":
        summary = grouped.mean()
    elif aggregation == "median":
        summary = grouped.median()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return summary.to_frame(name="signed_interaction_value").reset_index()


def aggregate_individual_values(frame: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    grouped = frame.groupby(
        ["baseline", "value_mode", "player"],
        sort=False,
    )["individual_value"]
    if aggregation == "mean":
        summary = grouped.mean()
    elif aggregation == "median":
        summary = grouped.median()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return summary.to_frame(name="signed_individual_value").reset_index()


def color_limit(
    interaction_summary: pd.DataFrame,
    individual_summary: pd.DataFrame,
    requested_vmax: float | None,
) -> float:
    if requested_vmax is not None:
        if requested_vmax <= 0.0:
            raise ValueError("--vmax must be positive when provided.")
        return requested_vmax

    scaled_individual_summary = individual_summary.loc[
        individual_summary["player"].isin(CAISO_INFO_PLAYERS)
    ]
    values = np.concatenate(
        [
            interaction_summary["signed_interaction_value"].to_numpy(dtype=float),
            scaled_individual_summary["signed_individual_value"].to_numpy(dtype=float),
        ]
    )
    max_abs = float(np.nanmax(np.abs(values)))
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        return 1.0
    return max_abs


def scenario_matrix(
    interaction_summary: pd.DataFrame,
    individual_summary: pd.DataFrame | None = None,
    *,
    baseline: str,
    value_mode: str,
) -> pd.DataFrame:
    scenario_summary = interaction_summary.loc[
        interaction_summary["baseline"].eq(baseline)
        & interaction_summary["value_mode"].eq(value_mode)
    ]
    interaction_matrix = scenario_summary.pivot(
        index="feature",
        columns="battery_parameter",
        values="signed_interaction_value",
    )
    interaction_matrix = interaction_matrix.reindex(
        index=[feature for feature, _ in FEATURES],
        columns=[parameter for parameter, _ in BATTERY_PARAMETERS],
    )
    if individual_summary is None:
        return interaction_matrix

    matrix = pd.DataFrame(
        np.nan,
        index=[feature for feature, _ in FEATURES] + [INDIVIDUAL_VALUE_KEY],
        columns=[
            parameter for parameter, _ in BATTERY_PARAMETERS
        ] + [INDIVIDUAL_VALUE_KEY],
    )
    matrix.loc[interaction_matrix.index, interaction_matrix.columns] = interaction_matrix

    scenario_individual = individual_summary.loc[
        individual_summary["baseline"].eq(baseline)
        & individual_summary["value_mode"].eq(value_mode)
    ]
    player_values = scenario_individual.set_index("player")["signed_individual_value"]
    feature_values = player_values.reindex([feature for feature, _ in FEATURES])
    parameter_values = player_values.reindex(
        [parameter for parameter, _ in BATTERY_PARAMETERS]
    )
    matrix.loc[feature_values.index, INDIVIDUAL_VALUE_KEY] = feature_values
    matrix.loc[INDIVIDUAL_VALUE_KEY, parameter_values.index] = parameter_values
    return matrix


def _column_labels(matrix: pd.DataFrame) -> list[str]:
    parameter_labels = dict(BATTERY_PARAMETERS)
    return [
        INDIVIDUAL_VALUE_LABEL
        if column == INDIVIDUAL_VALUE_KEY
        else parameter_labels.get(str(column), str(column))
        for column in matrix.columns
    ]


def _row_labels(matrix: pd.DataFrame) -> list[str]:
    feature_labels = dict(FEATURES)
    return [
        INDIVIDUAL_VALUE_LABEL
        if row == INDIVIDUAL_VALUE_KEY
        else feature_labels.get(str(row), str(row))
        for row in matrix.index
    ]


def _battery_individual_mask(matrix: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(matrix.shape, dtype=bool)
    if INDIVIDUAL_VALUE_KEY not in matrix.index:
        return mask

    row_idx = matrix.index.get_loc(INDIVIDUAL_VALUE_KEY)
    for parameter, _ in BATTERY_PARAMETERS:
        if parameter in matrix.columns:
            mask[row_idx, matrix.columns.get_loc(parameter)] = True
    return mask


def _battery_individual_color(value: float) -> str | tuple[float, ...]:
    if value > 0.0:
        return BATTERY_INDIVIDUAL_POSITIVE_COLOR
    if value < 0.0:
        return BATTERY_INDIVIDUAL_NEGATIVE_COLOR
    return BATTERY_INDIVIDUAL_ZERO_COLOR


def _draw_fixed_battery_individual_cells(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    values: np.ndarray,
) -> None:
    mask = _battery_individual_mask(matrix)
    for row_idx, col_idx in np.argwhere(mask):
        value = values[row_idx, col_idx]
        if not np.isfinite(value):
            continue
        ax.add_patch(
            Rectangle(
                (col_idx - 0.5, row_idx - 0.5),
                1.0,
                1.0,
                facecolor=_battery_individual_color(float(value)),
                edgecolor=GRID_COLOR,
                linewidth=1.0,
                zorder=2,
            )
        )


def plot_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    color_norm: TwoSlopeNorm,
    colorbar_label: str,
    output_stem: Path,
    value_format: str,
) -> tuple[Path, Path]:
    cmap = INTERACTION_CMAP.copy()
    cmap.set_bad(MISSING_COLOR)

    fig_width = max(4.2, 1.25 * matrix.shape[1] + 1.1)
    fig_height = max(4.8, 0.48 * matrix.shape[0] + 1.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    values = matrix.to_numpy(dtype=float)
    fixed_battery_mask = _battery_individual_mask(matrix)
    display_values = np.ma.masked_where(fixed_battery_mask, values)
    image = ax.imshow(display_values, cmap=cmap, norm=color_norm, aspect="auto")
    _draw_fixed_battery_individual_cells(ax, matrix, values)

    ax.set_title(title, pad=8)
    ax.set_xlabel("Battery parameter")
    ax.set_ylabel("Feature")
    ax.set_xticks(
        np.arange(matrix.shape[1]),
        labels=_column_labels(matrix),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(
        np.arange(matrix.shape[0]),
        labels=_row_labels(matrix),
    )
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1.0), minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    if INDIVIDUAL_VALUE_KEY in matrix.columns:
        ax.axvline(
            matrix.columns.get_loc(INDIVIDUAL_VALUE_KEY) - 0.5,
            color=SEPARATOR_COLOR,
            linewidth=1.1,
        )
    if INDIVIDUAL_VALUE_KEY in matrix.index:
        ax.axhline(
            matrix.index.get_loc(INDIVIDUAL_VALUE_KEY) - 0.5,
            color=SEPARATOR_COLOR,
            linewidth=1.1,
        )
    for spine in ax.spines.values():
        spine.set_visible(False)

    if color_norm.vmin is None or color_norm.vmax is None:
        abs_limit = 1.0
    else:
        abs_limit = max(abs(float(color_norm.vmin)), abs(float(color_norm.vmax)))
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            is_margin_corner = (
                matrix.index[row_idx] == INDIVIDUAL_VALUE_KEY
                and matrix.columns[col_idx] == INDIVIDUAL_VALUE_KEY
            )
            if is_margin_corner:
                continue
            if not np.isfinite(value):
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
            text_color = TEXT_COLOR_ON_DARK if abs(value) / abs_limit > 0.55 else TEXT_COLOR
            if fixed_battery_mask[row_idx, col_idx] and value > 0.0:
                text_color = TEXT_COLOR_ON_DARK
            ax.text(
                col_idx,
                row_idx,
                format(value, value_format),
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.075, pad=0.04)
    colorbar.set_label(colorbar_label)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def write_heatmaps(
    *,
    result_root: Path = DEFAULT_RESULT_ROOT,
    outdir: Path = DEFAULT_OUTDIR,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
    aggregation: str = "mean",
    vmax: float | None = None,
    value_format: str = ".2f",
) -> tuple[Path, ...]:
    apply_plot_style()
    frame = load_interaction_frame(result_root)
    summary = aggregate_interactions(frame, aggregation)
    individual_frame = load_individual_value_frame(result_root)
    individual_summary = aggregate_individual_values(individual_frame, aggregation)

    outdir.mkdir(parents=True, exist_ok=True)
    summary_csv = outdir / f"{output_prefix}_{aggregation}_values.csv"
    summary.to_csv(summary_csv, index=False)
    individual_summary_csv = outdir / f"{output_prefix}_{aggregation}_individual_values.csv"
    individual_summary.to_csv(individual_summary_csv, index=False)

    limit = color_limit(summary, individual_summary, vmax)
    color_norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    colorbar_label = f"{aggregation.title()} signed DVI value\n(interactions + feature individual)"

    output_paths: list[Path] = [summary_csv, individual_summary_csv]
    for position, baseline, value_mode, title in SCENARIOS:
        matrix = scenario_matrix(
            summary,
            individual_summary,
            baseline=baseline,
            value_mode=value_mode,
        )
        output_stem = outdir / f"{output_prefix}_{position}_{baseline}_{value_mode}"
        output_paths.extend(
            plot_heatmap(
                matrix,
                title=title,
                color_norm=color_norm,
                colorbar_label=colorbar_label,
                output_stem=output_stem,
                value_format=value_format,
            )
        )
    return tuple(output_paths)


def main() -> None:
    args = build_parser().parse_args()
    output_paths = write_heatmaps(
        result_root=args.result_root,
        outdir=args.outdir,
        output_prefix=args.output_prefix,
        aggregation=args.aggregation,
        vmax=args.vmax,
        value_format=args.value_format,
    )
    for output_path in output_paths:
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
