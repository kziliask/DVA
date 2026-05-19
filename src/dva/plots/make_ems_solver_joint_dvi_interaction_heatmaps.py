from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from dva.plots.make_ems_joint_dvi_interaction_heatmaps import (
    DEFAULT_OUTPUT_PREFIX as DEFAULT_EMS_OUTPUT_PREFIX,
    DEFAULT_PERCENT_VMAX,
    PERCENT_FLOAT_FORMAT,
    PERCENT_SCALE,
    aggregate_individual_values as aggregate_base_individual_values,
    aggregate_interactions as aggregate_base_interactions,
    apply_plot_style,
    color_limit,
    design_parameters_present,
    load_scenario_frame,
    load_scenario_individual_frame,
    plot_heatmap,
    round_percent_columns,
    scenario_matrix,
)


DEFAULT_RESULT_ROOT = Path("results/ems/experiment_c_solver_dva")
DEFAULT_OUTDIR = Path("data/plots/ems_solver_joint_dvi_interaction_heatmaps")
DEFAULT_OUTPUT_PREFIX = DEFAULT_EMS_OUTPUT_PREFIX.replace(
    "ems_joint_dvi",
    "ems_solver_joint_dvi",
)
SOLVER_SCENARIOS = (
    ("top_left", "exact_vs_greedy", "ante", "Pre (ante)-DVI: Exact vs greedy"),
    ("top_right", "exact_vs_naive", "ante", "Pre (ante)-DVI: Exact vs naive"),
    ("bottom_left", "exact_vs_greedy", "post", "Post-DVI: Exact vs greedy"),
    ("bottom_right", "exact_vs_naive", "post", "Post-DVI: Exact vs naive"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot signed EMS JointDVI feature-solver interaction heatmaps for "
            "exact-vs-greedy and exact-vs-naive solver comparisons."
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
        help="How to aggregate signed hourly interaction values across hours and runs.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=DEFAULT_PERCENT_VMAX,
        help=(
            "Absolute color limit in percentage points. Use 100 for a full "
            "-100% to 100% colorbar."
        ),
    )
    parser.add_argument(
        "--value-format",
        default=".2f",
        help="Format specifier used for cell annotations.",
    )
    return parser


def interaction_paths(
    result_root: Path,
    *,
    solver_comparison: str,
    value_mode: str,
) -> list[Path]:
    pattern = f"xgb_*/{solver_comparison}_{value_mode}/dvi_interactions.csv"
    paths = sorted(result_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No dvi_interactions.csv files found under {result_root!s} "
            f"with pattern {pattern!r}."
        )
    return paths


def load_solver_scenario_frame(
    paths: list[Path],
    *,
    solver_comparison: str,
    value_mode: str,
) -> pd.DataFrame:
    frame = load_scenario_frame(paths, value_mode=value_mode)
    frame["solver_comparison"] = solver_comparison
    return frame.loc[
        :,
        [
            "solver_comparison",
            "value_mode",
            "model_id",
            "timestamp_hour",
            "feature",
            "design_parameter",
            "decision_interaction_value",
        ],
    ]


def load_solver_scenario_individual_frame(
    paths: list[Path],
    *,
    solver_comparison: str,
    value_mode: str,
) -> pd.DataFrame:
    frame = load_scenario_individual_frame(paths, value_mode=value_mode)
    frame["solver_comparison"] = solver_comparison
    return frame.loc[
        :,
        [
            "solver_comparison",
            "value_mode",
            "model_id",
            "timestamp_hour",
            "player",
            "individual_value",
        ],
    ]


def load_interaction_frame(result_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, solver_comparison, value_mode, _ in SOLVER_SCENARIOS:
        frames.append(
            load_solver_scenario_frame(
                interaction_paths(
                    result_root,
                    solver_comparison=solver_comparison,
                    value_mode=value_mode,
                ),
                solver_comparison=solver_comparison,
                value_mode=value_mode,
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_individual_value_frame(result_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, solver_comparison, value_mode, _ in SOLVER_SCENARIOS:
        frames.append(
            load_solver_scenario_individual_frame(
                interaction_paths(
                    result_root,
                    solver_comparison=solver_comparison,
                    value_mode=value_mode,
                ),
                solver_comparison=solver_comparison,
                value_mode=value_mode,
            )
        )
    return pd.concat(frames, ignore_index=True)


def aggregate_interactions(frame: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    return pd.concat(
        [
            aggregate_base_interactions(group, aggregation).assign(
                solver_comparison=solver_comparison
            )
            for solver_comparison, group in frame.groupby("solver_comparison", sort=False)
        ],
        ignore_index=True,
    ).loc[
        :,
        [
            "solver_comparison",
            "value_mode",
            "feature",
            "design_parameter",
            "signed_interaction_value",
        ],
    ]


def aggregate_individual_values(frame: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    return pd.concat(
        [
            aggregate_base_individual_values(group, aggregation).assign(
                solver_comparison=solver_comparison
            )
            for solver_comparison, group in frame.groupby("solver_comparison", sort=False)
        ],
        ignore_index=True,
    ).loc[
        :,
        [
            "solver_comparison",
            "value_mode",
            "player",
            "signed_individual_value",
        ],
    ]


def solver_scenario_matrix(
    interaction_summary: pd.DataFrame,
    individual_summary: pd.DataFrame,
    *,
    solver_comparison: str,
    value_mode: str,
) -> pd.DataFrame:
    scenario_interactions = interaction_summary.loc[
        interaction_summary["solver_comparison"].eq(solver_comparison)
    ]
    scenario_individuals = individual_summary.loc[
        individual_summary["solver_comparison"].eq(solver_comparison)
    ]
    return scenario_matrix(
        scenario_interactions,
        scenario_individuals,
        value_mode=value_mode,
        design_parameters=design_parameters_present(
            scenario_interactions,
            scenario_individuals,
        ),
    )


def write_heatmaps(
    *,
    result_root: Path = DEFAULT_RESULT_ROOT,
    outdir: Path = DEFAULT_OUTDIR,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
    aggregation: str = "mean",
    vmax: float | None = DEFAULT_PERCENT_VMAX,
    value_format: str = ".2f",
) -> tuple[Path, ...]:
    apply_plot_style()
    frame = load_interaction_frame(result_root)
    summary = aggregate_interactions(frame, aggregation)
    individual_frame = load_individual_value_frame(result_root)
    individual_summary = aggregate_individual_values(individual_frame, aggregation)
    summary["signed_interaction_value"] *= PERCENT_SCALE
    individual_summary["signed_individual_value"] *= PERCENT_SCALE

    outdir.mkdir(parents=True, exist_ok=True)
    summary_csv = outdir / f"{output_prefix}_{aggregation}_values.csv"
    round_percent_columns(summary, ("signed_interaction_value",)).to_csv(
        summary_csv,
        index=False,
        float_format=PERCENT_FLOAT_FORMAT,
    )
    individual_summary_csv = outdir / f"{output_prefix}_{aggregation}_individual_values.csv"
    round_percent_columns(individual_summary, ("signed_individual_value",)).to_csv(
        individual_summary_csv,
        index=False,
        float_format=PERCENT_FLOAT_FORMAT,
    )

    limit = color_limit(summary, individual_summary, vmax)
    color_norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    colorbar_label = f"{aggregation.title()} signed DVI (%)\n(interactions + feature individual)"

    output_paths: list[Path] = [summary_csv, individual_summary_csv]
    for position, solver_comparison, value_mode, title in SOLVER_SCENARIOS:
        matrix = solver_scenario_matrix(
            summary,
            individual_summary,
            solver_comparison=solver_comparison,
            value_mode=value_mode,
        )
        output_stem = (
            outdir / f"{output_prefix}_{position}_{solver_comparison}_{value_mode}"
        )
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
