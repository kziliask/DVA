from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cmcrameri  # noqa: F401 - registers cmc.* colormaps with Matplotlib.
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INFODVA_ROOT = Path("results/ems/experiment_a_infodva")
DEFAULT_JOINT_DVI_ROOT = Path("results/ems/experiment_b_joint_dvi")
DEFAULT_DESIGN_UTILITY_ROOT = Path("results/ems/experiment_d_design_utility")
DEFAULT_OUTDIR = Path("data/plots/ems_decision_regimes")
DEFAULT_OUTPUT_STEM = "ems_decision_regimes_heatmaps"
DEFAULT_CMAP = "YlGnBu"
DEFAULT_COVERAGE_RADII_KM = (1.0, 2.0, 3.0)
DEFAULT_FACILITY_BUDGETS = (3, 5, 8)
DEFAULT_NONZERO_TOLERANCE = 1e-12
DEFAULT_TOP_JOINT_DVI_TERMS = 8
GRID_COLOR = "#ffffff"
TEXT_COLOR = "#202020"
TEXT_COLOR_ON_DARK = "#ffffff"

INFO_PLAYERS = {
    "hour",
    "day_of_week",
    "temp_c",
    "precip_mm",
    "citywide_ems_incidents_lag_1",
    "ems_incidents_lag_1",
    "neighbor_ems_incidents_lag_1_mean",
    "zone_hour_baseline",
}
DESIGN_PLAYERS = {"radius_km", "staging_areas"}
FEATURE_LABELS = {
    "hour": "Hour",
    "day_of_week": "DOW",
    "temp_c": "Temp",
    "precip_mm": "Precip",
    "citywide_ems_incidents_lag_1": "City lag",
    "ems_incidents_lag_1": "EMS lag",
    "neighbor_ems_incidents_lag_1_mean": "N lag",
    "zone_hour_baseline": "ZH",
}
DESIGN_LABELS = {
    "radius_km": "radius",
    "staging_areas": "budget",
}
DETAILED_FEATURE_LABELS = {
    "hour": "time-of-day",
    "day_of_week": "day-of-week",
    "temp_c": "temperature",
    "precip_mm": "precipitation",
    "citywide_ems_incidents_lag_1": "citywide demand",
    "ems_incidents_lag_1": "lagged demand",
    "neighbor_ems_incidents_lag_1_mean": "neighboring demand",
    "zone_hour_baseline": "ZH",
}
DETAILED_DESIGN_LABELS = {
    "radius_km": "radius",
    "staging_areas": "budget",
}
REGIME_LABELS = {
    (1.0, 3): "active",
    (1.0, 5): "active",
    (1.0, 8): "active",
    (2.0, 3): "stable",
    (2.0, 5): "active",
    (2.0, 8): "sat.",
    (3.0, 3): "sat.",
    (3.0, 5): "sat.",
    (3.0, 8): "sat.",
}
FONT_CANDIDATES = (
    "Latin Computer Roman",
    "Latin Modern Roman",
    "Computer Modern Roman",
    "CMU Serif",
    "DejaVu Serif",
)


@dataclass(frozen=True, slots=True)
class DecisionRegimeOutputs:
    png: Path
    pdf: Path
    policy_change_png: Path
    policy_change_pdf: Path
    top_terms_png: Path
    top_terms_pdf: Path
    top_terms_csv: Path
    csv: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a three-panel EMS heatmap figure for coverage saturation, "
            "decision activity, and design-conditioned information value."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--infodva-root",
        type=Path,
        default=DEFAULT_INFODVA_ROOT,
        help="Root containing experiment_a_infodva xgb_*/models/*/runs outputs.",
    )
    parser.add_argument(
        "--joint-dvi-root",
        type=Path,
        default=DEFAULT_JOINT_DVI_ROOT,
        help="Root containing experiment_b_joint_dvi xgb_*/.../joint_summary_dva.csv.",
    )
    parser.add_argument(
        "--design-utility-root",
        type=Path,
        default=DEFAULT_DESIGN_UTILITY_ROOT,
        help="Root containing experiment_d_design_utility xgb_*/configuration_summary.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory where the figure and summary CSV are written.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Filename stem used for the PNG, PDF, and CSV outputs.",
    )
    parser.add_argument(
        "--coverage-radii-km",
        type=float,
        nargs="+",
        default=DEFAULT_COVERAGE_RADII_KM,
        help="Coverage-radius grid to show as heatmap rows.",
    )
    parser.add_argument(
        "--facility-budgets",
        type=int,
        nargs="+",
        default=DEFAULT_FACILITY_BUDGETS,
        help="Facility-budget grid to show as heatmap columns.",
    )
    parser.add_argument(
        "--cmap",
        default=DEFAULT_CMAP,
        help="Registered Matplotlib colormap name for all three panels.",
    )
    parser.add_argument(
        "--nonzero-tolerance",
        type=float,
        default=DEFAULT_NONZERO_TOLERANCE,
        help="Absolute decision-characteristic threshold used for Panel B activity.",
    )
    parser.add_argument(
        "--panel-c-value",
        choices=("top-term", "joint-info", "cross-dvi"),
        default="top-term",
        help=(
            "Cell value for Panel C. 'top-term' uses the largest individual "
            "post-JointDVI main effect or info-design interaction; 'joint-info' "
            "uses total post-JointDVI information main-effect value; 'cross-dvi' "
            "uses total absolute information-design Cross-DVI intensity."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-grid",
        action="store_true",
        help="Allow missing grid cells instead of failing.",
    )
    parser.add_argument(
        "--top-joint-dvi-terms",
        type=int,
        default=DEFAULT_TOP_JOINT_DVI_TERMS,
        help="Number of post-JointDVI terms to show in each detailed regime panel.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = write_ems_decision_regime_heatmaps(
        infodva_root=args.infodva_root,
        joint_dvi_root=args.joint_dvi_root,
        design_utility_root=args.design_utility_root,
        outdir=args.outdir,
        output_stem=args.output_stem,
        coverage_radii_km=tuple(args.coverage_radii_km),
        facility_budgets=tuple(args.facility_budgets),
        cmap_name=args.cmap,
        nonzero_tolerance=args.nonzero_tolerance,
        panel_c_value=args.panel_c_value,
        allow_incomplete_grid=args.allow_incomplete_grid,
        top_joint_dvi_terms=args.top_joint_dvi_terms,
    )
    print(f"Wrote EMS decision-regime PNG to {outputs.png}")
    print(f"Wrote EMS decision-regime PDF to {outputs.pdf}")
    print(f"Wrote EMS coalition policy-change PNG to {outputs.policy_change_png}")
    print(f"Wrote EMS coalition policy-change PDF to {outputs.policy_change_pdf}")
    print(f"Wrote EMS top JointDVI terms PNG to {outputs.top_terms_png}")
    print(f"Wrote EMS top JointDVI terms PDF to {outputs.top_terms_pdf}")
    print(f"Wrote EMS top JointDVI terms CSV to {outputs.top_terms_csv}")
    print(f"Wrote EMS decision-regime summary CSV to {outputs.csv}")


def write_ems_decision_regime_heatmaps(
    *,
    infodva_root: Path = DEFAULT_INFODVA_ROOT,
    joint_dvi_root: Path = DEFAULT_JOINT_DVI_ROOT,
    design_utility_root: Path = DEFAULT_DESIGN_UTILITY_ROOT,
    outdir: Path = DEFAULT_OUTDIR,
    output_stem: str = DEFAULT_OUTPUT_STEM,
    coverage_radii_km: Sequence[float] = DEFAULT_COVERAGE_RADII_KM,
    facility_budgets: Sequence[int] = DEFAULT_FACILITY_BUDGETS,
    cmap_name: str = DEFAULT_CMAP,
    nonzero_tolerance: float = DEFAULT_NONZERO_TOLERANCE,
    panel_c_value: str = "top-term",
    allow_incomplete_grid: bool = False,
    top_joint_dvi_terms: int = DEFAULT_TOP_JOINT_DVI_TERMS,
) -> DecisionRegimeOutputs:
    outdir.mkdir(parents=True, exist_ok=True)
    coverage_radii = tuple(float(value) for value in coverage_radii_km)
    budgets = tuple(int(value) for value in facility_budgets)

    coverage = load_coverage_summary(design_utility_root)
    activity = load_decision_activity_summary(
        infodva_root,
        nonzero_tolerance=nonzero_tolerance,
    )
    joint_dvi = load_joint_dvi_summary(joint_dvi_root)
    joint_dvi_terms = load_post_joint_dvi_terms(joint_dvi_root)
    top_joint_dvi_terms_summary = summarize_top_post_joint_dvi_terms(joint_dvi_terms)
    summary = merge_regime_summaries(
        coverage=coverage,
        activity=activity,
        joint_dvi=joint_dvi,
        top_joint_dvi_terms=top_joint_dvi_terms_summary,
        coverage_radii_km=coverage_radii,
        facility_budgets=budgets,
        allow_incomplete_grid=allow_incomplete_grid,
    )

    csv_path = outdir / f"{output_stem}.csv"
    summary.to_csv(csv_path, index=False)

    top_terms_csv_path = outdir / f"{output_stem}_top_post_joint_dvi_terms.csv"
    joint_dvi_terms.to_csv(top_terms_csv_path, index=False)

    png_path = outdir / f"{output_stem}.png"
    pdf_path = outdir / f"{output_stem}.pdf"
    plot_decision_regime_heatmaps(
        summary=summary,
        coverage_radii_km=coverage_radii,
        facility_budgets=budgets,
        output_paths=(png_path, pdf_path),
        cmap_name=cmap_name,
        panel_c_value=panel_c_value,
    )

    policy_png_path = outdir / f"{output_stem}_coalition_policy_change_rate.png"
    policy_pdf_path = outdir / f"{output_stem}_coalition_policy_change_rate.pdf"
    plot_coalition_policy_change_heatmap(
        summary=summary,
        coverage_radii_km=coverage_radii,
        facility_budgets=budgets,
        output_paths=(policy_png_path, policy_pdf_path),
        cmap_name=cmap_name,
    )

    top_terms_png_path = outdir / f"{output_stem}_top_post_joint_dvi_terms.png"
    top_terms_pdf_path = outdir / f"{output_stem}_top_post_joint_dvi_terms.pdf"
    plot_top_post_joint_dvi_terms(
        terms=joint_dvi_terms,
        coverage_radii_km=coverage_radii,
        facility_budgets=budgets,
        output_paths=(top_terms_png_path, top_terms_pdf_path),
        cmap_name=cmap_name,
        top_n=top_joint_dvi_terms,
    )

    return DecisionRegimeOutputs(
        png=png_path,
        pdf=pdf_path,
        policy_change_png=policy_png_path,
        policy_change_pdf=policy_pdf_path,
        top_terms_png=top_terms_png_path,
        top_terms_pdf=top_terms_pdf_path,
        top_terms_csv=top_terms_csv_path,
        csv=csv_path,
    )


def load_coverage_summary(design_utility_root: Path) -> pd.DataFrame:
    paths = sorted(design_utility_root.glob("xgb_*/configuration_summary.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No configuration_summary.csv files found under {design_utility_root}"
        )

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame.loc[frame["coverage_solver"].astype(str).eq("exact")].copy()
        frame["model_id"] = path.parent.name
        frames.append(frame)
    records = pd.concat(frames, ignore_index=True)
    records["coverage_radius_km"] = records["coverage_radius_km"].astype(float)
    records["facility_budget"] = records["facility_budget"].astype(int)
    records["uncovered_demand"] = (
        records["mean_actual_total_demand"].astype(float)
        - records["mean_realized_covered_demand"].astype(float)
    )

    return (
        records.groupby(["coverage_radius_km", "facility_budget"], as_index=False)
        .agg(
            coverage_run_count=("model_id", "nunique"),
            mean_realized_coverage=("mean_realized_coverage", "mean"),
            mean_realized_covered_demand=("mean_realized_covered_demand", "mean"),
            mean_actual_total_demand=("mean_actual_total_demand", "mean"),
            mean_uncovered_demand=("uncovered_demand", "mean"),
        )
        .sort_values(["coverage_radius_km", "facility_budget"])
    )


def load_decision_activity_summary(
    infodva_root: Path,
    *,
    nonzero_tolerance: float,
) -> pd.DataFrame:
    paths = sorted(infodva_root.glob("xgb_*/models/*/runs/*/coalition_values.csv"))
    if not paths:
        raise FileNotFoundError(f"No coalition_values.csv files found under {infodva_root}")

    run_records = []
    for path in paths:
        run_dir = path.parent
        metadata = _read_json(run_dir / "run_metadata.json")
        if str(metadata.get("coverage_solver", "")) != "exact":
            continue
        frame = pd.read_csv(
            path,
            usecols=[
                "timestamp_hour",
                "coalition_mask",
                "decision_characteristic_value",
                "decision_selected_facility_indices",
            ],
        )
        values = frame["decision_characteristic_value"].astype(float).abs()
        run_records.append(
            {
                "coverage_radius_km": float(metadata["coverage_radius_km"]),
                "facility_budget": int(metadata["facility_budget"]),
                "model_id": str(metadata.get("model_id", run_dir.parent.parent.name)),
                "decision_characteristic_nonzero_rate": float(
                    (values > nonzero_tolerance).mean()
                ),
                "coalition_policy_change_rate": compute_coalition_policy_change_rate(
                    frame
                ),
                "mean_abs_decision_characteristic": float(values.mean()),
                "max_abs_decision_characteristic": float(values.max()),
            }
        )
    if not run_records:
        raise ValueError(f"No exact-solver EMS InfoDVA runs found under {infodva_root}")

    records = pd.DataFrame.from_records(run_records)
    return (
        records.groupby(["coverage_radius_km", "facility_budget"], as_index=False)
        .agg(
            activity_run_count=("model_id", "nunique"),
            decision_characteristic_nonzero_rate=(
                "decision_characteristic_nonzero_rate",
                "mean",
            ),
            coalition_policy_change_rate=("coalition_policy_change_rate", "mean"),
            mean_abs_decision_characteristic=(
                "mean_abs_decision_characteristic",
                "mean",
            ),
            max_abs_decision_characteristic=(
                "max_abs_decision_characteristic",
                "mean",
            ),
        )
        .sort_values(["coverage_radius_km", "facility_budget"])
    )


def load_joint_dvi_summary(joint_dvi_root: Path) -> pd.DataFrame:
    metadata_paths = sorted(joint_dvi_root.glob("xgb_*/**/design_dva_metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(
            f"No design_dva_metadata.json files found under {joint_dvi_root}"
        )

    info_records = []
    interaction_records = []
    for metadata_path in metadata_paths:
        metadata = _read_json(metadata_path)
        if str(metadata.get("value_mode", "")) != "post":
            continue
        actual_design = metadata.get("actual_design", {})
        if str(actual_design.get("solver", "")) != "exact":
            continue
        run_dir = metadata_path.parent
        model_id = str(metadata["model_id"])
        coverage_radius_km = float(actual_design["radius_km"])
        facility_budget = int(actual_design["staging_areas"])

        joint_summary_path = run_dir / "joint_summary_dva.csv"
        if not joint_summary_path.exists():
            continue
        joint_summary = pd.read_csv(joint_summary_path)
        info_summary = joint_summary.loc[joint_summary["player_kind"].astype(str).eq("info")]
        if info_summary.empty:
            continue
        info_summary = info_summary.copy()
        info_summary["dva_mean_abs"] = info_summary["dva_mean_abs"].astype(float)
        top_info = info_summary.sort_values("dva_mean_abs", ascending=False).iloc[0]
        info_records.append(
            {
                "coverage_radius_km": coverage_radius_km,
                "facility_budget": facility_budget,
                "model_id": model_id,
                "total_post_joint_info_value": float(info_summary["dva_mean_abs"].sum()),
                "top_joint_info_feature": str(top_info["player"]),
                "top_joint_info_value": float(top_info["dva_mean_abs"]),
            }
        )

        interaction_path = run_dir / "dvi_interactions.csv"
        if not interaction_path.exists():
            continue
        interactions = pd.read_csv(
            interaction_path,
            usecols=["players", "interaction_type", "decision_interaction_value"],
        )
        interactions = interactions.loc[
            interactions["interaction_type"].astype(str).eq("Cross-DVI")
        ].copy()
        if interactions.empty:
            continue
        pairs = interactions["players"].astype(str).str.split("|", n=1, expand=True)
        interactions["left_player"] = pairs[0]
        interactions["right_player"] = pairs[1]
        interactions["info_player"] = interactions.apply(
            lambda row: _info_player_from_pair(row["left_player"], row["right_player"]),
            axis=1,
        )
        interactions["design_player"] = interactions.apply(
            lambda row: _design_player_from_pair(row["left_player"], row["right_player"]),
            axis=1,
        )
        interactions = interactions.loc[
            interactions["info_player"].notna()
            & interactions["design_player"].isin(DESIGN_PLAYERS)
        ].copy()
        if interactions.empty:
            continue
        interactions["abs_interaction"] = interactions[
            "decision_interaction_value"
        ].astype(float).abs()
        pair_summary = (
            interactions.groupby(["info_player", "design_player"], as_index=False)
            .agg(mean_abs_cross_dvi=("abs_interaction", "mean"))
            .sort_values(["info_player", "design_player"])
        )
        for row in pair_summary.itertuples(index=False):
            interaction_records.append(
                {
                    "coverage_radius_km": coverage_radius_km,
                    "facility_budget": facility_budget,
                    "model_id": model_id,
                    "info_player": str(row.info_player),
                    "design_player": str(row.design_player),
                    "mean_abs_cross_dvi": float(row.mean_abs_cross_dvi),
                }
            )

    if not info_records:
        raise ValueError(f"No post JointDVI summaries found under {joint_dvi_root}")

    info_frame = pd.DataFrame.from_records(info_records)
    info_agg = (
        info_frame.groupby(["coverage_radius_km", "facility_budget"], as_index=False)
        .agg(
            joint_dvi_run_count=("model_id", "nunique"),
            total_post_joint_info_value=("total_post_joint_info_value", "mean"),
            top_joint_info_value=("top_joint_info_value", "mean"),
        )
        .sort_values(["coverage_radius_km", "facility_budget"])
    )

    top_info = (
        info_frame.groupby(
            ["coverage_radius_km", "facility_budget", "top_joint_info_feature"],
            as_index=False,
        )
        .agg(mean_top_joint_info_value=("top_joint_info_value", "mean"))
        .sort_values("mean_top_joint_info_value", ascending=False)
        .drop_duplicates(["coverage_radius_km", "facility_budget"])
        .loc[:, ["coverage_radius_km", "facility_budget", "top_joint_info_feature"]]
    )
    info_agg = info_agg.merge(
        top_info,
        on=["coverage_radius_km", "facility_budget"],
        how="left",
    )

    if not interaction_records:
        info_agg["cross_dvi_intensity"] = 0.0
        info_agg["top_cross_dvi_interaction"] = ""
        info_agg["top_cross_dvi_value"] = 0.0
        return info_agg

    interaction_frame = pd.DataFrame.from_records(interaction_records)
    pair_agg = (
        interaction_frame.groupby(
            ["coverage_radius_km", "facility_budget", "info_player", "design_player"],
            as_index=False,
        )
        .agg(mean_abs_cross_dvi=("mean_abs_cross_dvi", "mean"))
    )
    cross_total = (
        pair_agg.groupby(["coverage_radius_km", "facility_budget"], as_index=False)
        .agg(cross_dvi_intensity=("mean_abs_cross_dvi", "sum"))
    )
    top_cross = (
        pair_agg.sort_values("mean_abs_cross_dvi", ascending=False)
        .drop_duplicates(["coverage_radius_km", "facility_budget"])
        .rename(columns={"mean_abs_cross_dvi": "top_cross_dvi_value"})
        .loc[
            :,
            [
                "coverage_radius_km",
                "facility_budget",
                "info_player",
                "design_player",
                "top_cross_dvi_value",
            ],
        ]
    )
    top_cross["top_cross_dvi_interaction"] = top_cross.apply(
        lambda row: format_interaction_label(row["info_player"], row["design_player"]),
        axis=1,
    )
    top_cross = top_cross.drop(columns=["info_player", "design_player"])

    return (
        info_agg.merge(cross_total, on=["coverage_radius_km", "facility_budget"], how="left")
        .merge(top_cross, on=["coverage_radius_km", "facility_budget"], how="left")
        .assign(
            cross_dvi_intensity=lambda frame: frame["cross_dvi_intensity"].fillna(0.0),
            top_cross_dvi_value=lambda frame: frame["top_cross_dvi_value"].fillna(0.0),
            top_cross_dvi_interaction=lambda frame: frame[
                "top_cross_dvi_interaction"
            ].fillna(""),
        )
    )


def load_post_joint_dvi_terms(joint_dvi_root: Path) -> pd.DataFrame:
    metadata_paths = sorted(joint_dvi_root.glob("xgb_*/**/design_dva_metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(
            f"No design_dva_metadata.json files found under {joint_dvi_root}"
        )

    term_records = []
    for metadata_path in metadata_paths:
        metadata = _read_json(metadata_path)
        if str(metadata.get("value_mode", "")) != "post":
            continue
        actual_design = metadata.get("actual_design", {})
        if str(actual_design.get("solver", "")) != "exact":
            continue

        run_dir = metadata_path.parent
        model_id = str(metadata["model_id"])
        coverage_radius_km = float(actual_design["radius_km"])
        facility_budget = int(actual_design["staging_areas"])

        joint_summary_path = run_dir / "joint_summary_dva.csv"
        if joint_summary_path.exists():
            joint_summary = pd.read_csv(
                joint_summary_path,
                usecols=["player", "player_kind", "dva_mean_abs"],
            )
            info_summary = joint_summary.loc[
                joint_summary["player_kind"].astype(str).eq("info")
            ].copy()
            info_summary["dva_mean_abs"] = info_summary["dva_mean_abs"].astype(float)
            for row in info_summary.itertuples(index=False):
                info_player = str(row.player)
                term_records.append(
                    {
                        "coverage_radius_km": coverage_radius_km,
                        "facility_budget": facility_budget,
                        "model_id": model_id,
                        "term_kind": "info_main",
                        "term_key": f"main:{info_player}",
                        "term_label": format_post_joint_term_label(info_player),
                        "mean_abs_post_joint_dvi": float(row.dva_mean_abs),
                    }
                )

        interaction_path = run_dir / "dvi_interactions.csv"
        if not interaction_path.exists():
            continue
        interactions = pd.read_csv(
            interaction_path,
            usecols=["players", "interaction_type", "decision_interaction_value"],
        )
        interactions = interactions.loc[
            interactions["interaction_type"].astype(str).eq("Cross-DVI")
        ].copy()
        if interactions.empty:
            continue
        pairs = interactions["players"].astype(str).str.split("|", n=1, expand=True)
        interactions["left_player"] = pairs[0]
        interactions["right_player"] = pairs[1]
        interactions["info_player"] = interactions.apply(
            lambda row: _info_player_from_pair(row["left_player"], row["right_player"]),
            axis=1,
        )
        interactions["design_player"] = interactions.apply(
            lambda row: _design_player_from_pair(row["left_player"], row["right_player"]),
            axis=1,
        )
        interactions = interactions.loc[
            interactions["info_player"].notna()
            & interactions["design_player"].isin(DESIGN_PLAYERS)
        ].copy()
        if interactions.empty:
            continue
        interactions["abs_interaction"] = interactions[
            "decision_interaction_value"
        ].astype(float).abs()
        pair_summary = (
            interactions.groupby(["info_player", "design_player"], as_index=False)
            .agg(mean_abs_post_joint_dvi=("abs_interaction", "mean"))
            .sort_values(["info_player", "design_player"])
        )
        for row in pair_summary.itertuples(index=False):
            info_player = str(row.info_player)
            design_player = str(row.design_player)
            term_records.append(
                {
                    "coverage_radius_km": coverage_radius_km,
                    "facility_budget": facility_budget,
                    "model_id": model_id,
                    "term_kind": "info_design_interaction",
                    "term_key": f"cross:{info_player}:{design_player}",
                    "term_label": format_post_joint_term_label(
                        info_player,
                        design_player=design_player,
                    ),
                    "mean_abs_post_joint_dvi": float(row.mean_abs_post_joint_dvi),
                }
            )

    if not term_records:
        raise ValueError(f"No post JointDVI term records found under {joint_dvi_root}")

    terms = pd.DataFrame.from_records(term_records)
    term_summary = (
        terms.groupby(
            [
                "coverage_radius_km",
                "facility_budget",
                "term_kind",
                "term_key",
                "term_label",
            ],
            as_index=False,
        )
        .agg(
            joint_dvi_run_count=("model_id", "nunique"),
            mean_abs_post_joint_dvi=("mean_abs_post_joint_dvi", "mean"),
        )
        .sort_values(
            [
                "coverage_radius_km",
                "facility_budget",
                "mean_abs_post_joint_dvi",
                "term_label",
            ],
            ascending=[True, True, False, True],
        )
    )
    term_summary["value_pct_points"] = (
        100.0 * term_summary["mean_abs_post_joint_dvi"].astype(float)
    )
    return term_summary.reset_index(drop=True)


def summarize_top_post_joint_dvi_terms(terms: pd.DataFrame) -> pd.DataFrame:
    ordered = terms.sort_values(
        [
            "coverage_radius_km",
            "facility_budget",
            "mean_abs_post_joint_dvi",
            "term_label",
        ],
        ascending=[True, True, False, True],
    )
    top_terms = (
        ordered.drop_duplicates(["coverage_radius_km", "facility_budget"])
        .rename(
            columns={
                "term_kind": "top_post_joint_dvi_term_kind",
                "term_key": "top_post_joint_dvi_term_key",
                "term_label": "top_post_joint_dvi_term_label",
                "mean_abs_post_joint_dvi": "top_post_joint_dvi_value",
                "value_pct_points": "top_post_joint_dvi_pct_points",
            }
        )
        .loc[
            :,
            [
                "coverage_radius_km",
                "facility_budget",
                "top_post_joint_dvi_term_kind",
                "top_post_joint_dvi_term_key",
                "top_post_joint_dvi_term_label",
                "top_post_joint_dvi_value",
                "top_post_joint_dvi_pct_points",
            ],
        ]
    )
    return top_terms.reset_index(drop=True)


def merge_regime_summaries(
    *,
    coverage: pd.DataFrame,
    activity: pd.DataFrame,
    joint_dvi: pd.DataFrame,
    top_joint_dvi_terms: pd.DataFrame,
    coverage_radii_km: Sequence[float],
    facility_budgets: Sequence[int],
    allow_incomplete_grid: bool,
) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product(
        [tuple(float(value) for value in coverage_radii_km), tuple(int(value) for value in facility_budgets)],
        names=["coverage_radius_km", "facility_budget"],
    ).to_frame(index=False)
    summary = (
        grid.merge(coverage, on=["coverage_radius_km", "facility_budget"], how="left")
        .merge(activity, on=["coverage_radius_km", "facility_budget"], how="left")
        .merge(joint_dvi, on=["coverage_radius_km", "facility_budget"], how="left")
        .merge(
            top_joint_dvi_terms,
            on=["coverage_radius_km", "facility_budget"],
            how="left",
        )
    )
    required = [
        "mean_uncovered_demand",
        "decision_characteristic_nonzero_rate",
        "coalition_policy_change_rate",
        "total_post_joint_info_value",
        "top_post_joint_dvi_value",
    ]
    missing = summary.loc[summary[required].isna().any(axis=1)]
    if not missing.empty and not allow_incomplete_grid:
        missing_cells = ", ".join(
            f"tau={row.coverage_radius_km:g}, p={int(row.facility_budget)}"
            for row in missing.itertuples(index=False)
        )
        raise ValueError(
            "Missing required EMS decision-regime cells: "
            f"{missing_cells}. Pass --allow-incomplete-grid to plot available cells."
        )

    summary["cross_dvi_intensity"] = summary["cross_dvi_intensity"].fillna(0.0)
    summary["top_cross_dvi_value"] = summary["top_cross_dvi_value"].fillna(0.0)
    summary["top_cross_dvi_interaction"] = summary["top_cross_dvi_interaction"].fillna("")
    summary["top_post_joint_dvi_value"] = summary["top_post_joint_dvi_value"].fillna(0.0)
    summary["top_post_joint_dvi_pct_points"] = summary[
        "top_post_joint_dvi_pct_points"
    ].fillna(0.0)
    summary["top_post_joint_dvi_term_label"] = summary[
        "top_post_joint_dvi_term_label"
    ].fillna("")
    summary["regime_label"] = summary.apply(
        lambda row: REGIME_LABELS.get(
            (float(row.coverage_radius_km), int(row.facility_budget)),
            "",
        ),
        axis=1,
    )
    summary["panel_c_top_annotation"] = summary.apply(_panel_c_top_annotation, axis=1)
    summary["panel_c_main_annotation"] = summary.apply(_panel_c_main_annotation, axis=1)
    return summary.sort_values(["coverage_radius_km", "facility_budget"]).reset_index(
        drop=True
    )


def plot_decision_regime_heatmaps(
    *,
    summary: pd.DataFrame,
    coverage_radii_km: Sequence[float],
    facility_budgets: Sequence[int],
    output_paths: tuple[Path, Path],
    cmap_name: str,
    panel_c_value: str,
) -> None:
    apply_plot_style()
    radii = tuple(float(value) for value in coverage_radii_km)
    budgets = tuple(int(value) for value in facility_budgets)

    panel_c_column, panel_c_label, panel_c_annotation_column, panel_c_title = (
        panel_c_plot_spec(panel_c_value)
    )
    panel_c_values = 100.0 * _matrix(summary, radii, budgets, panel_c_column)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.75), constrained_layout=False)

    draw_heatmap_panel(
        ax=axes[0],
        values=100.0 * _matrix(summary, radii, budgets, "mean_realized_coverage"),
        annotations=_annotation_matrix(
            summary,
            radii,
            budgets,
            lambda row: f"{100.0 * row.mean_realized_coverage:.1f}%",
        ),
        title="A. Actual Realized Coverage",
        colorbar_label="Mean covered demand (%)",
        cmap_name=cmap_name,
        radii=radii,
        budgets=budgets,
        ylabel=True,
    )
    draw_heatmap_panel(
        ax=axes[1],
        values=_matrix(summary, radii, budgets, "decision_characteristic_nonzero_rate"),
        annotations=_annotation_matrix(
            summary,
            radii,
            budgets,
            lambda row: (
                f"{100.0 * row.decision_characteristic_nonzero_rate:.1f}%\n"
                f"{row.regime_label}"
            ),
        ),
        title="B. DVA Coalition Activity",
        colorbar_label="Active InfoDVA coalitions (%)",
        cmap_name=cmap_name,
        radii=radii,
        budgets=budgets,
        value_is_fraction=True,
    )
    draw_heatmap_panel(
        ax=axes[2],
        values=panel_c_values,
        annotations=_annotation_matrix(
            summary,
            radii,
            budgets,
            lambda row: (
                f"{100.0 * getattr(row, panel_c_column):.1f}%\n"
                f"{getattr(row, panel_c_annotation_column)}"
            ),
        ),
        title=panel_c_title,
        colorbar_label=f"{panel_c_label} (%)",
        cmap_name=cmap_name,
        radii=radii,
        budgets=budgets,
    )

    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.17, wspace=0.38)
    for output_path in output_paths:
        if output_path.suffix.lower() == ".png":
            fig.savefig(output_path, bbox_inches="tight", dpi=320)
        else:
            fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_coalition_policy_change_heatmap(
    *,
    summary: pd.DataFrame,
    coverage_radii_km: Sequence[float],
    facility_budgets: Sequence[int],
    output_paths: tuple[Path, Path],
    cmap_name: str,
) -> None:
    apply_plot_style()
    radii = tuple(float(value) for value in coverage_radii_km)
    budgets = tuple(int(value) for value in facility_budgets)

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.75), constrained_layout=False)
    draw_heatmap_panel(
        ax=ax,
        values=_matrix(summary, radii, budgets, "coalition_policy_change_rate"),
        annotations=_annotation_matrix(
            summary,
            radii,
            budgets,
            lambda row: f"{100.0 * row.coalition_policy_change_rate:.1f}%",
        ),
        title="Coalition Policy Change Rate",
        colorbar_label="Coalition policy change rate (%)",
        cmap_name=cmap_name,
        radii=radii,
        budgets=budgets,
        ylabel=True,
        value_is_fraction=True,
    )

    fig.subplots_adjust(left=0.17, right=0.92, top=0.84, bottom=0.17)
    for output_path in output_paths:
        if output_path.suffix.lower() == ".png":
            fig.savefig(output_path, bbox_inches="tight", dpi=320)
        else:
            fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_top_post_joint_dvi_terms(
    *,
    terms: pd.DataFrame,
    coverage_radii_km: Sequence[float],
    facility_budgets: Sequence[int],
    output_paths: tuple[Path, Path],
    cmap_name: str,
    top_n: int,
) -> None:
    apply_plot_style()
    radii = tuple(float(value) for value in coverage_radii_km)
    budgets = tuple(int(value) for value in facility_budgets)
    terms_per_panel = max(1, int(top_n))

    finite_values = terms["value_pct_points"].dropna().astype(float)
    xmax = float(finite_values.max()) if not finite_values.empty else 1.0
    if xmax <= 0.0:
        xmax = 1.0
    xlim = 1.22 * xmax
    label_offset = 0.018 * xlim
    cmap = plt.get_cmap(cmap_name)

    fig, axes = plt.subplots(
        len(radii),
        len(budgets),
        figsize=(16.8, 11.2),
        sharex=True,
        squeeze=False,
    )

    for row_idx, radius in enumerate(radii):
        for col_idx, budget in enumerate(budgets):
            ax = axes[row_idx, col_idx]
            all_panel_terms = terms.loc[
                np.isclose(terms["coverage_radius_km"], radius)
                & terms["facility_budget"].eq(int(budget))
            ].copy()
            main_effect_total = all_panel_terms.loc[
                all_panel_terms["term_kind"].eq("info_main"),
                "value_pct_points",
            ].sum()
            interaction_total = all_panel_terms.loc[
                all_panel_terms["term_kind"].eq("info_design_interaction"),
                "value_pct_points",
            ].sum()
            panel_terms = all_panel_terms.sort_values(
                ["mean_abs_post_joint_dvi", "term_label"],
                ascending=[False, True],
            ).head(terms_per_panel)

            ax.set_title(
                format_design_regime_title(radius, budget),
                loc="left",
                fontsize=11.0,
                fontweight="bold",
                pad=18,
            )
            ax.text(
                0.0,
                1.02,
                (
                    f"Total main effect: {main_effect_total:.1f}%, "
                    f"total interaction: {interaction_total:.1f}%"
                ),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.9,
                color=TEXT_COLOR,
            )
            ax.set_xlim(0.0, xlim)
            ax.grid(axis="x", color="#e5e5e5", linewidth=0.75)
            ax.tick_params(axis="x", labelsize=8.8)
            ax.tick_params(axis="y", length=0, labelsize=8.6)
            for spine_name in ("top", "right", "left"):
                ax.spines[spine_name].set_visible(False)
            ax.spines["bottom"].set_color("#bdbdbd")

            if panel_terms.empty:
                ax.set_yticks([])
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#666666",
                    fontsize=9.0,
                )
                continue

            values = panel_terms["value_pct_points"].to_numpy(dtype=float)
            y_positions = np.arange(len(panel_terms))
            colors = [
                cmap(0.32 + 0.58 * (float(value) / xmax if xmax else 0.0))
                for value in values
            ]
            ax.barh(
                y_positions,
                values,
                color=colors,
                edgecolor=GRID_COLOR,
                linewidth=0.55,
            )
            ax.set_yticks(y_positions)
            ax.set_yticklabels(panel_terms["term_label"].to_list(), fontsize=8.2)
            ax.invert_yaxis()

            for y_position, value in zip(y_positions, values, strict=True):
                ax.text(
                    min(value + label_offset, 0.985 * xlim),
                    y_position,
                    f"{value:.2f}",
                    ha="left",
                    va="center",
                    fontsize=8.0,
                    color=TEXT_COLOR,
                )

    fig.suptitle(
        "Top post-JointDVI terms by EMS design regime",
        fontsize=15.0,
        y=0.985,
    )
    fig.supxlabel(
        "Absolute post-JointDVI value (percentage points of realized coverage)",
        fontsize=11.2,
        y=0.03,
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.985,
        top=0.9,
        bottom=0.075,
        hspace=0.82,
        wspace=0.68,
    )
    for output_path in output_paths:
        if output_path.suffix.lower() == ".png":
            fig.savefig(output_path, bbox_inches="tight", dpi=320)
        else:
            fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_heatmap_panel(
    *,
    ax: plt.Axes,
    values: np.ndarray,
    annotations: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap_name: str,
    radii: Sequence[float],
    budgets: Sequence[int],
    ylabel: bool = False,
    value_is_fraction: bool = False,
) -> None:
    finite_values = values[np.isfinite(values)]
    vmax = float(finite_values.max()) if finite_values.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#f2f2f2")
    image = ax.imshow(values, aspect="equal", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=vmax)

    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    ax.set_xlabel(r"Facility budget $p$")
    if ylabel:
        ax.set_ylabel(r"Coverage radius $\tau$ (km)")
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([str(value) for value in budgets])
    ax.set_yticks(range(len(radii)))
    ax.set_yticklabels([f"{value:g}" for value in radii])
    ax.set_xticks(np.arange(-0.5, len(budgets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(radii), 1), minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            text_color = TEXT_COLOR_ON_DARK if value / vmax >= 0.58 else TEXT_COLOR
            ax.text(
                col_idx,
                row_idx,
                annotations[row_idx, col_idx],
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.6,
                linespacing=1.25,
            )

    colorbar = ax.figure.colorbar(image, ax=ax, pad=0.02, fraction=0.046)
    colorbar.set_label(colorbar_label)
    if value_is_fraction:
        ticks = [float(tick) for tick in colorbar.get_ticks()]
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"{100.0 * tick:.0f}%" for tick in ticks])


def _matrix(
    summary: pd.DataFrame,
    radii: Sequence[float],
    budgets: Sequence[int],
    column: str,
) -> np.ndarray:
    pivot = summary.pivot(
        index="coverage_radius_km",
        columns="facility_budget",
        values=column,
    ).reindex(index=radii, columns=budgets)
    return pivot.to_numpy(dtype=float)


def _annotation_matrix(
    summary: pd.DataFrame,
    radii: Sequence[float],
    budgets: Sequence[int],
    formatter,
) -> np.ndarray:
    rows = []
    for radius in radii:
        row_values = []
        for budget in budgets:
            match = summary.loc[
                np.isclose(summary["coverage_radius_km"], radius)
                & summary["facility_budget"].eq(int(budget))
            ]
            if match.empty:
                row_values.append("")
            else:
                row_values.append(formatter(match.iloc[0]))
        rows.append(row_values)
    return np.asarray(rows, dtype=object)


def _panel_c_top_annotation(row: pd.Series) -> str:
    top_cross = str(row.get("top_cross_dvi_interaction", ""))
    if top_cross:
        return top_cross
    top_feature = str(row.get("top_joint_info_feature", ""))
    if top_feature:
        return f"{FEATURE_LABELS.get(top_feature, top_feature)} (base)"
    return "baseline"


def _panel_c_main_annotation(row: pd.Series) -> str:
    top_feature = str(row.get("top_joint_info_feature", ""))
    if top_feature and top_feature != "nan":
        return format_post_joint_term_label(top_feature)
    return "baseline"


def panel_c_plot_spec(panel_c_value: str) -> tuple[str, str, str, str]:
    if panel_c_value == "top-term":
        return (
            "top_post_joint_dvi_value",
            "Top term value",
            "top_post_joint_dvi_term_label",
            "C. Top Terms",
        )
    if panel_c_value == "joint-info":
        return (
            "total_post_joint_info_value",
            "Post-JointDVI main-effect total",
            "panel_c_main_annotation",
            "C. Main Effects",
        )
    if panel_c_value == "cross-dvi":
        return (
            "cross_dvi_intensity",
            "Info-design Cross-DVI intensity",
            "panel_c_top_annotation",
            "C. Top Interactions",
        )
    raise ValueError(f"Unknown Panel C value mode: {panel_c_value}")


def format_design_regime_title(radius: float, budget: int) -> str:
    regime = REGIME_LABELS.get((float(radius), int(budget)), "")
    suffix = f" ({regime})" if regime else ""
    return rf"$\tau={radius:g}$ km, $p={int(budget)}${suffix}"


def format_interaction_label(info_player: str, design_player: str) -> str:
    info_label = FEATURE_LABELS.get(info_player, info_player)
    design_label = DESIGN_LABELS.get(design_player, design_player)
    return rf"{info_label} $\times$ {design_label}"


def format_post_joint_term_label(
    info_player: str,
    *,
    design_player: str | None = None,
) -> str:
    info_label = DETAILED_FEATURE_LABELS.get(info_player, info_player)
    if design_player is not None:
        design_label = DETAILED_DESIGN_LABELS.get(design_player, design_player)
        return rf"{info_label} $\times$ {design_label}"
    if info_player == "zone_hour_baseline":
        return "ZH baseline"
    return f"{info_label} main effect"


def format_uncovered(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value == 0.0:
        return "0.0"
    if abs(value) < 0.1:
        return "<0.1"
    return f"{value:.1f}"


def compute_coalition_policy_change_rate(coalition_values: pd.DataFrame) -> float:
    frame = coalition_values.loc[
        :,
        [
            "timestamp_hour",
            "coalition_mask",
            "decision_selected_facility_indices",
        ],
    ].copy()
    frame["coalition_mask"] = frame["coalition_mask"].astype(int)
    frame["_selected_policy"] = frame["decision_selected_facility_indices"].map(
        _parse_facility_index_set
    )

    baselines = frame.loc[
        frame["coalition_mask"].eq(0),
        ["timestamp_hour", "_selected_policy"],
    ]
    if baselines.empty:
        raise ValueError("Coalition values are missing the empty-coalition baseline.")
    if baselines["timestamp_hour"].duplicated().any():
        raise ValueError("Expected one empty-coalition baseline per timestamp.")

    baseline_by_hour = baselines.set_index("timestamp_hour")["_selected_policy"]
    baseline_policy = frame["timestamp_hour"].map(baseline_by_hour)
    if baseline_policy.isna().any():
        raise ValueError("Some coalition rows have no matching baseline timestamp.")

    return float(frame["_selected_policy"].ne(baseline_policy).mean())


def _parse_facility_index_set(raw_indices: object) -> tuple[int, ...]:
    parsed = json.loads(str(raw_indices))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list of facility indices, got {raw_indices}")
    return tuple(sorted(int(value) for value in parsed))


def _info_player_from_pair(left: str, right: str) -> str | None:
    if left in INFO_PLAYERS:
        return left
    if right in INFO_PLAYERS:
        return right
    return None


def _design_player_from_pair(left: str, right: str) -> str | None:
    if left in DESIGN_PLAYERS:
        return left
    if right in DESIGN_PLAYERS:
        return right
    return None


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


if __name__ == "__main__":
    main()
