from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dva.analysis.caiso_shap import compute_exact_faith_shap_values
from dva.analysis.ems_exact_shap import (
    EmsExactShapConfig,
    compute_exact_shapley_values,
    load_ems_exact_shap_outputs,
    normalize_ems_coverage_solver,
    run_ems_exact_shap,
    write_ems_exact_shap_outputs,
)
from dva.case_studies.ems.designs import EMS_DESIGN_ACTUALS, EMS_DESIGN_BASELINE, EmsDesign
from dva.case_studies.ems.models import (
    EMS_XGB_MODEL_IDS,
    ems_xgb_config_kwargs,
    resolve_ems_xgb_model_record,
)
from dva.case_studies.ems.outputs import write_canonical_ems_dva_outputs
from dva.games import build_design_players, build_joint_players, materialize_dvi_interactions


DESIGN_FIELDS = ("solver", "radius_km", "staging_areas")
SOLVER_ALIASES = {
    "exact": "exact",
    "gurobi": "exact",
    "naive": "naive_greedy",
    "greedy": "greedy_max_cover",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EMS DesignDVA, JointDVA, and order-2 Faith-SHAP DVI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--analysis-kind", choices=("designdva", "joint_dvi"), required=True)
    parser.add_argument(
        "--solver",
        choices=tuple(EMS_DESIGN_ACTUALS),
        default=None,
        help=(
            "Legacy shorthand: compare exact p=8,tau=3 baseline against the "
            "selected heuristic p=3,tau=1 design."
        ),
    )
    parser.add_argument("--model-id", choices=EMS_XGB_MODEL_IDS, default="xgb_001")
    parser.add_argument("--value-mode", choices=("post", "ante"), required=True)
    parser.add_argument("--baseline-solver", default=None)
    parser.add_argument("--target-solver", default=None)
    parser.add_argument("--baseline-radius-km", type=float, default=None)
    parser.add_argument("--target-radius-km", type=float, default=None)
    parser.add_argument("--baseline-staging-areas", "--baseline-p", type=int, default=None)
    parser.add_argument("--target-staging-areas", "--target-p", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--holdout-hours", type=int, default=100)
    parser.add_argument("--max-hours", type=int, default=None)
    parser.add_argument("--background-rows", type=int, default=100)
    parser.add_argument("--train-sample-rows", type=int, default=None)
    parser.add_argument(
        "--solver-threads",
        "--gurobi-threads",
        dest="gurobi_threads",
        type=int,
        default=1,
    )
    parser.add_argument("--optimization-solver", choices=("highs", "gurobi"), default="highs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _default_outdir(
    args: argparse.Namespace,
    baseline: EmsDesign,
    actual: EmsDesign,
    design_players: tuple[str, ...],
) -> Path:
    player_label = "_".join(design_players)
    return (
        Path("results/ems")
        / args.analysis_kind
        / args.model_id
        / f"{baseline.name}_to_{actual.name}_{player_label}_{args.value_mode}"
    )


def _normalize_solver_name(solver_name: str) -> str:
    key = str(solver_name).strip().lower().replace("-", "_")
    return normalize_ems_coverage_solver(SOLVER_ALIASES.get(key, key))


def _design_name(design: EmsDesign) -> str:
    solver_label = {
        "gurobi": "exact",
        "exact": "exact",
        "naive_greedy": "naive",
        "greedy_max_cover": "greedy",
    }.get(design.solver, design.solver)
    return f"{solver_label}_p{int(design.staging_areas)}_tau{float(design.radius_km):g}"


def _resolve_designs(args: argparse.Namespace) -> tuple[EmsDesign, EmsDesign]:
    custom_design_requested = any(
        value is not None
        for value in (
            args.baseline_solver,
            args.target_solver,
            args.baseline_radius_km,
            args.target_radius_km,
            args.baseline_staging_areas,
            args.target_staging_areas,
        )
    )
    if args.solver is not None and not custom_design_requested:
        return EMS_DESIGN_BASELINE, EMS_DESIGN_ACTUALS[args.solver]

    baseline = EmsDesign(
        name="",
        solver=_normalize_solver_name(args.baseline_solver or "exact"),
        radius_km=float(1.0 if args.baseline_radius_km is None else args.baseline_radius_km),
        staging_areas=int(
            3 if args.baseline_staging_areas is None else args.baseline_staging_areas
        ),
    )
    actual = EmsDesign(
        name="",
        solver=_normalize_solver_name(args.target_solver or "exact"),
        radius_km=float(1.0 if args.target_radius_km is None else args.target_radius_km),
        staging_areas=int(8 if args.target_staging_areas is None else args.target_staging_areas),
    )
    return (
        EmsDesign(
            name=_design_name(baseline),
            solver=baseline.solver,
            radius_km=baseline.radius_km,
            staging_areas=baseline.staging_areas,
        ),
        EmsDesign(
            name=_design_name(actual),
            solver=actual.solver,
            radius_km=actual.radius_km,
            staging_areas=actual.staging_areas,
        ),
    )


def _design_players_for_designs(
    baseline: EmsDesign,
    actual: EmsDesign,
) -> tuple[str, ...]:
    players = []
    for field in DESIGN_FIELDS:
        baseline_value = getattr(baseline, field)
        actual_value = getattr(actual, field)
        if isinstance(baseline_value, float) or isinstance(actual_value, float):
            differs = not np.isclose(float(baseline_value), float(actual_value))
        else:
            differs = baseline_value != actual_value
        if differs:
            players.append(field)
    return tuple(players)


def _design_for_mask(
    mask: int,
    *,
    design_players: tuple[str, ...],
    baseline: EmsDesign,
    actual: EmsDesign,
) -> dict[str, Any]:
    design = {
        "solver": baseline.solver,
        "radius_km": baseline.radius_km,
        "staging_areas": baseline.staging_areas,
    }
    for player_idx, field in enumerate(design_players):
        if mask & (1 << player_idx):
            design[field] = getattr(actual, field)
    return design


def _setting_dir(outdir: Path, mask: int, design: dict[str, Any], player_count: int) -> Path:
    return outdir / "design_coalitions" / (
        f"mask_{mask:0{player_count}b}_{design['solver']}_"
        f"tau{float(design['radius_km']):g}_p{int(design['staging_areas'])}"
    )


def _run_or_load_design_setting(
    *,
    args: argparse.Namespace,
    model_record: dict[str, Any],
    design_players: tuple[str, ...],
    mask: int,
    design: dict[str, Any],
) -> Any:
    setting_dir = _setting_dir(args.outdir, mask, design, len(design_players))
    if not args.overwrite and (setting_dir / "hourly_shap.csv").exists():
        return load_ems_exact_shap_outputs(setting_dir)
    config = EmsExactShapConfig(
        outdir=setting_dir,
        holdout_hours=args.holdout_hours,
        max_hours=args.max_hours,
        background_rows=args.background_rows,
        train_sample_rows=args.train_sample_rows,
        model_id=args.model_id,
        **ems_xgb_config_kwargs(model_record),
        coverage_solver=str(design["solver"]),
        coverage_radius_km=float(design["radius_km"]),
        facility_budget=int(design["staging_areas"]),
        gurobi_threads=args.gurobi_threads,
        optimization_solver=args.optimization_solver,
        compute_cvar_decision_shap=False,
        compute_ante_decision_shap=args.value_mode == "ante",
        save_coalition_values=args.analysis_kind == "joint_dvi",
        progress_every_coalitions=0,
    )
    outputs = run_ems_exact_shap(config)
    write_ems_exact_shap_outputs(outputs, config.outdir)
    write_canonical_ems_dva_outputs(config.outdir, value_mode=args.value_mode)
    return outputs


def _mean_design_value(outputs: Any, value_mode: str) -> float:
    hourly = outputs.hourly_shap
    if value_mode == "ante":
        if "ante_decision_full_value" not in hourly:
            raise ValueError("Ante DesignDVA requires ante EMS decision values.")
        return float(hourly["ante_decision_full_value"].mean())
    return float(hourly["decision_full_value"].mean())


def _write_design_dva(
    args: argparse.Namespace,
    *,
    baseline: EmsDesign,
    actual: EmsDesign,
    design_players: tuple[str, ...],
    design_values: np.ndarray,
) -> None:
    shap_values = compute_exact_shapley_values(
        design_values - design_values[0],
        len(design_players),
    )
    rows = [
        {
            "player": player,
            "dva_value": float(value),
            "actual": getattr(actual, player),
            "baseline": getattr(baseline, player),
            "value_mode": args.value_mode,
            "model_id": args.model_id,
        }
        for player, value in zip(design_players, shap_values, strict=True)
    ]
    pd.DataFrame(
        rows,
        columns=[
            "player",
            "dva_value",
            "actual",
            "baseline",
            "value_mode",
            "model_id",
        ],
    ).to_csv(args.outdir / "design_dva.csv", index=False)
    coalition_rows = []
    for mask, value in enumerate(design_values):
        design = _design_for_mask(
            mask,
            design_players=design_players,
            baseline=baseline,
            actual=actual,
        )
        coalition_rows.append({"coalition_mask": mask, "design_value": float(value), **design})
    pd.DataFrame(coalition_rows).to_csv(args.outdir / "design_coalition_values.csv", index=False)


def _design_player_set(
    *,
    baseline: EmsDesign,
    actual: EmsDesign,
    design_players: tuple[str, ...],
) -> Any:
    return build_design_players(
        {field: getattr(actual, field) for field in design_players},
        {field: getattr(baseline, field) for field in design_players},
    )


def _joint_value_column(value_mode: str) -> str:
    return "ante_decision_value" if value_mode == "ante" else "decision_value"


def _write_joint_dva_and_dvi(
    args: argparse.Namespace,
    *,
    baseline: EmsDesign,
    actual: EmsDesign,
    design_players: tuple[str, ...],
    outputs_by_mask: dict[int, Any],
) -> None:
    reference_outputs = outputs_by_mask[(1 << len(design_players)) - 1]
    info_player_names = tuple(reference_outputs.run_metadata["player_names"])
    player_set = build_joint_players(
        info_player_names,
        _design_player_set(
            baseline=baseline,
            actual=actual,
            design_players=design_players,
        ),
    )
    info_player_count = len(info_player_names)
    value_column = _joint_value_column(args.value_mode)

    coalition_frames: dict[int, pd.DataFrame] = {}
    for design_mask, outputs in outputs_by_mask.items():
        if outputs.coalition_values is None:
            raise ValueError("JointDVA/DVI requires saved EMS coalition values.")
        frame = outputs.coalition_values.copy()
        if value_column not in frame.columns:
            raise ValueError(f"JointDVA/DVI requires {value_column!r} in coalition values.")
        frame["timestamp_hour"] = frame["timestamp_hour"].astype(str)
        coalition_frames[design_mask] = frame

    timestamp_order = list(
        dict.fromkeys(coalition_frames[0]["timestamp_hour"].astype(str).tolist())
    )
    hourly_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    for timestamp in timestamp_order:
        joint_values = np.zeros(player_set.coalition_count, dtype=float)
        for design_mask, frame in coalition_frames.items():
            group = frame.loc[frame["timestamp_hour"].eq(timestamp)].sort_values(
                "coalition_mask"
            )
            expected_info_coalitions = 1 << info_player_count
            if len(group) != expected_info_coalitions:
                raise ValueError(
                    f"Expected {expected_info_coalitions} info coalitions for "
                    f"{timestamp} and design mask {design_mask:b}; got {len(group)}."
                )
            for row in group.itertuples(index=False):
                info_mask = int(getattr(row, "coalition_mask"))
                joint_mask = info_mask | (design_mask << info_player_count)
                joint_values[joint_mask] = float(getattr(row, value_column))

        characteristic_values = joint_values - joint_values[0]
        shap_values = compute_exact_shapley_values(
            characteristic_values,
            player_set.count,
        )
        for player, value in zip(player_set.players, shap_values, strict=True):
            hourly_rows.append(
                {
                    "timestamp_hour": timestamp,
                    "player": player.name,
                    "player_kind": player.kind.value,
                    "baseline": player.baseline,
                    "actual": player.actual,
                    "dva_value": float(value),
                    "value_mode": args.value_mode,
                    "model_id": args.model_id,
                }
            )

        interaction_values = compute_exact_faith_shap_values(
            characteristic_values,
            player_set.count,
            order=2,
        )
        for interaction in materialize_dvi_interactions(player_set, interaction_values):
            if len(interaction.indices) != 2:
                continue
            interaction_rows.append(
                {
                    "timestamp_hour": timestamp,
                    "players": "|".join(interaction.players),
                    "order": len(interaction.indices),
                    "interaction_type": interaction.interaction_type,
                    "decision_interaction_value": float(np.asarray(interaction.value)),
                    "value_mode": args.value_mode,
                    "model_id": args.model_id,
                }
            )

    hourly = pd.DataFrame(hourly_rows)
    hourly.to_csv(args.outdir / "joint_dva.csv", index=False)
    summary = (
        hourly.groupby(["player", "player_kind"], as_index=False)["dva_value"]
        .agg(dva_mean_signed="mean", dva_mean_abs=lambda series: series.abs().mean())
        .sort_values("dva_mean_abs", ascending=False)
        .reset_index(drop=True)
    )
    summary["dva_rank"] = np.arange(1, len(summary) + 1)
    summary["value_mode"] = args.value_mode
    summary["model_id"] = args.model_id
    summary.to_csv(args.outdir / "joint_summary_dva.csv", index=False)
    pd.DataFrame(interaction_rows).to_csv(args.outdir / "dvi_interactions.csv", index=False)


def _dry_run_command(args: argparse.Namespace) -> list[str]:
    command = [
        "uv",
        "run",
        "dva-ems-design-joint-dvi",
        "--analysis-kind",
        args.analysis_kind,
        "--model-id",
        args.model_id,
        "--value-mode",
        args.value_mode,
        "--outdir",
        str(args.outdir),
    ]
    if args.solver is not None:
        command.extend(["--solver", args.solver])
    for flag in (
        "baseline_solver",
        "target_solver",
        "baseline_radius_km",
        "target_radius_km",
        "baseline_staging_areas",
        "target_staging_areas",
    ):
        value = getattr(args, flag)
        if value is None:
            continue
        command.extend([f"--{flag.replace('_', '-')}", str(value)])
    if args.max_hours is not None:
        command.extend(["--max-hours", str(args.max_hours)])
    if args.train_sample_rows is not None:
        command.extend(["--train-sample-rows", str(args.train_sample_rows)])
    command.extend(
        [
            "--solver-threads",
            str(args.gurobi_threads),
            "--optimization-solver",
            args.optimization_solver,
        ]
    )
    if args.overwrite:
        command.append("--overwrite")
    return command


def main() -> None:
    args = build_parser().parse_args()
    baseline, actual = _resolve_designs(args)
    design_players = _design_players_for_designs(baseline, actual)
    args.outdir = args.outdir or _default_outdir(args, baseline, actual, design_players)
    if args.dry_run:
        print(shlex.join(_dry_run_command(args)))
        return

    model_record = resolve_ems_xgb_model_record(args.model_id)
    args.outdir.mkdir(parents=True, exist_ok=True)
    design_count = 1 << len(design_players)
    design_values = np.zeros(design_count, dtype=float)
    outputs_by_mask: dict[int, Any] = {}
    for mask in range(design_count):
        design = _design_for_mask(
            mask,
            design_players=design_players,
            baseline=baseline,
            actual=actual,
        )
        outputs = _run_or_load_design_setting(
            args=args,
            model_record=model_record,
            design_players=design_players,
            mask=mask,
            design=design,
        )
        design_values[mask] = _mean_design_value(outputs, args.value_mode)
        outputs_by_mask[mask] = outputs
    _write_design_dva(
        args,
        baseline=baseline,
        actual=actual,
        design_players=design_players,
        design_values=design_values,
    )
    if args.analysis_kind == "joint_dvi":
        _write_joint_dva_and_dvi(
            args,
            baseline=baseline,
            actual=actual,
            design_players=design_players,
            outputs_by_mask=outputs_by_mask,
        )
    metadata = {
        "analysis_kind": args.analysis_kind,
        "model_id": args.model_id,
        "model_record": model_record,
        "value_mode": args.value_mode,
        "solver_threads": args.gurobi_threads,
        "optimization_solver": args.optimization_solver,
        "design_players": list(design_players),
        "actual_design": {
            "name": actual.name,
            "solver": actual.solver,
            "radius_km": actual.radius_km,
            "staging_areas": actual.staging_areas,
        },
        "baseline_design": {
            "name": baseline.name,
            "solver": baseline.solver,
            "radius_km": baseline.radius_km,
            "staging_areas": baseline.staging_areas,
        },
    }
    (args.outdir / "design_dva_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote EMS {args.analysis_kind} outputs to {args.outdir}")


if __name__ == "__main__":
    main()
