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
    run_ems_exact_shap,
    write_ems_exact_shap_outputs,
)
from dva.case_studies.ems.designs import EMS_DESIGN_ACTUALS, EMS_DESIGN_BASELINE
from dva.case_studies.ems.outputs import write_canonical_ems_dva_outputs
from dva.games import build_design_players, build_joint_players, materialize_dvi_interactions


DESIGN_PLAYERS = ("solver", "radius_km", "staging_areas")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EMS DesignDVA or joint-DVI-oriented design sweeps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--analysis-kind", choices=("designdva", "joint_dvi"), required=True)
    parser.add_argument("--solver", choices=tuple(EMS_DESIGN_ACTUALS), required=True)
    parser.add_argument("--value-mode", choices=("post", "ante"), required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--holdout-hours", type=int, default=100)
    parser.add_argument("--max-hours", type=int, default=None)
    parser.add_argument("--background-rows", type=int, default=100)
    parser.add_argument("--train-sample-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _default_outdir(args: argparse.Namespace) -> Path:
    return (
        Path("results/ems")
        / args.analysis_kind
        / f"{args.solver}_{args.value_mode}"
    )


def _design_for_mask(mask: int, solver_key: str) -> dict[str, Any]:
    actual = EMS_DESIGN_ACTUALS[solver_key]
    baseline = EMS_DESIGN_BASELINE
    return {
        "solver": actual.solver if mask & 0b001 else baseline.solver,
        "radius_km": actual.radius_km if mask & 0b010 else baseline.radius_km,
        "staging_areas": actual.staging_areas if mask & 0b100 else baseline.staging_areas,
    }


def _setting_dir(outdir: Path, mask: int, design: dict[str, Any]) -> Path:
    return outdir / "design_coalitions" / (
        f"mask_{mask:03b}_{design['solver']}_"
        f"r{float(design['radius_km']):g}_p{int(design['staging_areas'])}"
    )


def _run_or_load_design_setting(
    *,
    args: argparse.Namespace,
    mask: int,
    design: dict[str, Any],
) -> Any:
    setting_dir = _setting_dir(args.outdir, mask, design)
    if not args.overwrite and (setting_dir / "hourly_shap.csv").exists():
        return load_ems_exact_shap_outputs(setting_dir)
    config = EmsExactShapConfig(
        outdir=setting_dir,
        holdout_hours=args.holdout_hours,
        max_hours=args.max_hours,
        background_rows=args.background_rows,
        train_sample_rows=args.train_sample_rows,
        coverage_solver=str(design["solver"]),
        coverage_radius_km=float(design["radius_km"]),
        facility_budget=int(design["staging_areas"]),
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


def _write_design_dva(args: argparse.Namespace, design_values: np.ndarray) -> None:
    shap_values = compute_exact_shapley_values(design_values - design_values[0], 3)
    actual = EMS_DESIGN_ACTUALS[args.solver]
    baseline = EMS_DESIGN_BASELINE
    attr_by_player = {
        "solver": "solver",
        "radius_km": "radius_km",
        "staging_areas": "staging_areas",
    }
    rows = [
        {
            "player": player,
            "dva_value": float(value),
            "actual": getattr(actual, attr_by_player[player]),
            "baseline": getattr(baseline, attr_by_player[player]),
            "value_mode": args.value_mode,
        }
        for player, value in zip(DESIGN_PLAYERS, shap_values, strict=True)
    ]
    pd.DataFrame(rows).to_csv(args.outdir / "design_dva.csv", index=False)
    coalition_rows = []
    for mask, value in enumerate(design_values):
        design = _design_for_mask(mask, args.solver)
        coalition_rows.append({"coalition_mask": mask, "design_value": float(value), **design})
    pd.DataFrame(coalition_rows).to_csv(args.outdir / "design_coalition_values.csv", index=False)


def _design_player_set(args: argparse.Namespace) -> Any:
    actual = EMS_DESIGN_ACTUALS[args.solver]
    baseline = EMS_DESIGN_BASELINE
    return build_design_players(
        {
            "solver": actual.solver,
            "radius_km": actual.radius_km,
            "staging_areas": actual.staging_areas,
        },
        {
            "solver": baseline.solver,
            "radius_km": baseline.radius_km,
            "staging_areas": baseline.staging_areas,
        },
    )


def _joint_value_column(value_mode: str) -> str:
    return "ante_decision_value" if value_mode == "ante" else "decision_value"


def _write_joint_dva_and_dvi(
    args: argparse.Namespace,
    outputs_by_mask: dict[int, Any],
) -> None:
    reference_outputs = outputs_by_mask[0b111]
    info_player_names = tuple(reference_outputs.run_metadata["player_names"])
    player_set = build_joint_players(info_player_names, _design_player_set(args))
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
                    f"{timestamp} and design mask {design_mask:03b}; got {len(group)}."
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
    summary.to_csv(args.outdir / "joint_summary_dva.csv", index=False)
    pd.DataFrame(interaction_rows).to_csv(args.outdir / "dvi_interactions.csv", index=False)


def main() -> None:
    args = build_parser().parse_args()
    args.outdir = args.outdir or _default_outdir(args)
    if args.dry_run:
        command = [
            "uv",
            "run",
            "dva-ems-design-joint-dvi",
            "--analysis-kind",
            args.analysis_kind,
            "--solver",
            args.solver,
            "--value-mode",
            args.value_mode,
            "--outdir",
            str(args.outdir),
        ]
        if args.max_hours is not None:
            command.extend(["--max-hours", str(args.max_hours)])
        if args.train_sample_rows is not None:
            command.extend(["--train-sample-rows", str(args.train_sample_rows)])
        if args.overwrite:
            command.append("--overwrite")
        print(shlex.join(command))
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    design_values = np.zeros(8, dtype=float)
    outputs_by_mask: dict[int, Any] = {}
    for mask in range(8):
        design = _design_for_mask(mask, args.solver)
        outputs = _run_or_load_design_setting(args=args, mask=mask, design=design)
        design_values[mask] = _mean_design_value(outputs, args.value_mode)
        outputs_by_mask[mask] = outputs
    _write_design_dva(args, design_values)
    if args.analysis_kind == "joint_dvi":
        _write_joint_dva_and_dvi(args, outputs_by_mask)
    metadata = {
        "analysis_kind": args.analysis_kind,
        "solver": args.solver,
        "value_mode": args.value_mode,
        "design_players": list(DESIGN_PLAYERS),
        "actual_design": {
            "name": EMS_DESIGN_ACTUALS[args.solver].name,
            "solver": EMS_DESIGN_ACTUALS[args.solver].solver,
            "radius_km": EMS_DESIGN_ACTUALS[args.solver].radius_km,
            "staging_areas": EMS_DESIGN_ACTUALS[args.solver].staging_areas,
        },
        "baseline_design": {
            "name": EMS_DESIGN_BASELINE.name,
            "solver": EMS_DESIGN_BASELINE.solver,
            "radius_km": EMS_DESIGN_BASELINE.radius_km,
            "staging_areas": EMS_DESIGN_BASELINE.staging_areas,
        },
    }
    (args.outdir / "design_dva_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote EMS {args.analysis_kind} outputs to {args.outdir}")


if __name__ == "__main__":
    main()
