from __future__ import annotations

import argparse
import shlex
from dataclasses import replace
from pathlib import Path
from typing import Any

from dva.analysis.caiso_shap import (
    CaisoShapCaseStudyConfig,
    run_caiso_shap_case_study_with_artifacts,
    write_caiso_shap_case_study_outputs,
)
from dva.analysis.run_caiso_decision_shap_guided_validation import (
    build_fixed_caiso_guided_validation_split,
    make_model_manifest,
)
from dva.case_studies.caiso.designs import (
    CAISO_ACTUAL_DESIGN,
    CAISO_BASELINE_DESIGNS,
    CaisoStorageDesign,
    parameter_player_spec_for_baseline,
)
from dva.case_studies.caiso.outputs import write_canonical_caiso_dva_outputs
from dva.model.train import train_model
from dva.model.storage_dispatch import StorageDispatchParameters


def _xgb_model_ids() -> tuple[str, ...]:
    manifest = make_model_manifest()
    return tuple(
        manifest.loc[manifest["model_name"].eq("xgb"), "model_id"].astype(str)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CAISO JointDVA and order-2 Faith-SHAP DVI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/cleaned/caiso_sp15_daily_lmp_weather_2023-01-26_2026-05-07.csv"),
    )
    design_group = parser.add_mutually_exclusive_group(required=True)
    design_group.add_argument(
        "--baseline",
        choices=tuple(CAISO_BASELINE_DESIGNS),
        help=(
            "Compare the selected design baseline against the current CAISO "
            "storage configuration."
        ),
    )
    design_group.add_argument(
        "--target",
        choices=tuple(CAISO_BASELINE_DESIGNS),
        help=(
            "Flipped orientation: keep the current CAISO storage configuration "
            "as the baseline and compare against the selected target."
        ),
    )
    parser.add_argument("--model-id", choices=_xgb_model_ids(), default="xgb_001")
    parser.add_argument("--value-mode", choices=("post", "ante"), required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--background-days", type=int, default=365)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _experiment_name(args: argparse.Namespace) -> str:
    return "joint_dvi_flipped" if args.target is not None else "joint_dvi"


def _design_label(args: argparse.Namespace) -> str:
    return str(args.target if args.target is not None else args.baseline)


def _default_outdir(
    model_id: str,
    design_label: str,
    value_mode: str,
    *,
    experiment_name: str = "joint_dvi",
) -> Path:
    return Path("results/caiso") / experiment_name / model_id / f"{design_label}_{value_mode}"


def _xgb_model_record(model_id: str) -> dict[str, Any]:
    manifest = make_model_manifest()
    row = manifest.loc[
        manifest["model_id"].eq(model_id) & manifest["model_name"].eq("xgb")
    ].iloc[0]
    return dict(row)


def _flipped_target_parameters(target: CaisoStorageDesign) -> StorageDispatchParameters:
    current_parameters = CAISO_ACTUAL_DESIGN.parameters()
    return replace(
        target.parameters(),
        throughput_penalty=current_parameters.throughput_penalty,
    )


def _config_from_record(args: argparse.Namespace) -> CaisoShapCaseStudyConfig:
    record = _xgb_model_record(args.model_id)
    if args.target is None:
        baseline = CAISO_BASELINE_DESIGNS[args.baseline]
        target = CAISO_ACTUAL_DESIGN
        storage_parameters = target.parameters()
        parameter_player_spec = parameter_player_spec_for_baseline(baseline)
    else:
        baseline = CAISO_ACTUAL_DESIGN
        target = CAISO_BASELINE_DESIGNS[args.target]
        storage_parameters = _flipped_target_parameters(target)
        parameter_player_spec = parameter_player_spec_for_baseline(
            baseline,
            include_state_of_charge=True,
        )
    return CaisoShapCaseStudyConfig(
        dataset_path=args.dataset_path,
        holdout_days=0,
        outdir=args.outdir
        or _default_outdir(
            args.model_id,
            _design_label(args),
            args.value_mode,
            experiment_name=_experiment_name(args),
        ),
        model_name="xgb",
        random_state=0,
        n_jobs=1,
        xgb_n_estimators=int(record["n_estimators"]),
        xgb_max_depth=int(record["max_depth"]),
        xgb_learning_rate=float(record["learning_rate"]),
        xgb_subsample=float(record["subsample"]),
        xgb_colsample_bytree=float(record["colsample_bytree"]),
        xgb_reg_lambda=float(record["reg_lambda"]),
        storage_parameters=storage_parameters,
        background_days=args.background_days,
        max_days=args.max_days,
        compute_ead_decision_shap=args.value_mode == "ante",
        interaction_order=2,
        interaction_method="faith_shap",
        parameter_player_spec=parameter_player_spec,
    )


def main() -> None:
    args = build_parser().parse_args()
    experiment_name = _experiment_name(args)
    design_label = _design_label(args)
    args.outdir = args.outdir or _default_outdir(
        args.model_id,
        design_label,
        args.value_mode,
        experiment_name=experiment_name,
    )
    if args.dry_run:
        design_arg = "--target" if args.target is not None else "--baseline"
        command = [
            "uv",
            "run",
            "dva-caiso-joint-dvi",
            "--model-id",
            args.model_id,
            design_arg,
            design_label,
            "--value-mode",
            args.value_mode,
            "--outdir",
            str(args.outdir),
        ]
        if args.max_days is not None:
            command.extend(["--max-days", str(args.max_days)])
        print(shlex.join(command))
        return

    split = build_fixed_caiso_guided_validation_split(
        args.dataset_path,
        train_months=24,
        validation_months=12,
        test_rest=True,
        background_days=args.background_days,
    )
    config = _config_from_record(args)
    train_x = split.train_frame.loc[:, list(split.feature_columns)]
    train_y = split.train_frame.loc[:, list(split.target_columns)]
    artifacts = train_model(
        train_x,
        train_y,
        model_name=config.model_name,
        feature_columns=split.feature_columns,
        target_columns=split.target_columns,
        random_state=config.random_state,
        n_jobs=1,
        xgb_n_estimators=config.xgb_n_estimators,
        xgb_max_depth=config.xgb_max_depth,
        xgb_learning_rate=config.xgb_learning_rate,
        xgb_subsample=config.xgb_subsample,
        xgb_colsample_bytree=config.xgb_colsample_bytree,
        xgb_reg_lambda=config.xgb_reg_lambda,
        xgb_verbosity=config.xgb_verbosity,
    )
    explain_frame = split.test_frame
    if args.max_days is not None:
        explain_frame = explain_frame.iloc[: args.max_days].reset_index(drop=True)
    outputs = run_caiso_shap_case_study_with_artifacts(
        config=config,
        training_artifacts=artifacts,
        dataset_path=split.dataset_path,
        date_column=split.date_column,
        feature_columns=split.feature_columns,
        target_columns=split.target_columns,
        train_frame=split.train_frame,
        background_frame=split.background_frame,
        explain_frame=explain_frame,
        holdout_days=split.validation_days + split.test_days,
        max_days=len(explain_frame),
        evaluation_label=f"{experiment_name}_{args.model_id}_{design_label}_{args.value_mode}",
    )
    write_caiso_shap_case_study_outputs(outputs, config.outdir)
    write_canonical_caiso_dva_outputs(config.outdir, value_mode=args.value_mode)
    print(f"Wrote CAISO JointDVA/DVI outputs to {config.outdir}")


if __name__ == "__main__":
    main()
