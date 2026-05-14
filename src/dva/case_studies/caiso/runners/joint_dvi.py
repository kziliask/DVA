from __future__ import annotations

import argparse
import shlex
from pathlib import Path

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
    parameter_player_spec_for_baseline,
)
from dva.case_studies.caiso.outputs import write_canonical_caiso_dva_outputs
from dva.model.train import train_model


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
    parser.add_argument(
        "--baseline",
        choices=tuple(CAISO_BASELINE_DESIGNS),
        required=True,
    )
    parser.add_argument("--value-mode", choices=("post", "ante"), required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--background-days", type=int, default=365)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _default_outdir(baseline: str, value_mode: str) -> Path:
    return Path("results/caiso/joint_dvi") / f"{baseline}_{value_mode}"


def _xgb_001_record() -> dict[str, object]:
    manifest = make_model_manifest()
    row = manifest.loc[manifest["model_id"].eq("xgb_001")].iloc[0]
    return dict(row)


def _config_from_record(args: argparse.Namespace) -> CaisoShapCaseStudyConfig:
    record = _xgb_001_record()
    baseline = CAISO_BASELINE_DESIGNS[args.baseline]
    return CaisoShapCaseStudyConfig(
        dataset_path=args.dataset_path,
        holdout_days=0,
        outdir=args.outdir or _default_outdir(args.baseline, args.value_mode),
        model_name="xgb",
        random_state=0,
        n_jobs=1,
        xgb_n_estimators=int(record["n_estimators"]),
        xgb_max_depth=int(record["max_depth"]),
        xgb_learning_rate=float(record["learning_rate"]),
        xgb_subsample=float(record["subsample"]),
        xgb_colsample_bytree=float(record["colsample_bytree"]),
        xgb_reg_lambda=float(record["reg_lambda"]),
        storage_parameters=CAISO_ACTUAL_DESIGN.parameters(),
        background_days=args.background_days,
        max_days=args.max_days,
        compute_ead_decision_shap=args.value_mode == "ante",
        interaction_order=2,
        interaction_method="faith_shap",
        parameter_player_spec=parameter_player_spec_for_baseline(baseline),
    )


def main() -> None:
    args = build_parser().parse_args()
    args.outdir = args.outdir or _default_outdir(args.baseline, args.value_mode)
    if args.dry_run:
        command = [
            "uv",
            "run",
            "dva-caiso-joint-dvi",
            "--baseline",
            args.baseline,
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
        evaluation_label=f"joint_dvi_{args.baseline}_{args.value_mode}",
    )
    write_caiso_shap_case_study_outputs(outputs, config.outdir)
    write_canonical_caiso_dva_outputs(config.outdir, value_mode=args.value_mode)
    print(f"Wrote CAISO JointDVA/DVI outputs to {config.outdir}")


if __name__ == "__main__":
    main()
