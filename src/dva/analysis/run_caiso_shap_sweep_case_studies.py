from __future__ import annotations

import argparse
from dataclasses import replace

from dva.analysis.caiso_shap import (
    run_caiso_shap_case_study,
    write_caiso_shap_case_study_outputs,
)
from dva.analysis.caiso_sweep_runs import (
    case_study_outputs_are_complete,
    load_caiso_shap_case_study_sweep_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CAISO SHAP case studies for every setting listed in a sweep manifest."
        ),
    )
    parser.add_argument("--manifest", required=True, help="Path to the sweep manifest CSV.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run settings even when all required outputs already exist.",
    )
    parser.add_argument(
        "--setting-id",
        action="append",
        default=None,
        help="Optional setting_id to run. Repeat to run multiple settings.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Shared learning rate override for torch_mlp and spo_mlp runs. "
            "For spo_mlp, this applies to both phases unless overridden by "
            "--mse-lr or --spo-lr."
        ),
    )
    parser.add_argument(
        "--mse-lr",
        type=float,
        default=None,
        help="Optional MSE-phase learning rate override for spo_mlp runs.",
    )
    parser.add_argument(
        "--spo-lr",
        type=float,
        default=None,
        help="Optional SPO+-phase learning rate override for spo_mlp runs.",
    )
    parser.add_argument(
        "--training-verbose",
        action="store_true",
        help="Print progress logs during torch_mlp or spo_mlp training.",
    )
    parser.add_argument(
        "--training-log-every",
        type=int,
        default=None,
        help="Log every N training epochs when --training-verbose is enabled.",
    )
    parser.add_argument(
        "--spo-processes",
        type=int,
        default=None,
        help=(
            "Number of PyEPO worker processes for SPO+ training. "
            "Defaults to a small multicore setting when omitted."
        ),
    )
    parser.add_argument(
        "--spo-warm-start-with-mse",
        action="store_true",
        help=(
            "For spo_mlp runs, add an MSE pretraining phase for "
            "the configured mlp_max_iter before SPO+ training."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_entries = load_caiso_shap_case_study_sweep_manifest(args.manifest)
    selected_setting_ids = set(args.setting_id or [])
    if selected_setting_ids:
        manifest_setting_ids = {
            run_entry.manifest_entry.setting_id
            for run_entry in run_entries
        }
        unknown_setting_ids = sorted(selected_setting_ids - manifest_setting_ids)
        if unknown_setting_ids:
            raise ValueError(
                "Unknown setting_id values: " + ", ".join(unknown_setting_ids)
            )
        run_entries = [
            run_entry
            for run_entry in run_entries
            if run_entry.manifest_entry.setting_id in selected_setting_ids
        ]

    executed = 0
    skipped = 0
    total = len(run_entries)
    for index, run_entry in enumerate(run_entries, start=1):
        setting_id = run_entry.manifest_entry.setting_id
        results_dir = run_entry.manifest_entry.results_dir
        if not args.overwrite and case_study_outputs_are_complete(results_dir):
            skipped += 1
            print(f"[{index}/{total}] Skipping {setting_id}; outputs already exist at {results_dir}")
            continue

        run_config = run_entry.config
        if (
            args.lr is not None
            or args.mse_lr is not None
            or args.spo_lr is not None
            or args.training_verbose
            or args.training_log_every is not None
            or args.spo_processes is not None
            or args.spo_warm_start_with_mse
        ):
            run_config = replace(
                run_config,
                learning_rate=(
                    args.lr if args.lr is not None else run_config.learning_rate
                ),
                mse_learning_rate=(
                    args.mse_lr
                    if args.mse_lr is not None
                    else run_config.mse_learning_rate
                ),
                spo_learning_rate=(
                    args.spo_lr
                    if args.spo_lr is not None
                    else run_config.spo_learning_rate
                ),
                training_verbose=args.training_verbose or run_config.training_verbose,
                training_log_every=(
                    args.training_log_every
                    if args.training_log_every is not None
                    else run_config.training_log_every
                ),
                spo_processes=(
                    args.spo_processes
                    if args.spo_processes is not None
                    else run_config.spo_processes
                ),
                spo_warm_start_with_mse=(
                    args.spo_warm_start_with_mse
                    or run_config.spo_warm_start_with_mse
                ),
            )

        print(f"[{index}/{total}] Running {setting_id} -> {results_dir}")
        outputs = run_caiso_shap_case_study(run_config)
        write_caiso_shap_case_study_outputs(outputs, run_config.outdir)
        executed += 1
        print(f"[{index}/{total}] Wrote CAISO SHAP case study outputs to {results_dir}")

    print(
        "Finished sweep case studies: "
        f"executed={executed}, skipped={skipped}, total={total}"
    )


if __name__ == "__main__":
    main()
