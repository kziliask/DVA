from __future__ import annotations

import argparse

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
