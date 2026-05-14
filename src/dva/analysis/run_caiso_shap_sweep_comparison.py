from __future__ import annotations

import argparse

from dva.analysis.caiso_sweep import (
    DEFAULT_MIN_INVARIANT_DAYS,
    DEFAULT_SWEEP_OUTPUT_DIR,
    compare_caiso_shap_sweeps,
    write_caiso_sweep_comparison_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare neighboring CAISO SHAP sweep settings from a manifest.",
    )
    parser.add_argument("--manifest", required=True, help="Path to the sweep manifest CSV.")
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_SWEEP_OUTPUT_DIR),
        help="Directory where comparison artifacts will be written.",
    )
    parser.add_argument(
        "--min-invariant-days",
        type=int,
        default=DEFAULT_MIN_INVARIANT_DAYS,
        help="Minimum invariant-day count required before invariant metrics are defined.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = compare_caiso_shap_sweeps(
        args.manifest,
        min_invariant_days=args.min_invariant_days,
    )
    write_caiso_sweep_comparison_outputs(outputs, args.outdir)
    print(f"Wrote CAISO SHAP sweep comparison outputs to {args.outdir}")


if __name__ == "__main__":
    main()
