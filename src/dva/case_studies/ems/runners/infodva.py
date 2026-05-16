from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from dva.analysis import run_ems_shap_exhaustive_comparison as legacy_runner


DEFAULT_OUT_ROOT = Path("results/ems/infodva_3x3_exact")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EMS post/ante InfoDVA over the 3x3 exact-solver design.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--plot-root", type=Path, default=Path("results/ems/infodva_3x3_exact_plots"))
    parser.add_argument("--radius", "--coverage-radius-km", dest="radii", action="append", type=float)
    parser.add_argument("--staging", "--facility-budget", dest="budgets", action="append", type=int)
    parser.add_argument("--model-id", action="append", default=None)
    parser.add_argument("--holdout-hours", type=int, default=100)
    parser.add_argument("--background-rows", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_forward_args(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    radii = args.radii or [1.0, 2.0, 3.0]
    budgets = args.budgets or [3, 5, 8]
    forwarded = [
        "--out-root",
        str(args.out_root),
        "--plot-root",
        str(args.plot_root),
        "--solver",
        "exact",
        "--coverage-radius-km",
        *[str(value) for value in radii],
        "--facility-budget",
        *[str(value) for value in budgets],
        "--holdout-hours",
        str(args.holdout_hours),
        "--background-rows",
        str(args.background_rows),
        "--n-jobs",
        str(args.n_jobs),
        "--compute-ante-infodva",
        "--no-plots",
    ]
    if args.overwrite:
        forwarded.append("--overwrite")
    for model_id in args.model_id or ():
        forwarded.extend(["--model-id", str(model_id)])
    forwarded.extend(passthrough)
    return forwarded


def main() -> None:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    forwarded = build_forward_args(args, passthrough)
    if args.dry_run:
        print("uv run python -m dva.analysis.run_ems_shap_exhaustive_comparison " + shlex.join(forwarded))
        return
    original_argv = sys.argv
    try:
        sys.argv = ["run_ems_shap_exhaustive_comparison.py", *forwarded]
        legacy_runner.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
