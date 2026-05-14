from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from dva.analysis import run_caiso_decision_shap_guided_validation as legacy_runner


DEFAULT_OUTDIR = Path("results/caiso/gdsi_xgb_l25_24_12_rest")


def build_forward_args(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    forwarded = [
        "--outdir",
        str(args.outdir),
        "--dataset-path",
        str(args.dataset_path),
        "--model-family",
        "xgb",
        "--train-months",
        "24",
        "--validation-months",
        "12",
        "--test-rest",
        "--compute-ante-infodva",
        "--throughput-penalty",
        "5",
        "--energy-capacity",
        "4",
        "--power-limit",
        "1",
        "--charge-efficiency",
        "0.95",
        "--discharge-efficiency",
        "0.95",
        "--initial-state-of-charge",
        "2",
        "--terminal-state-of-charge",
        "2",
        "--max-workers",
        str(args.max_workers),
    ]
    for model_id in args.model_id or ():
        forwarded.extend(["--model-id", model_id])
    if args.overwrite:
        forwarded.append("--overwrite")
    forwarded.extend(passthrough)
    return forwarded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CAISO 24/12/rest XGBoost InfoDVA/GDSI experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/cleaned/caiso_sp15_daily_lmp_weather_2023-01-26_2026-05-07.csv"),
    )
    parser.add_argument("--model-id", action="append", default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    forwarded = build_forward_args(args, passthrough)
    if args.dry_run:
        print("uv run python -m dva.analysis.run_caiso_decision_shap_guided_validation " + shlex.join(forwarded))
        return
    original_argv = sys.argv
    try:
        sys.argv = ["run_caiso_decision_shap_guided_validation.py", *forwarded]
        legacy_runner.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
