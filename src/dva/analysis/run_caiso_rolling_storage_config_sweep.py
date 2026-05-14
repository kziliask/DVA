from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dva.analysis.run_caiso_decision_shap_guided_validation import (
    DEFAULT_BACKGROUND_DAYS,
    DEFAULT_RANDOM_MASK_SEED,
)
from dva.analysis.run_caiso_decision_shap_guided_validation_rolling_windows import (
    DEFAULT_ROLLING_OUTDIR,
    DEFAULT_FOLD_COUNT,
    DEFAULT_STEP_MONTHS,
    DEFAULT_TEST_MONTHS,
    DEFAULT_TRAIN_MONTHS,
    DEFAULT_VALIDATION_MONTHS,
)
from dva.model.train import DEFAULT_DATASET_PATH


DEFAULT_OUTDIR_ROOT = DEFAULT_ROLLING_OUTDIR.with_name(
    "caiso_decision_shap_guided_validation_rolling_storage_configs"
)
DEFAULT_STORAGE_SWEEP_BOOTSTRAP_REPLICATES = 10000


@dataclass(frozen=True, slots=True)
class StorageConfigSpec:
    slug: str
    label: str
    throughput_penalty: float
    charge_efficiency: float
    discharge_efficiency: float
    power_limit: float
    energy_capacity: float
    initial_state_of_charge: float
    terminal_state_of_charge: float
    purpose: str


STORAGE_CONFIGS: tuple[StorageConfigSpec, ...] = (
    StorageConfigSpec(
        slug="base",
        label="Base",
        throughput_penalty=5.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        power_limit=1.0,
        energy_capacity=4.0,
        initial_state_of_charge=2.0,
        terminal_state_of_charge=2.0,
        purpose="Anchor / current paper setting",
    ),
    StorageConfigSpec(
        slug="small_battery",
        label="Small battery",
        throughput_penalty=5.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        power_limit=1.0,
        energy_capacity=2.0,
        initial_state_of_charge=1.0,
        terminal_state_of_charge=1.0,
        purpose="Capacity-constrained; fewer spreads can be exploited",
    ),
    StorageConfigSpec(
        slug="large_battery",
        label="Large battery",
        throughput_penalty=5.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        power_limit=1.0,
        energy_capacity=8.0,
        initial_state_of_charge=4.0,
        terminal_state_of_charge=4.0,
        purpose=(
            "More flexible storage; tests whether harmful features persist when "
            "capacity is relaxed"
        ),
    ),
    StorageConfigSpec(
        slug="low_efficiency_battery",
        label="Low-efficiency battery",
        throughput_penalty=5.0,
        charge_efficiency=0.85,
        discharge_efficiency=0.85,
        power_limit=1.0,
        energy_capacity=4.0,
        initial_state_of_charge=2.0,
        terminal_state_of_charge=2.0,
        purpose="Requires larger price spreads; filters marginal trades",
    ),
    StorageConfigSpec(
        slug="low_throughput",
        label="Low-Throughput",
        throughput_penalty=10.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        power_limit=1.0,
        energy_capacity=4.0,
        initial_state_of_charge=2.0,
        terminal_state_of_charge=2.0,
        purpose="Higher throughput penalty",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CAISO rolling GDSI/SAGE experiment for the five storage "
            "parameter configurations."
        )
    )
    parser.add_argument("--outdir-root", type=Path, default=DEFAULT_OUTDIR_ROOT)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--model-family",
        choices=("all", "xgb", "nn", "torch_mlp"),
        default="xgb",
    )
    parser.add_argument("--model-id", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--config",
        action="append",
        choices=tuple(spec.slug for spec in STORAGE_CONFIGS),
        default=None,
        help="Run only selected storage configuration slug(s).",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_STORAGE_SWEEP_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--background-days", type=int, default=DEFAULT_BACKGROUND_DAYS)
    parser.add_argument("--random-mask-seed", type=int, default=DEFAULT_RANDOM_MASK_SEED)
    parser.add_argument("--training-verbose", action="store_true")
    parser.add_argument("--skip-sage-ablation", action="store_true")
    parser.add_argument("--fold-count", type=int, default=DEFAULT_FOLD_COUNT)
    parser.add_argument("--train-months", type=int, default=DEFAULT_TRAIN_MONTHS)
    parser.add_argument(
        "--validation-months",
        type=int,
        default=DEFAULT_VALIDATION_MONTHS,
    )
    parser.add_argument("--test-months", type=int, default=DEFAULT_TEST_MONTHS)
    parser.add_argument("--step-months", type=int, default=DEFAULT_STEP_MONTHS)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--validation-max-days", type=int, default=None)
    parser.add_argument("--test-max-days", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected_slugs = set(args.config or [spec.slug for spec in STORAGE_CONFIGS])
    configs = [spec for spec in STORAGE_CONFIGS if spec.slug in selected_slugs]
    if not configs:
        raise ValueError("No storage configurations selected.")

    args.outdir_root.mkdir(parents=True, exist_ok=True)
    for spec in configs:
        command = _build_runner_command(args, spec)
        print(f"\n=== {spec.label} ({spec.slug}) ===", flush=True)
        print(spec.purpose, flush=True)
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


def _build_runner_command(
    args: argparse.Namespace,
    spec: StorageConfigSpec,
) -> list[str]:
    command = [
        "uv",
        "run",
        "src/analysis/run_caiso_decision_shap_guided_validation_rolling_windows.py",
        "--outdir",
        str(args.outdir_root / spec.slug),
        "--dataset-path",
        str(args.dataset_path),
        "--max-workers",
        str(args.max_workers),
        "--model-family",
        str(args.model_family),
        "--bootstrap-replicates",
        str(args.bootstrap_replicates),
        "--background-days",
        str(args.background_days),
        "--random-mask-seed",
        str(args.random_mask_seed),
        "--fold-count",
        str(args.fold_count),
        "--train-months",
        str(args.train_months),
        "--validation-months",
        str(args.validation_months),
        "--test-months",
        str(args.test_months),
        "--step-months",
        str(args.step_months),
        "--throughput-penalty",
        str(spec.throughput_penalty),
        "--charge-efficiency",
        str(spec.charge_efficiency),
        "--discharge-efficiency",
        str(spec.discharge_efficiency),
        "--power-limit",
        str(spec.power_limit),
        "--energy-capacity",
        str(spec.energy_capacity),
        "--initial-state-of-charge",
        str(spec.initial_state_of_charge),
        "--terminal-state-of-charge",
        str(spec.terminal_state_of_charge),
    ]
    for model_id in args.model_id or ():
        command.extend(["--model-id", str(model_id)])
    if args.start_date is not None:
        command.extend(["--start-date", str(args.start_date)])
    if args.validation_max_days is not None:
        command.extend(["--validation-max-days", str(args.validation_max_days)])
    if args.test_max_days is not None:
        command.extend(["--test-max-days", str(args.test_max_days)])
    if args.overwrite:
        command.append("--overwrite")
    if args.training_verbose:
        command.append("--training-verbose")
    if args.skip_sage_ablation:
        command.append("--skip-sage-ablation")
    return command


if __name__ == "__main__":
    main()
