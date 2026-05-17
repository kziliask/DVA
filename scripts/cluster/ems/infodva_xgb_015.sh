#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"
mkdir -p logs/cluster/ems

uv run dva-ems-infodva --model-id xgb_015 --out-root results/ems/experiment_a_infodva/xgb_015 --plot-root results/ems/experiment_a_infodva_plots/xgb_015 --optimization-solver "${OPTIMIZATION_SOLVER}" --solver-threads "${SOLVER_THREADS}"
