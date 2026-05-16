#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"
mkdir -p logs/cluster/ems

uv run dva-ems-infodva --model-id xgb_005 --out-root results/ems/experiment_a_infodva/xgb_005 --plot-root results/ems/experiment_a_infodva_plots/xgb_005
