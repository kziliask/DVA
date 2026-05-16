#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"
mkdir -p logs/cluster/ems

uv run dva-ems-design-joint-dvi --analysis-kind joint_dvi --model-id xgb_017 --value-mode post --baseline-solver exact --target-solver greedy --baseline-radius-km 1 --target-radius-km 1 --baseline-staging-areas 8 --target-staging-areas 8 --outdir results/ems/experiment_c_solver_dva/xgb_017/exact_vs_greedy_post
