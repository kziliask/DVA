#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"
mkdir -p logs/cluster/ems

uv run dva-ems-design-joint-dvi --analysis-kind joint_dvi --model-id xgb_013 --value-mode post --baseline-solver exact --target-solver exact --baseline-radius-km 1 --target-radius-km 3 --baseline-staging-areas 3 --target-staging-areas 3 --optimization-solver "${OPTIMIZATION_SOLVER}" --solver-threads "${SOLVER_THREADS}" --outdir results/ems/experiment_b_joint_dvi/xgb_013/p3_tau3/post
