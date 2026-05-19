#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"
mkdir -p logs/cluster/ems

uv run dva-ems-design-joint-dvi --analysis-kind joint_dvi --model-id xgb_009 --value-mode post --baseline-solver exact --target-solver exact --baseline-radius-km 1 --target-radius-km 2 --baseline-staging-areas 3 --target-staging-areas 5 --optimization-solver "${OPTIMIZATION_SOLVER}" --solver-threads "${SOLVER_THREADS}" --outdir results/ems/experiment_b_joint_dvi/xgb_009/p5_tau2/post
