#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"
mkdir -p logs/cluster/ems

uv run dva-ems-design-joint-dvi --analysis-kind joint_dvi --model-id xgb_016 --value-mode post --baseline-solver exact --target-solver exact --baseline-radius-km 1 --target-radius-km 1 --baseline-staging-areas 3 --target-staging-areas 8 --outdir results/ems/experiment_b_joint_dvi/xgb_016/post
