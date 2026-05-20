#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"
mkdir -p logs/cluster/caiso

uv run dva-caiso-joint-dvi --model-id xgb_006 --target optimistic --value-mode post --outdir results/caiso/joint_dvi_flipped/xgb_006/optimistic_post
