#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"
mkdir -p logs/cluster/caiso

uv run dva-caiso-joint-dvi --model-id xgb_020 --target optimistic --value-mode post --outdir results/caiso/joint_dvi_flipped/xgb_020/optimistic_post
