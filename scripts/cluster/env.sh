#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${DVA_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_ROOT}"

mkdir -p logs/cluster results
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export SOLVER_THREADS="${SOLVER_THREADS:-${GUROBI_THREADS:-1}}"
export GUROBI_THREADS="${GUROBI_THREADS:-${SOLVER_THREADS}}"
export OPTIMIZATION_SOLVER="${OPTIMIZATION_SOLVER:-highs}"

UV_SYNC_ARGS=(--frozen)
if [[ "${OPTIMIZATION_SOLVER}" == "gurobi" ]]; then
    UV_SYNC_ARGS+=(--extra gurobi)
fi
uv sync "${UV_SYNC_ARGS[@]}"
