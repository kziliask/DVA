#!/usr/bin/env bash

# No "set -e" here: we want to keep going if one qsub fails.
set -u
shopt -s nullglob

mkdir -p logs/cluster/submission

submitted_log="logs/cluster/submission/submitted_ems.tsv"
failed_log="logs/cluster/submission/failed_ems.tsv"
active_names="$(mktemp)"

# Capture names of jobs already queued/running.
qstat -u "$USER" -f 2>/dev/null \
  | awk -F'= ' '/Job_Name =/ {print $2}' \
  | sort -u > "$active_names"

jobs=(
  scripts/cluster/ems/infodva_*.pbs
  scripts/cluster/ems/joint_dvi_active_design_*.pbs
  scripts/cluster/ems/solver_dva_*.pbs
)

echo "Found ${#jobs[@]} EMS PBS files"

for job in "${jobs[@]}"; do
  job_name="$(awk '/^#PBS -N / {print $3; exit}' "$job")"

  if grep -Fxq "$job_name" "$active_names"; then
    echo "SKIP already queued/running: $job_name"
    continue
  fi

  echo "Submitting: $job_name from $job"

  if out="$(qsub -q medium "$job" 2>&1)"; then
    printf '%s\t%s\t%s\n' "$out" "$job_name" "$job" | tee -a "$submitted_log"
  else
    rc=$?
    printf 'rc=%s\t%s\t%s\t%s\n' "$rc" "$job_name" "$job" "$out" | tee -a "$failed_log" >&2
  fi

  sleep 0.2
done

rm -f "$active_names"

echo "Done."
echo "Submitted log: $submitted_log"
echo "Failed log:    $failed_log"
