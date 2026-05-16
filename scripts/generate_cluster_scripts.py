from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLUSTER_ROOT = ROOT / "scripts" / "cluster"


@dataclass(frozen=True, slots=True)
class ClusterJob:
    group: str
    name: str
    command: str
    ncpus: int
    mem: str
    walltime: str


def _jobs() -> list[ClusterJob]:
    jobs: list[ClusterJob] = []
    model_ids = tuple(f"xgb_{index:03d}" for index in range(1, 26))
    for model_id in model_ids:
        jobs.append(
            ClusterJob(
                group="caiso",
                name=f"gdsi_{model_id}",
                command=(
                    "uv run dva-caiso-gdsi "
                    f"--model-id {model_id} "
                    f"--outdir results/caiso/gdsi/{model_id}"
                ),
                ncpus=8,
                mem="64gb",
                walltime="48:00:00",
            )
        )
        for baseline in ("conservative", "optimistic"):
            for mode in ("post", "ante"):
                jobs.append(
                    ClusterJob(
                        group="caiso",
                        name=f"joint_dvi_{model_id}_{baseline}_{mode}",
                        command=(
                            "uv run dva-caiso-joint-dvi "
                            f"--model-id {model_id} "
                            f"--baseline {baseline} --value-mode {mode} "
                            f"--outdir results/caiso/joint_dvi/{model_id}/"
                            f"{baseline}_{mode}"
                        ),
                        ncpus=8,
                        mem="64gb",
                        walltime="48:00:00",
                    )
                )
    for model_id in model_ids:
        jobs.append(
            ClusterJob(
                group="ems",
                name=f"infodva_{model_id}",
                command=(
                    "uv run dva-ems-infodva "
                    f"--model-id {model_id} "
                    f"--out-root results/ems/experiment_a_infodva/{model_id} "
                    f"--plot-root results/ems/experiment_a_infodva_plots/{model_id}"
                ),
                ncpus=16,
                mem="128gb",
                walltime="72:00:00",
            )
        )
        for mode in ("post", "ante"):
            jobs.append(
                ClusterJob(
                    group="ems",
                    name=f"joint_dvi_active_design_{model_id}_{mode}",
                    command=(
                        "uv run dva-ems-design-joint-dvi "
                        "--analysis-kind joint_dvi "
                        f"--model-id {model_id} --value-mode {mode} "
                        "--baseline-solver exact --target-solver exact "
                        "--baseline-radius-km 1 --target-radius-km 1 "
                        "--baseline-staging-areas 3 --target-staging-areas 8 "
                        f"--outdir results/ems/experiment_b_joint_dvi/{model_id}/{mode}"
                    ),
                    ncpus=16,
                    mem="128gb",
                    walltime="72:00:00",
                )
            )
            for solver in ("greedy", "naive"):
                jobs.append(
                    ClusterJob(
                        group="ems",
                        name=f"solver_dva_exact_vs_{solver}_{model_id}_{mode}",
                        command=(
                            "uv run dva-ems-design-joint-dvi "
                            "--analysis-kind joint_dvi "
                            f"--model-id {model_id} --value-mode {mode} "
                            f"--baseline-solver exact --target-solver {solver} "
                            "--baseline-radius-km 1 --target-radius-km 1 "
                            "--baseline-staging-areas 8 --target-staging-areas 8 "
                            f"--outdir results/ems/experiment_c_solver_dva/"
                            f"{model_id}/exact_vs_{solver}_{mode}"
                        ),
                        ncpus=16,
                        mem="128gb",
                        walltime="72:00:00",
                    )
                )
    jobs.extend(
        [
            ClusterJob(
                group="ems",
                name="compute_benchmark",
                command="uv run dva-ems-compute-benchmark --outdir results/ems/compute_benchmark",
                ncpus=16,
                mem="128gb",
                walltime="72:00:00",
            ),
            ClusterJob(
                group="ems",
                name="kernel_permutation_benchmark",
                command=(
                    "uv run dva-ems-kernel-permutation-benchmark "
                    "--outdir results/ems/kernel_permutation_benchmark"
                ),
                ncpus=16,
                mem="128gb",
                walltime="72:00:00",
            ),
        ]
    )
    return jobs


def _write_env() -> None:
    CLUSTER_ROOT.mkdir(parents=True, exist_ok=True)
    (CLUSTER_ROOT / "env.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${DVA_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_ROOT}"

mkdir -p logs/cluster results
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export GUROBI_THREADS="${GUROBI_THREADS:-1}"

uv sync --frozen
""",
        encoding="utf-8",
    )
    (CLUSTER_ROOT / "env.sh").chmod(0o755)


def _write_job(job: ClusterJob) -> None:
    job_dir = CLUSTER_ROOT / job.group
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = f"logs/cluster/{job.group}"
    shell_path = job_dir / f"{job.name}.sh"
    pbs_path = job_dir / f"{job.name}.pbs"
    shell_path.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
source "${{SCRIPT_DIR}}/../env.sh"
mkdir -p {log_dir}

{job.command}
""",
        encoding="utf-8",
    )
    shell_path.chmod(0o755)
    pbs_path.write_text(
        f"""#!/usr/bin/env bash
#PBS -N dva_{job.name}
#PBS -l select=1:ncpus={job.ncpus}:mem={job.mem}
#PBS -l walltime={job.walltime}
#PBS -o {log_dir}/{job.name}.out
#PBS -e {log_dir}/{job.name}.err

set -euo pipefail
cd "${{PBS_O_WORKDIR:-$(pwd)}}"
bash scripts/cluster/{job.group}/{job.name}.sh
""",
        encoding="utf-8",
    )
    pbs_path.chmod(0o755)


def main() -> None:
    _write_env()
    for job in _jobs():
        _write_job(job)


if __name__ == "__main__":
    main()
