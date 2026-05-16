from __future__ import annotations

from pathlib import Path


CLUSTER_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "cluster"


def test_cluster_scripts_are_static_no_array_jobs() -> None:
    pbs_files = sorted(CLUSTER_ROOT.glob("*/*.pbs"))
    shell_files = sorted(CLUSTER_ROOT.glob("*/*.sh"))

    assert len(pbs_files) == 302
    assert len(shell_files) == 302
    assert (CLUSTER_ROOT / "env.sh").exists()
    for pbs_path in pbs_files:
        text = pbs_path.read_text(encoding="utf-8")
        assert "#PBS -J" not in text
        assert "PBS_ARRAY" not in text
        assert f"bash scripts/cluster/{pbs_path.parent.name}/{pbs_path.stem}.sh" in text
        assert (pbs_path.with_suffix(".sh")).exists()


def test_cluster_scripts_include_required_experiment_families() -> None:
    expected = [
        "caiso/gdsi_xgb_001.pbs",
        "caiso/gdsi_xgb_025.pbs",
        "caiso/joint_dvi_xgb_001_conservative_post.pbs",
        "caiso/joint_dvi_xgb_025_optimistic_ante.pbs",
        "ems/infodva_xgb_001.pbs",
        "ems/infodva_xgb_025.pbs",
        "ems/joint_dvi_active_design_xgb_001_post.pbs",
        "ems/joint_dvi_active_design_xgb_025_ante.pbs",
        "ems/solver_dva_exact_vs_greedy_xgb_001_ante.pbs",
        "ems/solver_dva_exact_vs_naive_xgb_025_post.pbs",
        "ems/compute_benchmark.pbs",
        "ems/kernel_permutation_benchmark.pbs",
    ]

    for relative_path in expected:
        assert (CLUSTER_ROOT / relative_path).exists()


def test_cluster_scripts_use_conservative_resources() -> None:
    for pbs_path in sorted((CLUSTER_ROOT / "caiso").glob("*.pbs")):
        text = pbs_path.read_text(encoding="utf-8")
        assert "#PBS -l select=1:ncpus=2:mem=16gb" in text

    for pbs_path in sorted((CLUSTER_ROOT / "ems").glob("*.pbs")):
        text = pbs_path.read_text(encoding="utf-8")
        assert "#PBS -l select=1:ncpus=2:mem=32gb" in text


def test_cluster_env_uses_uv_frozen_setup_and_thread_guards() -> None:
    env_text = (CLUSTER_ROOT / "env.sh").read_text(encoding="utf-8")

    assert "uv sync --frozen" in env_text
    assert 'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"' in env_text
    assert 'export GUROBI_THREADS="${GUROBI_THREADS:-1}"' in env_text
    assert "mkdir -p logs/cluster results" in env_text
