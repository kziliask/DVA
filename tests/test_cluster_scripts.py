from __future__ import annotations

from pathlib import Path


CLUSTER_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "cluster"


def test_cluster_scripts_are_static_no_array_jobs() -> None:
    pbs_files = sorted(CLUSTER_ROOT.glob("*/*.pbs"))
    shell_files = sorted(CLUSTER_ROOT.glob("*/*.sh"))

    assert len(pbs_files) == 827
    assert len(shell_files) == 827
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
        "caiso/joint_dvi_flipped_xgb_001_conservative_post.pbs",
        "caiso/joint_dvi_flipped_xgb_025_optimistic_ante.pbs",
        "ems/infodva_xgb_001.pbs",
        "ems/infodva_xgb_025.pbs",
        "ems/joint_dvi_p3_tau1_xgb_001_post.pbs",
        "ems/joint_dvi_p5_tau2_xgb_013_ante.pbs",
        "ems/joint_dvi_p8_tau3_xgb_025_post.pbs",
        "ems/joint_dvi_active_design_xgb_001_post.pbs",
        "ems/joint_dvi_active_design_xgb_025_ante.pbs",
        "ems/solver_dva_exact_vs_greedy_xgb_001_ante.pbs",
        "ems/solver_dva_exact_vs_naive_xgb_025_post.pbs",
        "ems/design_utility_dva_xgb_001.pbs",
        "ems/design_utility_dva_xgb_025.pbs",
        "ems/compute_benchmark.pbs",
        "ems/kernel_permutation_benchmark.pbs",
    ]

    for relative_path in expected:
        assert (CLUSTER_ROOT / relative_path).exists()


def test_cluster_scripts_use_conservative_resources() -> None:
    for pbs_path in sorted((CLUSTER_ROOT / "caiso").glob("*.pbs")):
        text = pbs_path.read_text(encoding="utf-8")
        assert "#PBS -l select=1:ncpus=1:mem=16gb" in text

    for pbs_path in sorted((CLUSTER_ROOT / "ems").glob("*.pbs")):
        text = pbs_path.read_text(encoding="utf-8")
        assert "#PBS -l select=1:ncpus=1:mem=16gb" in text


def test_cluster_env_uses_uv_frozen_setup_and_thread_guards() -> None:
    env_text = (CLUSTER_ROOT / "env.sh").read_text(encoding="utf-8")

    assert "UV_SYNC_ARGS=(--frozen)" in env_text
    assert 'UV_SYNC_ARGS+=(--extra gurobi)' in env_text
    assert 'uv sync "${UV_SYNC_ARGS[@]}"' in env_text
    assert 'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"' in env_text
    assert 'export SOLVER_THREADS="${SOLVER_THREADS:-${GUROBI_THREADS:-1}}"' in env_text
    assert 'export GUROBI_THREADS="${GUROBI_THREADS:-${SOLVER_THREADS}}"' in env_text
    assert 'export OPTIMIZATION_SOLVER="${OPTIMIZATION_SOLVER:-highs}"' in env_text
    assert "mkdir -p logs/cluster results" in env_text


def test_ems_cluster_scripts_pass_pyomo_solver_controls() -> None:
    for shell_path in sorted((CLUSTER_ROOT / "ems").glob("*.sh")):
        text = shell_path.read_text(encoding="utf-8")
        assert '--optimization-solver "${OPTIMIZATION_SOLVER}"' in text
        assert '--solver-threads "${SOLVER_THREADS}"' in text


def test_caiso_flipped_joint_dvi_scripts_use_target_orientation() -> None:
    shell_path = CLUSTER_ROOT / "caiso" / "joint_dvi_flipped_xgb_001_optimistic_ante.sh"
    text = shell_path.read_text(encoding="utf-8")

    assert "uv run dva-caiso-joint-dvi" in text
    assert "--target optimistic --value-mode ante" in text
    assert "--baseline" not in text
    assert (
        "--outdir results/caiso/joint_dvi_flipped/xgb_001/optimistic_ante"
        in text
    )


def test_ems_joint_dvi_grid_uses_fixed_baseline_design() -> None:
    shell_path = CLUSTER_ROOT / "ems" / "joint_dvi_p5_tau3_xgb_001_ante.sh"
    text = shell_path.read_text(encoding="utf-8")

    assert "--baseline-solver exact --target-solver exact" in text
    assert "--baseline-radius-km 1 --target-radius-km 3" in text
    assert "--baseline-staging-areas 3 --target-staging-areas 5" in text
    assert (
        "--outdir results/ems/experiment_b_joint_dvi/xgb_001/p5_tau3/ante"
        in text
    )
