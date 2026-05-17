from __future__ import annotations

from dva.analysis.run_ems_shap_exhaustive_comparison import (
    _build_setting_config,
    _build_sweep_settings,
    _resolve_model_records,
    _resolve_solvers,
    build_parser,
)


def test_exhaustive_runner_builds_parallel_no_cvar_setting_config(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--solver",
            "lp",
            "--coverage-radius-km",
            "1",
            "--facility-budget",
            "3",
            "--out-root",
            str(tmp_path / "results"),
            "--plot-root",
            str(tmp_path / "plots"),
            "--no-cvar-decision-shap",
            "--n-jobs",
            "6",
        ]
    )
    settings = _build_sweep_settings(
        model_records=_resolve_model_records(["xgb_001"]),
        solvers=_resolve_solvers(args.solver),
        coverage_radii_km=args.coverage_radius_km,
        facility_budgets=args.facility_budget,
        out_root=args.out_root,
        plot_root=args.plot_root,
    )

    assert [setting.setting_id for setting in settings] == [
        "ems_xgb_001_lp_relaxation_radius_1km_budget_3"
    ]

    config = _build_setting_config(settings[0], args=args, solver_seed=17)

    assert config.model_id == "xgb_001"
    assert config.coverage_solver == "lp_relaxation"
    assert config.compute_cvar_decision_shap is False
    assert config.n_jobs == 6
    assert config.solver_seed == 17
    assert config.outdir == (
        tmp_path
        / "results"
        / "models"
        / "xgb_001"
        / "runs"
        / "ems_xgb_001_lp_relaxation_radius_1km_budget_3"
    )
