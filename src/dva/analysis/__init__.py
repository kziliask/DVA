from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "CaisoSweepComparisonOutputs": "dva.analysis.caiso_sweep",
    "SweepManifestEntry": "dva.analysis.caiso_sweep",
    "compare_caiso_shap_sweeps": "dva.analysis.caiso_sweep",
    "load_sweep_manifest": "dva.analysis.caiso_sweep",
    "write_caiso_sweep_comparison_outputs": "dva.analysis.caiso_sweep",
    "CASE_STUDY_REQUIRED_OUTPUT_FILENAMES": "dva.analysis.caiso_sweep_runs",
    "CaisoSweepRunEntry": "dva.analysis.caiso_sweep_runs",
    "case_study_outputs_are_complete": "dva.analysis.caiso_sweep_runs",
    "load_caiso_shap_case_study_sweep_manifest": "dva.analysis.caiso_sweep_runs",
    "CaisoRegretShapCaseStudyConfig": "dva.analysis.caiso_regret_shap",
    "CaisoRegretShapCaseStudyOutputs": "dva.analysis.caiso_regret_shap",
    "resolve_regret_model_name": "dva.analysis.caiso_regret_shap",
    "run_caiso_regret_shap_case_study": "dva.analysis.caiso_regret_shap",
    "write_caiso_regret_shap_case_study_outputs": "dva.analysis.caiso_regret_shap",
    "CaisoShapCaseStudyConfig": "dva.analysis.caiso_shap",
    "CaisoShapCaseStudyOutputs": "dva.analysis.caiso_shap",
    "DEFAULT_INTERACTION_METHOD": "dva.analysis.caiso_shap",
    "DailyInteractionExplanation": "dva.analysis.caiso_shap",
    "DailyShapExplanation": "dva.analysis.caiso_shap",
    "DailyShapleyTaylorExplanation": "dva.analysis.caiso_shap",
    "ExtendedPlayerCoalitionEvaluator": "dva.analysis.caiso_shap",
    "ParameterPlayerSpec": "dva.analysis.caiso_shap",
    "SUPPORTED_INTERACTION_METHODS": "dva.analysis.caiso_shap",
    "build_default_storage_parameters": "dva.analysis.caiso_shap",
    "compute_exact_faith_shap_values": "dva.analysis.caiso_shap",
    "compute_exact_interaction_values": "dva.analysis.caiso_shap",
    "compute_exact_shapley_taylor_values": "dva.analysis.caiso_shap",
    "compute_mobius_transform": "dva.analysis.caiso_shap",
    "run_caiso_shap_case_study": "dva.analysis.caiso_shap",
    "run_caiso_shap_case_study_with_artifacts": "dva.analysis.caiso_shap",
    "write_caiso_shap_case_study_outputs": "dva.analysis.caiso_shap",
    "EmsExactShapConfig": "dva.analysis.ems_exact_shap",
    "EmsExactShapOutputs": "dva.analysis.ems_exact_shap",
    "EmsFeatureGroup": "dva.analysis.ems_exact_shap",
    "EMS_COVERAGE_SOLVER_GREEDY_MAX_COVER": "dva.analysis.ems_exact_shap",
    "EMS_COVERAGE_SOLVER_GUROBI": "dva.analysis.ems_exact_shap",
    "EMS_COVERAGE_SOLVER_GUROBI_LP_RELAXATION": "dva.analysis.ems_exact_shap",
    "EMS_COVERAGE_SOLVER_EXACT": "dva.analysis.ems_exact_shap",
    "EMS_COVERAGE_SOLVER_LP_RELAXATION": "dva.analysis.ems_exact_shap",
    "EMS_COVERAGE_SOLVER_NAIVE_GREEDY": "dva.analysis.ems_exact_shap",
    "GroupedBackgroundCoalitionPredictor": "dva.analysis.ems_exact_shap",
    "MaximumCoverageResult": "dva.analysis.ems_exact_shap",
    "SUPPORTED_EMS_COVERAGE_SOLVERS": "dva.analysis.ems_exact_shap",
    "build_coverage_matrix": "dva.analysis.ems_exact_shap",
    "build_ems_feature_groups": "dva.analysis.ems_exact_shap",
    "load_ems_exact_shap_outputs": "dva.analysis.ems_exact_shap",
    "normalize_ems_coverage_solver": "dva.analysis.ems_exact_shap",
    "run_ems_exact_shap": "dva.analysis.ems_exact_shap",
    "solve_cvar_coverage": "dva.analysis.ems_exact_shap",
    "solve_ems_coverage": "dva.analysis.ems_exact_shap",
    "solve_greedy_max_cover_coverage": "dva.analysis.ems_exact_shap",
    "solve_gurobi_lp_relaxation_coverage": "dva.analysis.ems_exact_shap",
    "solve_lp_relaxation_coverage": "dva.analysis.ems_exact_shap",
    "solve_maximum_coverage": "dva.analysis.ems_exact_shap",
    "solve_naive_greedy_coverage": "dva.analysis.ems_exact_shap",
    "write_ems_exact_shap_outputs": "dva.analysis.ems_exact_shap",
    "DailyDispatchPolicy": "dva.analysis.evaluation_metrics",
    "build_attribution_ranking": "dva.analysis.evaluation_metrics",
    "compute_decision_deletion_auc": "dva.analysis.evaluation_metrics",
    "compute_decision_insertion_auc": "dva.analysis.evaluation_metrics",
    "compute_exact_decision_infidelity": "dva.analysis.evaluation_metrics",
    "compute_global_importance": "dva.analysis.evaluation_metrics",
    "compute_normalized_importance_l1": "dva.analysis.evaluation_metrics",
    "compute_rank_kendall_tau_from_rankings": "dva.analysis.evaluation_metrics",
    "compute_rank_spearman_from_rankings": "dva.analysis.evaluation_metrics",
    "compute_top_k_jaccard": "dva.analysis.evaluation_metrics",
    "compute_truncated_rbo": "dva.analysis.evaluation_metrics",
    "identify_invariant_policy_days": "dva.analysis.evaluation_metrics",
    "rank_features_from_scores": "dva.analysis.evaluation_metrics",
    "BootstrapConfig": "dva.analysis.paired_bootstrap",
    "bootstrap_metric_table": "dva.analysis.paired_bootstrap",
    "infer_metric_direction": "dva.analysis.paired_bootstrap",
    "wide_metric_frame": "dva.analysis.paired_bootstrap",
}

_SHARED_EXPORTS = {
    "compute_exact_shapley_values": (
        "dva.analysis.caiso_shap",
        "dva.analysis.ems_exact_shap",
    ),
}

__all__ = sorted([*_EXPORT_MODULES, *_SHARED_EXPORTS])


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MODULES:
        module = import_module(_EXPORT_MODULES[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SHARED_EXPORTS:
        for module_name in _SHARED_EXPORTS[name]:
            module = import_module(module_name)
            if hasattr(module, name):
                value = getattr(module, name)
                globals()[name] = value
                return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
