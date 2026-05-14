from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import numpy as np
import pandas as pd

from dva.analysis.evaluation_metrics import (
    DEFAULT_POLICY_ATOL,
    DEFAULT_RBO_DEPTH,
    DEFAULT_RBO_P,
    DEFAULT_TOP_K,
    compute_global_importance,
    compute_kendall_tau_correlation,
    compute_normalized_importance_l1,
    compute_rank_kendall_tau_from_rankings,
    compute_rank_spearman_from_rankings,
    compute_spearman_rank_correlation,
    compute_top_k_jaccard,
    compute_truncated_rbo,
    identify_invariant_policy_days,
    rank_features_from_scores,
)


DEFAULT_SWEEP_OUTPUT_DIR = Path("results/caiso_shap_sweep_comparison")
DEFAULT_MIN_INVARIANT_DAYS = 10


@dataclass(frozen=True, slots=True)
class SweepManifestEntry:
    setting_id: str
    results_dir: Path
    sweep_id: str
    parameter_name: str
    parameter_value: float
    step_index: int
    step_label: str


@dataclass(frozen=True, slots=True)
class SweepSettingArtifacts:
    manifest_entry: SweepManifestEntry
    daily_shap: pd.DataFrame
    daily_full_dispatch: pd.DataFrame
    summary_shap: pd.DataFrame
    evaluation_metrics: dict[str, Any]
    run_metadata: dict[str, Any]
    feature_names: tuple[str, ...]
    explain_dates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CaisoSweepComparisonOutputs:
    pairwise_rank_metrics: pd.DataFrame
    setting_metric_summary: pd.DataFrame
    sweep_trend_metrics: dict[str, Any]
    metadata: dict[str, Any]


def load_sweep_manifest(
    manifest_path: Path | str,
) -> list[SweepManifestEntry]:
    manifest_path_obj = Path(manifest_path)
    manifest_frame = pd.read_csv(manifest_path_obj)
    required_columns = {
        "setting_id",
        "results_dir",
        "sweep_id",
        "parameter_name",
        "parameter_value",
        "step_index",
        "step_label",
    }
    missing_columns = sorted(required_columns - set(manifest_frame.columns))
    if missing_columns:
        raise KeyError(
            "Sweep manifest is missing required columns: "
            + ", ".join(missing_columns)
        )
    if manifest_frame.empty:
        raise ValueError("Sweep manifest must contain at least one setting.")

    if manifest_frame["setting_id"].duplicated().any():
        duplicate_ids = sorted(
            manifest_frame.loc[manifest_frame["setting_id"].duplicated(), "setting_id"].unique()
        )
        raise ValueError("setting_id values must be unique: " + ", ".join(duplicate_ids))

    manifest_frame = manifest_frame.copy()
    manifest_frame["parameter_value"] = pd.to_numeric(
        manifest_frame["parameter_value"],
        errors="raise",
    )
    manifest_frame["step_index"] = pd.to_numeric(
        manifest_frame["step_index"],
        errors="raise",
    )
    if not np.allclose(
        manifest_frame["step_index"],
        np.round(manifest_frame["step_index"]),
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError("step_index must be integer-valued for every manifest row.")
    manifest_frame["step_index"] = manifest_frame["step_index"].astype(int)
    manifest_frame["setting_id"] = manifest_frame["setting_id"].astype(str)
    manifest_frame["sweep_id"] = manifest_frame["sweep_id"].astype(str)
    manifest_frame["parameter_name"] = manifest_frame["parameter_name"].astype(str)
    manifest_frame["step_label"] = manifest_frame["step_label"].astype(str)

    entries: list[SweepManifestEntry] = []
    for row in manifest_frame.to_dict(orient="records"):
        row_dict = cast(dict[str, Any], row)
        results_dir = Path(str(row_dict["results_dir"]))
        if not results_dir.is_absolute():
            results_dir = (manifest_path_obj.parent / results_dir).resolve()
        entries.append(
            SweepManifestEntry(
                setting_id=str(row_dict["setting_id"]),
                results_dir=results_dir,
                sweep_id=str(row_dict["sweep_id"]),
                parameter_name=str(row_dict["parameter_name"]),
                parameter_value=float(row_dict["parameter_value"]),
                step_index=int(row_dict["step_index"]),
                step_label=str(row_dict["step_label"]),
            )
        )

    _validate_manifest_entries(entries)
    return sorted(entries, key=lambda entry: (entry.sweep_id, entry.step_index))


def compare_caiso_shap_sweeps(
    manifest_path: Path | str,
    *,
    min_invariant_days: int = DEFAULT_MIN_INVARIANT_DAYS,
) -> CaisoSweepComparisonOutputs:
    if min_invariant_days <= 0:
        raise ValueError("min_invariant_days must be strictly positive.")

    manifest_entries = load_sweep_manifest(manifest_path)
    artifacts_by_setting = {
        entry.setting_id: _load_setting_artifacts(
            entry,
            manifest_path=manifest_path,
        )
        for entry in manifest_entries
    }

    pairwise_rows: list[dict[str, Any]] = []
    for sweep_entries in _group_entries_by_sweep(manifest_entries).values():
        for left_entry, right_entry in zip(sweep_entries, sweep_entries[1:]):
            left_artifacts = artifacts_by_setting[left_entry.setting_id]
            right_artifacts = artifacts_by_setting[right_entry.setting_id]
            _validate_setting_pair(left_artifacts, right_artifacts)
            invariant_dates = identify_invariant_policy_days(
                left_artifacts.daily_full_dispatch,
                right_artifacts.daily_full_dispatch,
                atol=DEFAULT_POLICY_ATOL,
            )
            pairwise_rows.extend(
                _build_pairwise_rows(
                    left_artifacts=left_artifacts,
                    right_artifacts=right_artifacts,
                    invariant_dates=invariant_dates,
                    min_invariant_days=min_invariant_days,
                )
            )

    pairwise_rank_metrics = pd.DataFrame(pairwise_rows)
    if not pairwise_rank_metrics.empty:
        pairwise_rank_metrics = pairwise_rank_metrics.sort_values(
            ["sweep_id", "left_step_index", "explainer_family"],
        ).reset_index(drop=True)

    setting_metric_summary = _build_setting_metric_summary(
        manifest_entries,
        artifacts_by_setting,
    )
    sweep_trend_metrics = _build_sweep_trend_metrics(setting_metric_summary)
    metadata = {
        "manifest_path": str(Path(manifest_path)),
        "min_invariant_days": min_invariant_days,
        "top_k": DEFAULT_TOP_K,
        "rbo_depth": DEFAULT_RBO_DEPTH,
        "rbo_p": DEFAULT_RBO_P,
        "policy_atol": DEFAULT_POLICY_ATOL,
    }
    return CaisoSweepComparisonOutputs(
        pairwise_rank_metrics=pairwise_rank_metrics,
        setting_metric_summary=setting_metric_summary,
        sweep_trend_metrics=sweep_trend_metrics,
        metadata=metadata,
    )


def write_caiso_sweep_comparison_outputs(
    outputs: CaisoSweepComparisonOutputs,
    outdir: Path | str = DEFAULT_SWEEP_OUTPUT_DIR,
) -> None:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    outputs.pairwise_rank_metrics.to_csv(
        outdir_path / "pairwise_rank_metrics.csv",
        index=False,
    )
    outputs.setting_metric_summary.to_csv(
        outdir_path / "setting_metric_summary.csv",
        index=False,
    )

    with (outdir_path / "pairwise_rank_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": outputs.metadata,
                "rows": _frame_records_for_json(outputs.pairwise_rank_metrics),
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    with (outdir_path / "sweep_trend_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(outputs.sweep_trend_metrics, handle, indent=2, sort_keys=True)


def _build_pairwise_rows(
    *,
    left_artifacts: SweepSettingArtifacts,
    right_artifacts: SweepSettingArtifacts,
    invariant_dates: Sequence[str],
    min_invariant_days: int,
) -> list[dict[str, Any]]:
    pairwise_rows: list[dict[str, Any]] = []
    invariant_day_count = len(invariant_dates)
    total_days = len(left_artifacts.explain_dates)
    invariant_day_coverage = float(invariant_day_count / total_days) if total_days else None

    for explainer_family in sorted(left_artifacts.evaluation_metrics):
        left_global_importance = compute_global_importance(
            left_artifacts.daily_shap,
            left_artifacts.feature_names,
            explainer_family,
        )
        right_global_importance = compute_global_importance(
            right_artifacts.daily_shap,
            right_artifacts.feature_names,
            explainer_family,
        )
        left_ranking = rank_features_from_scores(
            left_global_importance,
            left_artifacts.feature_names,
        )
        right_ranking = rank_features_from_scores(
            right_global_importance,
            right_artifacts.feature_names,
        )

        decision_invariant_rbo = None
        decision_invariant_l1 = None
        if invariant_day_count >= min_invariant_days:
            left_invariant_importance = compute_global_importance(
                left_artifacts.daily_shap,
                left_artifacts.feature_names,
                explainer_family,
                dates=invariant_dates,
            )
            right_invariant_importance = compute_global_importance(
                right_artifacts.daily_shap,
                right_artifacts.feature_names,
                explainer_family,
                dates=invariant_dates,
            )
            left_invariant_ranking = rank_features_from_scores(
                left_invariant_importance,
                left_artifacts.feature_names,
            )
            right_invariant_ranking = rank_features_from_scores(
                right_invariant_importance,
                right_artifacts.feature_names,
            )
            decision_invariant_rbo = compute_truncated_rbo(
                left_invariant_ranking,
                right_invariant_ranking,
                depth=DEFAULT_RBO_DEPTH,
                p=DEFAULT_RBO_P,
            )
            decision_invariant_l1 = compute_normalized_importance_l1(
                left_invariant_importance,
                right_invariant_importance,
                left_artifacts.feature_names,
            )

        pairwise_rows.append(
            {
                "sweep_id": left_artifacts.manifest_entry.sweep_id,
                "parameter_name": left_artifacts.manifest_entry.parameter_name,
                "left_setting_id": left_artifacts.manifest_entry.setting_id,
                "right_setting_id": right_artifacts.manifest_entry.setting_id,
                "left_parameter_value": left_artifacts.manifest_entry.parameter_value,
                "right_parameter_value": right_artifacts.manifest_entry.parameter_value,
                "left_step_index": left_artifacts.manifest_entry.step_index,
                "right_step_index": right_artifacts.manifest_entry.step_index,
                "left_step_label": left_artifacts.manifest_entry.step_label,
                "right_step_label": right_artifacts.manifest_entry.step_label,
                "explainer_family": explainer_family,
                "feature_count": len(left_artifacts.feature_names),
                "top_k_jaccard_5": compute_top_k_jaccard(
                    left_ranking,
                    right_ranking,
                    k=DEFAULT_TOP_K,
                ),
                "rbo_10": compute_truncated_rbo(
                    left_ranking,
                    right_ranking,
                    depth=DEFAULT_RBO_DEPTH,
                    p=DEFAULT_RBO_P,
                ),
                "rank_spearman": compute_rank_spearman_from_rankings(
                    left_ranking,
                    right_ranking,
                ),
                "rank_kendall_tau": compute_rank_kendall_tau_from_rankings(
                    left_ranking,
                    right_ranking,
                ),
                "normalized_importance_l1": compute_normalized_importance_l1(
                    left_global_importance,
                    right_global_importance,
                    left_artifacts.feature_names,
                ),
                "invariant_day_count": invariant_day_count,
                "invariant_day_coverage": invariant_day_coverage,
                "min_invariant_days": min_invariant_days,
                "decision_invariant_rbo_10": decision_invariant_rbo,
                "decision_invariant_normalized_importance_l1": decision_invariant_l1,
            }
        )

    return pairwise_rows


def _build_setting_metric_summary(
    manifest_entries: Sequence[SweepManifestEntry],
    artifacts_by_setting: dict[str, SweepSettingArtifacts],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in manifest_entries:
        artifacts = artifacts_by_setting[entry.setting_id]
        for explainer_family, metrics_by_name in sorted(artifacts.evaluation_metrics.items()):
            insertion_auc_metrics = metrics_by_name["decision_insertion_auc"]
            deletion_auc_metrics = metrics_by_name.get("decision_deletion_auc", {})
            decision_infidelity_metrics = metrics_by_name["decision_infidelity"]
            rows.append(
                {
                    "setting_id": entry.setting_id,
                    "sweep_id": entry.sweep_id,
                    "parameter_name": entry.parameter_name,
                    "parameter_value": entry.parameter_value,
                    "step_index": entry.step_index,
                    "step_label": entry.step_label,
                    "explainer_family": explainer_family,
                    "decision_insertion_auc_mean": insertion_auc_metrics["mean"],
                    "decision_insertion_auc_median": insertion_auc_metrics["median"],
                    "decision_insertion_auc_coverage": insertion_auc_metrics["coverage"],
                    "decision_deletion_auc_mean": deletion_auc_metrics.get("mean"),
                    "decision_deletion_auc_median": deletion_auc_metrics.get("median"),
                    "decision_deletion_auc_coverage": deletion_auc_metrics.get("coverage"),
                    "decision_infidelity_mean": decision_infidelity_metrics["mean"],
                    "decision_infidelity_median": decision_infidelity_metrics["median"],
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["sweep_id", "step_index", "explainer_family"],
    ).reset_index(drop=True)


def _build_sweep_trend_metrics(
    setting_metric_summary: pd.DataFrame,
) -> dict[str, Any]:
    sweep_trend_metrics: dict[str, Any] = {}
    if setting_metric_summary.empty:
        return sweep_trend_metrics

    for group_key, group in setting_metric_summary.groupby(
        ["sweep_id", "explainer_family"],
        sort=True,
    ):
        sweep_id, explainer_family = cast(tuple[Any, Any], group_key)
        sorted_group = group.sort_values("step_index").reset_index(drop=True)
        parameter_name = str(sorted_group["parameter_name"].iloc[0])
        insertion_auc_values = sorted_group["decision_insertion_auc_mean"]
        deletion_auc_values = sorted_group["decision_deletion_auc_mean"]
        decision_infidelity_values = sorted_group["decision_infidelity_mean"]
        sweep_trend_metrics.setdefault(sweep_id, {})[explainer_family] = {
            "parameter_name": parameter_name,
            "setting_count": int(len(sorted_group)),
            "decision_insertion_auc_spearman": _compute_trend_correlation(
                sorted_group["parameter_value"],
                insertion_auc_values,
                correlation_key="spearman",
                compute_correlation=compute_spearman_rank_correlation,
            ),
            "decision_insertion_auc_kendall_tau": _compute_trend_correlation(
                sorted_group["parameter_value"],
                insertion_auc_values,
                correlation_key="kendall_tau",
                compute_correlation=compute_kendall_tau_correlation,
            ),
            "decision_deletion_auc_spearman": _compute_trend_correlation(
                sorted_group["parameter_value"],
                deletion_auc_values,
                correlation_key="spearman",
                compute_correlation=compute_spearman_rank_correlation,
            ),
            "decision_deletion_auc_kendall_tau": _compute_trend_correlation(
                sorted_group["parameter_value"],
                deletion_auc_values,
                correlation_key="kendall_tau",
                compute_correlation=compute_kendall_tau_correlation,
            ),
            "decision_infidelity_spearman": _compute_trend_correlation(
                sorted_group["parameter_value"],
                decision_infidelity_values,
                correlation_key="spearman",
                compute_correlation=compute_spearman_rank_correlation,
            ),
            "decision_infidelity_kendall_tau": _compute_trend_correlation(
                sorted_group["parameter_value"],
                decision_infidelity_values,
                correlation_key="kendall_tau",
                compute_correlation=compute_kendall_tau_correlation,
            ),
        }
    return sweep_trend_metrics


def _compute_trend_correlation(
    parameter_values: pd.Series,
    metric_values: pd.Series,
    *,
    correlation_key: str,
    compute_correlation: Callable[[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray], float | None],
) -> dict[str, Any]:
    trend_frame = pd.DataFrame(
        {
            "parameter_value": pd.to_numeric(parameter_values, errors="coerce"),
            "metric_value": pd.to_numeric(metric_values, errors="coerce"),
        }
    ).dropna()
    if len(trend_frame) < 3:
        return {
            correlation_key: None,
            "computable": False,
            "setting_count": int(len(trend_frame)),
        }

    if (
        trend_frame["parameter_value"].nunique(dropna=False) <= 1
        or trend_frame["metric_value"].nunique(dropna=False) <= 1
    ):
        return {
            correlation_key: None,
            "computable": False,
            "setting_count": int(len(trend_frame)),
        }

    return {
        correlation_key: compute_correlation(
            trend_frame["parameter_value"].to_numpy(dtype=float),
            trend_frame["metric_value"].to_numpy(dtype=float),
        ),
        "computable": True,
        "setting_count": int(len(trend_frame)),
    }


def _load_setting_artifacts(
    entry: SweepManifestEntry,
    *,
    manifest_path: Path | str | None = None,
) -> SweepSettingArtifacts:
    required_paths = {
        "daily_shap": entry.results_dir / "daily_shap.csv",
        "daily_full_dispatch": entry.results_dir / "daily_full_dispatch.csv",
        "summary_shap": entry.results_dir / "summary_shap.csv",
        "evaluation_metrics": entry.results_dir / "evaluation_metrics.json",
        "run_metadata": entry.results_dir / "run_metadata.json",
    }
    missing_outputs = [
        label
        for label, path in required_paths.items()
        if not path.exists()
    ]
    if missing_outputs:
        hint = ""
        if manifest_path is not None:
            hint = (
                " Generate the missing case-study outputs first with "
                f"`uv run src/analysis/run_caiso_shap_sweep_case_studies.py --manifest {manifest_path}`."
            )
        raise FileNotFoundError(
            f"{entry.results_dir} is missing required sweep-comparison outputs: "
            + ", ".join(missing_outputs)
            + hint
        )

    daily_shap = pd.read_csv(required_paths["daily_shap"])
    daily_shap["date"] = daily_shap["date"].astype(str)
    daily_full_dispatch = pd.read_csv(required_paths["daily_full_dispatch"])
    daily_full_dispatch["date"] = daily_full_dispatch["date"].astype(str)
    summary_shap = pd.read_csv(required_paths["summary_shap"])

    with required_paths["evaluation_metrics"].open("r", encoding="utf-8") as handle:
        evaluation_metrics = json.load(handle)
    with required_paths["run_metadata"].open("r", encoding="utf-8") as handle:
        run_metadata = json.load(handle)

    feature_names = tuple(str(name) for name in run_metadata["feature_columns"])
    explain_dates = tuple(str(date) for date in daily_shap["date"].tolist())
    _validate_summary_shap(summary_shap, feature_names)
    return SweepSettingArtifacts(
        manifest_entry=entry,
        daily_shap=daily_shap,
        daily_full_dispatch=daily_full_dispatch,
        summary_shap=summary_shap,
        evaluation_metrics=evaluation_metrics,
        run_metadata=run_metadata,
        feature_names=feature_names,
        explain_dates=explain_dates,
    )


def _validate_manifest_entries(
    entries: Sequence[SweepManifestEntry],
) -> None:
    by_sweep = _group_entries_by_sweep(entries)
    for sweep_id, sweep_entries in by_sweep.items():
        parameter_names = {entry.parameter_name for entry in sweep_entries}
        if len(parameter_names) != 1:
            raise ValueError(
                f"All rows in sweep_id={sweep_id!r} must share one parameter_name."
            )

        sorted_steps = sorted(entry.step_index for entry in sweep_entries)
        if len(sorted_steps) != len(set(sorted_steps)):
            raise ValueError(f"step_index values must be unique within sweep_id={sweep_id!r}.")

        if any(
            next_step - current_step != 1
            for current_step, next_step in zip(sorted_steps, sorted_steps[1:])
        ):
            raise ValueError(
                f"step_index values must be consecutive within sweep_id={sweep_id!r}."
            )


def _group_entries_by_sweep(
    manifest_entries: Sequence[SweepManifestEntry],
) -> dict[str, list[SweepManifestEntry]]:
    grouped: dict[str, list[SweepManifestEntry]] = {}
    for entry in manifest_entries:
        grouped.setdefault(entry.sweep_id, []).append(entry)
    for sweep_id, sweep_entries in grouped.items():
        grouped[sweep_id] = sorted(sweep_entries, key=lambda entry: entry.step_index)
    return grouped


def _validate_setting_pair(
    left_artifacts: SweepSettingArtifacts,
    right_artifacts: SweepSettingArtifacts,
) -> None:
    if left_artifacts.feature_names != right_artifacts.feature_names:
        raise ValueError("Neighboring settings must share the same feature ordering.")
    if left_artifacts.explain_dates != right_artifacts.explain_dates:
        raise ValueError("Neighboring settings must share the same explanation dates.")


def _validate_summary_shap(
    summary_shap: pd.DataFrame,
    feature_names: Sequence[str],
) -> None:
    if "feature" not in summary_shap.columns:
        raise KeyError("summary_shap.csv must contain a feature column.")
    if set(summary_shap["feature"]) != set(feature_names):
        raise ValueError("summary_shap.csv features must match run_metadata feature_columns.")


def _frame_records_for_json(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    normalized = frame.astype(object).where(pd.notna(frame), None)
    return [
        {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in record.items()
        }
        for record in normalized.to_dict(orient="records")
    ]
