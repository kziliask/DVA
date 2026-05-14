from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dva.analysis.caiso_shap import (
    CaisoShapCaseStudyConfig,
    build_default_storage_parameters,
)
from dva.analysis.caiso_sweep import SweepManifestEntry, load_sweep_manifest
from dva.model.storage_dispatch import StorageDispatchParameters


CASE_STUDY_REQUIRED_OUTPUT_FILENAMES = (
    "daily_shap.csv",
    "daily_full_dispatch.csv",
    "summary_shap.csv",
    "evaluation_metrics.json",
    "run_metadata.json",
)


@dataclass(frozen=True, slots=True)
class CaisoSweepRunEntry:
    manifest_entry: SweepManifestEntry
    config: CaisoShapCaseStudyConfig


def load_caiso_shap_case_study_sweep_manifest(
    manifest_path: Path | str,
) -> list[CaisoSweepRunEntry]:
    manifest_path_obj = Path(manifest_path)
    manifest_entries = load_sweep_manifest(manifest_path_obj)
    manifest_frame = pd.read_csv(manifest_path_obj)
    records_by_setting_id = {
        str(record["setting_id"]): record
        for record in manifest_frame.to_dict(orient="records")
    }

    default_config = CaisoShapCaseStudyConfig()
    default_storage = build_default_storage_parameters()
    run_entries: list[CaisoSweepRunEntry] = []
    for manifest_entry in manifest_entries:
        record = records_by_setting_id[manifest_entry.setting_id]
        run_entries.append(
            CaisoSweepRunEntry(
                manifest_entry=manifest_entry,
                config=_build_case_study_config(
                    record,
                    manifest_dir=manifest_path_obj.parent,
                    resolved_results_dir=manifest_entry.results_dir,
                    default_config=default_config,
                    default_storage=default_storage,
                ),
            )
        )
    return run_entries


def case_study_outputs_are_complete(results_dir: Path | str) -> bool:
    results_dir_path = Path(results_dir)
    return all(
        (results_dir_path / filename).exists()
        for filename in CASE_STUDY_REQUIRED_OUTPUT_FILENAMES
    )


def _build_case_study_config(
    record: dict[str, Any],
    *,
    manifest_dir: Path,
    resolved_results_dir: Path,
    default_config: CaisoShapCaseStudyConfig,
    default_storage: StorageDispatchParameters,
) -> CaisoShapCaseStudyConfig:
    return CaisoShapCaseStudyConfig(
        dataset_path=_resolve_optional_path(
            record,
            manifest_dir=manifest_dir,
            default=default_config.dataset_path,
            column_names=("dataset_path",),
        ),
        holdout_days=_resolve_optional_int(
            record,
            default=default_config.holdout_days,
            column_names=("holdout_days",),
        ),
        outdir=resolved_results_dir,
        model_name=_resolve_optional_str(
            record,
            default=default_config.model_name,
            column_names=("model", "model_name"),
        ),
        random_state=_resolve_optional_int(
            record,
            default=default_config.random_state,
            column_names=("random_state",),
        ),
        n_jobs=_resolve_optional_int(
            record,
            default=default_config.n_jobs,
            column_names=("n_jobs",),
        ),
        mlp_hidden_layer_sizes=(
            _resolve_optional_int(
                record,
                default=default_config.mlp_hidden_layer_sizes[0],
                column_names=("mlp_hidden_units",),
            ),
        ),
        mlp_max_iter=_resolve_optional_int(
            record,
            default=default_config.mlp_max_iter,
            column_names=("mlp_max_iter",),
        ),
        mlp_dropout=_resolve_optional_float(
            record,
            default=default_config.mlp_dropout,
            column_names=("mlp_dropout", "dropout"),
        ),
        mlp_weight_decay=_resolve_optional_float(
            record,
            default=default_config.mlp_weight_decay,
            column_names=("mlp_weight_decay", "weight_decay"),
        ),
        mlp_batch_size=_resolve_optional_int_or_none(
            record,
            default=default_config.mlp_batch_size,
            column_names=("mlp_batch_size", "batch_size"),
        ),
        mlp_early_stopping_patience=_resolve_optional_int_or_none(
            record,
            default=default_config.mlp_early_stopping_patience,
            column_names=(
                "mlp_early_stopping_patience",
                "early_stopping_patience",
            ),
        ),
        mlp_activation=_resolve_optional_str(
            record,
            default=default_config.mlp_activation,
            column_names=("mlp_activation", "activation"),
        ),
        mlp_batch_norm=_resolve_optional_bool(
            record,
            default=default_config.mlp_batch_norm,
            column_names=("mlp_batch_norm", "batch_norm"),
        ),
        xgb_n_estimators=_resolve_optional_int(
            record,
            default=default_config.xgb_n_estimators,
            column_names=("xgb_n_estimators",),
        ),
        xgb_max_depth=_resolve_optional_int(
            record,
            default=default_config.xgb_max_depth,
            column_names=("xgb_max_depth",),
        ),
        xgb_learning_rate=_resolve_optional_float(
            record,
            default=default_config.xgb_learning_rate,
            column_names=("xgb_learning_rate",),
        ),
        xgb_subsample=_resolve_optional_float(
            record,
            default=default_config.xgb_subsample,
            column_names=("xgb_subsample",),
        ),
        xgb_colsample_bytree=_resolve_optional_float(
            record,
            default=default_config.xgb_colsample_bytree,
            column_names=("xgb_colsample_bytree",),
        ),
        xgb_reg_lambda=_resolve_optional_float(
            record,
            default=default_config.xgb_reg_lambda,
            column_names=("xgb_reg_lambda",),
        ),
        xgb_verbosity=_resolve_optional_int(
            record,
            default=default_config.xgb_verbosity,
            column_names=("xgb_verbosity",),
        ),
        learning_rate=_resolve_optional_float_or_none(
            record,
            default=default_config.learning_rate,
            column_names=("lr", "learning_rate"),
        ),
        mse_learning_rate=_resolve_optional_float_or_none(
            record,
            default=default_config.mse_learning_rate,
            column_names=("mse_lr", "mse_learning_rate"),
        ),
        spo_learning_rate=_resolve_optional_float_or_none(
            record,
            default=default_config.spo_learning_rate,
            column_names=("spo_lr", "spo_learning_rate"),
        ),
        spo_warm_start_with_mse=_resolve_optional_bool(
            record,
            default=default_config.spo_warm_start_with_mse,
            column_names=("spo_warm_start_with_mse",),
        ),
        spo_processes=_resolve_optional_int_or_none(
            record,
            default=default_config.spo_processes,
            column_names=("spo_processes",),
        ),
        solver_seed=_resolve_optional_int(
            record,
            default=default_config.solver_seed,
            column_names=("solver_seed",),
        ),
        mip_gap=_resolve_optional_float(
            record,
            default=default_config.mip_gap,
            column_names=("mip_gap",),
        ),
        mip_gap_abs=_resolve_optional_float(
            record,
            default=default_config.mip_gap_abs,
            column_names=("mip_gap_abs",),
        ),
        objective_tolerance=_resolve_optional_float(
            record,
            default=default_config.objective_tolerance,
            column_names=("objective_tolerance",),
        ),
        max_days=_resolve_optional_int_or_none(
            record,
            default=default_config.max_days,
            column_names=("max_days",),
        ),
        interaction_order=_resolve_optional_int_or_none(
            record,
            default=default_config.interaction_order,
            column_names=("interaction_order",),
        ),
        interaction_method=_resolve_optional_str(
            record,
            default=default_config.interaction_method,
            column_names=("interaction_method",),
        ),
        compute_ead_decision_shap=_resolve_optional_bool(
            record,
            default=default_config.compute_ead_decision_shap,
            column_names=("compute_ead_decision_shap", "compute_ead_shap"),
        ),
        storage_parameters=StorageDispatchParameters(
            energy_capacity=_resolve_optional_float(
                record,
                default=default_storage.energy_capacity,
                column_names=("energy_capacity",),
            ),
            power_limit=_resolve_optional_float(
                record,
                default=default_storage.power_limit,
                column_names=("power_limit",),
            ),
            charge_efficiency=_resolve_optional_float(
                record,
                default=default_storage.charge_efficiency,
                column_names=("charge_efficiency",),
            ),
            discharge_efficiency=_resolve_optional_float(
                record,
                default=default_storage.discharge_efficiency,
                column_names=("discharge_efficiency",),
            ),
            throughput_penalty=_resolve_optional_float(
                record,
                default=default_storage.throughput_penalty,
                column_names=("throughput_penalty",),
            ),
            initial_state_of_charge=_resolve_optional_float(
                record,
                default=default_storage.initial_state_of_charge,
                column_names=("initial_soc", "initial_state_of_charge"),
            ),
            terminal_state_of_charge=_resolve_optional_float_or_none(
                record,
                default=default_storage.terminal_state_of_charge,
                column_names=("terminal_soc", "terminal_state_of_charge"),
            ),
        ),
    )


def _resolve_optional_path(
    record: dict[str, Any],
    *,
    manifest_dir: Path,
    default: Path,
    column_names: tuple[str, ...],
) -> Path:
    value = _coalesce_record_value(record, column_names)
    if value is _MISSING:
        return default

    path = Path(str(value))
    if not path.is_absolute():
        path = (manifest_dir / path).resolve()
    return path


def _resolve_optional_str(
    record: dict[str, Any],
    *,
    default: str,
    column_names: tuple[str, ...],
) -> str:
    value = _coalesce_record_value(record, column_names)
    if value is _MISSING:
        return default
    return str(value)


def _resolve_optional_int(
    record: dict[str, Any],
    *,
    default: int,
    column_names: tuple[str, ...],
) -> int:
    value = _coalesce_record_value(record, column_names)
    if value is _MISSING:
        return default
    return int(value)


def _resolve_optional_float(
    record: dict[str, Any],
    *,
    default: float,
    column_names: tuple[str, ...],
) -> float:
    value = _coalesce_record_value(record, column_names)
    if value is _MISSING:
        return float(default)
    return float(value)


def _resolve_optional_int_or_none(
    record: dict[str, Any],
    *,
    default: int | None,
    column_names: tuple[str, ...],
) -> int | None:
    value = _coalesce_record_value(record, column_names)
    if value is _MISSING:
        return default
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return int(value)


def _resolve_optional_bool(
    record: dict[str, Any],
    *,
    default: bool,
    column_names: tuple[str, ...],
) -> bool:
    value = _coalesce_record_value(record, column_names)
    if value is _MISSING:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n"}:
            return False
        raise ValueError(
            "Boolean manifest values must be one of "
            "'true', 'false', '1', '0', 'yes', or 'no'."
        )
    if value in {0, 1}:
        return bool(value)
    raise ValueError(
        f"Boolean manifest value must be true/false or 1/0; got {value!r}."
    )


def _resolve_optional_float_or_none(
    record: dict[str, Any],
    *,
    default: float | None,
    column_names: tuple[str, ...],
) -> float | None:
    value = _coalesce_record_value(record, column_names)
    if value is _MISSING:
        return None if default is None else float(default)
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return float(value)


def _coalesce_record_value(
    record: dict[str, Any],
    column_names: tuple[str, ...],
) -> Any:
    for column_name in column_names:
        if column_name not in record:
            continue
        value = record[column_name]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                continue
            return stripped
        if pd.isna(value):
            continue
        return value
    return _MISSING


_MISSING = object()
