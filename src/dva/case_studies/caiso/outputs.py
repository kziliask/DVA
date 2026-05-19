from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pandas as pd


DESIGN_PLAYER_NAMES = {"efficiency", "energy_capacity"}
INFO_PLAYER_KIND = "info"
DESIGN_PLAYER_KIND = "design"


def write_canonical_caiso_dva_outputs(
    outdir: Path | str,
    *,
    value_mode: str,
) -> None:
    outdir_path = Path(outdir)
    daily_path = outdir_path / "daily_shap.csv"
    summary_path = outdir_path / "summary_shap.csv"
    if not daily_path.exists() or not summary_path.exists():
        return

    daily = pd.read_csv(daily_path)
    summary = pd.read_csv(summary_path)
    prefix = "ead_decision" if value_mode == "ante" else "decision"
    canonical = daily.copy()
    for column in list(canonical.columns):
        if column.startswith(f"{prefix}_shap_"):
            canonical[column.replace(f"{prefix}_shap_", "dva_")] = canonical[column]
    if f"{prefix}_value_gain" in canonical:
        canonical["dva_value_gain"] = canonical[f"{prefix}_value_gain"]
    elif "decision_value_gain" in canonical:
        canonical["dva_value_gain"] = canonical["decision_value_gain"]
    canonical["dva_value_mode"] = value_mode
    canonical.to_csv(outdir_path / "daily_dva.csv", index=False)

    canonical_summary = summary.copy()
    if value_mode == "ante" and "ead_decision_mean_abs_shap" in canonical_summary:
        canonical_summary["dva_mean_abs"] = canonical_summary["ead_decision_mean_abs_shap"]
        canonical_summary["dva_rank"] = canonical_summary["ead_decision_rank"]
    elif "decision_mean_abs_shap" in canonical_summary:
        canonical_summary["dva_mean_abs"] = canonical_summary["decision_mean_abs_shap"]
        canonical_summary["dva_rank"] = canonical_summary["decision_rank"]
    canonical_summary["dva_value_mode"] = value_mode
    canonical_summary.to_csv(outdir_path / "summary_dva.csv", index=False)

    metadata_path = outdir_path / "run_metadata.json"
    metadata = _load_metadata(metadata_path)
    _write_joint_dva_outputs(
        outdir_path,
        daily=daily,
        summary=summary,
        metadata=metadata,
        value_mode=value_mode,
    )

    interaction_path = _canonical_interaction_path(outdir_path, value_mode)
    if interaction_path.exists():
        interactions = pd.read_csv(interaction_path)
        if "subset_size" in interactions.columns:
            interactions = interactions.loc[interactions["subset_size"].eq(2)].copy()
        value_column = _interaction_value_column(value_mode)
        if value_column not in interactions.columns:
            raise KeyError(
                f"{interaction_path} is missing required column {value_column!r}."
            )
        if value_column != "decision_interaction_value":
            interactions["decision_interaction_value"] = interactions[value_column]
        interactions["interaction_type"] = interactions["players"].map(
            classify_caiso_interaction_players
        )
        interactions.to_csv(outdir_path / "dvi_interactions.csv", index=False)
    elif value_mode == "ante":
        raise FileNotFoundError(
            f"Ante CAISO DVI requires {interaction_path}; refusing to reuse post-DVI."
        )

    if metadata_path.exists():
        metadata["dva_value_mode"] = value_mode
        metadata["canonical_dva_outputs"] = {
            "daily": "daily_dva.csv",
            "summary": "summary_dva.csv",
            "joint_daily": "joint_dva.csv",
            "joint_summary": "joint_summary_dva.csv",
            "interactions": (
                "dvi_interactions.csv"
                if (outdir_path / "dvi_interactions.csv").exists()
                else None
            ),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _canonical_interaction_path(outdir_path: Path, value_mode: str) -> Path:
    if value_mode == "ante":
        return outdir_path / "daily_interaction_ead_decision.csv"
    if value_mode == "post":
        return outdir_path / "daily_interaction_decision.csv"
    raise ValueError(f"Unsupported CAISO DVA value_mode: {value_mode!r}")


def _interaction_value_column(value_mode: str) -> str:
    if value_mode == "ante":
        return "ead_decision_interaction_value"
    if value_mode == "post":
        return "decision_interaction_value"
    raise ValueError(f"Unsupported CAISO DVA value_mode: {value_mode!r}")


def _shap_prefix(value_mode: str) -> str:
    if value_mode == "ante":
        return "ead_decision"
    if value_mode == "post":
        return "decision"
    raise ValueError(f"Unsupported CAISO DVA value_mode: {value_mode!r}")


def _write_joint_dva_outputs(
    outdir_path: Path,
    *,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: dict[str, Any],
    value_mode: str,
) -> None:
    prefix = _shap_prefix(value_mode)
    players = _resolve_player_names(daily, summary, metadata, prefix)
    model_id = _resolve_model_id(outdir_path, metadata)

    daily_rows: list[dict[str, Any]] = []
    for row in daily.itertuples(index=False):
        date = getattr(row, "date")
        for player in players:
            shap_column = f"{prefix}_shap_{player}"
            if shap_column not in daily.columns:
                continue
            daily_rows.append(
                {
                    "date": date,
                    "player": player,
                    "player_kind": _player_kind(player),
                    "baseline": _player_baseline(player, metadata),
                    "actual": _player_actual(player, metadata),
                    "dva_value": float(getattr(row, shap_column)),
                    "value_mode": value_mode,
                    "model_id": model_id,
                }
            )
    pd.DataFrame(daily_rows).to_csv(outdir_path / "joint_dva.csv", index=False)

    required_summary_columns = {
        "feature",
        f"{prefix}_mean_signed_shap",
        f"{prefix}_mean_abs_shap",
        f"{prefix}_rank",
    }
    missing_columns = required_summary_columns - set(summary.columns)
    if missing_columns:
        raise KeyError(
            "Cannot write CAISO joint_summary_dva.csv; summary_shap.csv is missing: "
            + ", ".join(sorted(missing_columns))
        )

    summary_rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        player = str(getattr(row, "feature"))
        if player not in players:
            continue
        summary_rows.append(
            {
                "player": player,
                "player_kind": _player_kind(player),
                "dva_mean_signed": float(getattr(row, f"{prefix}_mean_signed_shap")),
                "dva_mean_abs": float(getattr(row, f"{prefix}_mean_abs_shap")),
                "dva_rank": int(getattr(row, f"{prefix}_rank")),
                "value_mode": value_mode,
                "model_id": model_id,
            }
        )
    joint_summary = pd.DataFrame(summary_rows)
    if not joint_summary.empty:
        joint_summary = joint_summary.sort_values("dva_rank").reset_index(drop=True)
    joint_summary.to_csv(
        outdir_path / "joint_summary_dva.csv",
        index=False,
    )


def _resolve_player_names(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: dict[str, Any],
    prefix: str,
) -> list[str]:
    metadata_players = metadata.get("player_names")
    if isinstance(metadata_players, list) and all(
        isinstance(player, str) for player in metadata_players
    ):
        return metadata_players

    summary_players = (
        summary["feature"].astype(str).tolist() if "feature" in summary.columns else []
    )
    if summary_players:
        return summary_players

    column_prefix = f"{prefix}_shap_"
    return [
        column.removeprefix(column_prefix)
        for column in daily.columns
        if column.startswith(column_prefix)
    ]


def _resolve_model_id(outdir_path: Path, metadata: dict[str, Any]) -> str | None:
    evaluation_label = str(metadata.get("evaluation_label", ""))
    match = re.search(r"(?:^|_)(xgb_\d{3})(?:_|$)", evaluation_label)
    if match:
        return match.group(1)
    for part in reversed(outdir_path.parts):
        if re.fullmatch(r"xgb_\d{3}", part):
            return part
    model_name = metadata.get("model_name")
    return str(model_name) if model_name is not None else None


def _player_kind(player: str) -> str:
    return DESIGN_PLAYER_KIND if player in DESIGN_PLAYER_NAMES else INFO_PLAYER_KIND


def _player_baseline(player: str, metadata: dict[str, Any]) -> Any:
    parameter_spec = metadata.get("parameter_player_spec")
    if not isinstance(parameter_spec, dict):
        return None
    if player == "efficiency":
        return parameter_spec.get("charge_efficiency_baseline")
    if player == "energy_capacity":
        return parameter_spec.get("energy_capacity_baseline")
    return None


def _player_actual(player: str, metadata: dict[str, Any]) -> Any:
    storage_parameters = metadata.get("storage_parameters")
    if not isinstance(storage_parameters, dict):
        return None
    if player == "efficiency":
        return storage_parameters.get("charge_efficiency")
    if player == "energy_capacity":
        return storage_parameters.get("energy_capacity")
    return None


def classify_caiso_interaction_players(players: Any) -> str:
    names = {name for name in str(players).split("|") if name}
    if not names:
        return "Unknown"
    design_count = len(names & DESIGN_PLAYER_NAMES)
    if design_count == 0:
        return "Info-Info"
    if design_count == len(names):
        return "Design-Design"
    return "Cross-DVI"
