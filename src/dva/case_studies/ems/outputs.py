from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EMS_DESIGN_PLAYER_NAMES = {"solver", "radius_km", "staging_areas"}


def write_canonical_ems_dva_outputs(
    outdir: Path | str,
    *,
    value_mode: str = "post",
) -> None:
    outdir_path = Path(outdir)
    hourly_path = outdir_path / "hourly_shap.csv"
    summary_path = outdir_path / "summary_shap.csv"
    if not hourly_path.exists() or not summary_path.exists():
        return

    hourly = pd.read_csv(hourly_path)
    canonical = hourly.copy()
    prefix = "ante_decision" if value_mode == "ante" else "decision"
    if value_mode == "ante" and not any(
        column.startswith("ante_decision_shap_") for column in canonical.columns
    ):
        prefix = "decision"
        canonical["dva_value_mode_warning"] = (
            "ante requested; deterministic EMS core currently reused post decision values"
        )
    for column in list(canonical.columns):
        if column.startswith(f"{prefix}_shap_"):
            canonical[column.replace(f"{prefix}_shap_", "dva_")] = canonical[column]
    if f"{prefix}_value_gain" in canonical:
        canonical["dva_value_gain"] = canonical[f"{prefix}_value_gain"]
    elif "decision_value_gain" in canonical:
        canonical["dva_value_gain"] = canonical["decision_value_gain"]
    canonical["dva_value_mode"] = value_mode
    canonical.to_csv(outdir_path / "daily_dva.csv", index=False)

    summary = pd.read_csv(summary_path)
    summary["dva_value_mode"] = value_mode
    if value_mode == "ante" and "ante_decision_mean_abs_shap" in summary:
        summary["dva_mean_abs"] = summary["ante_decision_mean_abs_shap"]
        summary["dva_rank"] = summary["ante_decision_rank"]
    elif "decision_mean_abs_shap" in summary:
        summary["dva_mean_abs"] = summary["decision_mean_abs_shap"]
        summary["dva_rank"] = summary["decision_rank"]
    summary.to_csv(outdir_path / "summary_dva.csv", index=False)

    metadata_path = outdir_path / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["dva_value_mode"] = value_mode
        metadata["canonical_dva_outputs"] = {
            "daily": "daily_dva.csv",
            "summary": "summary_dva.csv",
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
