from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DESIGN_PLAYER_NAMES = {"efficiency", "energy_capacity"}


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

    interaction_path = outdir_path / "daily_interaction_decision.csv"
    if interaction_path.exists():
        interactions = pd.read_csv(interaction_path)
        if "subset_size" in interactions.columns:
            interactions = interactions.loc[interactions["subset_size"].eq(2)].copy()
        interactions["interaction_type"] = interactions["players"].map(
            classify_caiso_interaction_players
        )
        interactions.to_csv(outdir_path / "dvi_interactions.csv", index=False)

    metadata_path = outdir_path / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["dva_value_mode"] = value_mode
        metadata["canonical_dva_outputs"] = {
            "daily": "daily_dva.csv",
            "summary": "summary_dva.csv",
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
