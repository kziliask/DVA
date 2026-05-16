from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def make_l25_oa_symbols() -> np.ndarray:
    p = 5
    rows = []
    for a in range(p):
        for b in range(p):
            rows.append(
                [
                    a,
                    b,
                    (a + b) % p,
                    (a + 2 * b) % p,
                    (a + 3 * b) % p,
                    (a + 4 * b) % p,
                ]
            )
    return np.array(rows, dtype=int)


EMS_XGB_LEVELS = {
    "n_estimators": [100, 50, 150, 250, 350],
    "max_depth": [3, 2, 4, 5, 6],
    "learning_rate": [0.05, 0.01, 0.03, 0.10, 0.15],
    "subsample": [0.90, 0.60, 0.75, 0.95, 1.00],
    "colsample_bytree": [0.90, 0.60, 0.75, 0.95, 1.00],
    "reg_lambda": [1.0, 0.3, 3.0, 10.0, 30.0],
}


def make_ems_xgb_model_manifest() -> pd.DataFrame:
    oa = make_l25_oa_symbols()
    design = pd.DataFrame(
        {
            name: [levels[symbol] for symbol in oa[:, factor_idx]]
            for factor_idx, (name, levels) in enumerate(EMS_XGB_LEVELS.items())
        }
    )
    design.insert(0, "run", np.arange(1, len(design) + 1))
    design.insert(0, "model_id", [f"xgb_{run:03d}" for run in design["run"]])
    design.insert(1, "model_name", "xgb")
    return design


EMS_XGB_MODEL_IDS = tuple(make_ems_xgb_model_manifest()["model_id"].astype(str))


def resolve_ems_xgb_model_record(model_id: str) -> dict[str, Any]:
    manifest = make_ems_xgb_model_manifest()
    matches = manifest.loc[manifest["model_id"].eq(str(model_id))]
    if matches.empty:
        raise ValueError(
            f"Unknown EMS model_id {model_id!r}. Expected one of: "
            + ", ".join(EMS_XGB_MODEL_IDS)
        )
    return dict(matches.iloc[0])


def ems_xgb_config_kwargs(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "xgb_n_estimators": int(record["n_estimators"]),
        "xgb_max_depth": int(record["max_depth"]),
        "xgb_learning_rate": float(record["learning_rate"]),
        "xgb_subsample": float(record["subsample"]),
        "xgb_colsample_bytree": float(record["colsample_bytree"]),
        "xgb_reg_lambda": float(record["reg_lambda"]),
    }


__all__ = [
    "EMS_XGB_LEVELS",
    "EMS_XGB_MODEL_IDS",
    "ems_xgb_config_kwargs",
    "make_ems_xgb_model_manifest",
    "make_l25_oa_symbols",
    "resolve_ems_xgb_model_record",
]
