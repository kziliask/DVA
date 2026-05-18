from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_ems_models_module():
    path = Path(__file__).parents[1] / "src" / "dva" / "case_studies" / "ems" / "models.py"
    spec = importlib.util.spec_from_file_location("_ems_models_for_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_ems_xgb_model_record_returns_json_serializable_scalars() -> None:
    ems_models = _load_ems_models_module()
    record = ems_models.resolve_ems_xgb_model_record("xgb_001")

    assert type(record["run"]) is int
    assert type(record["max_depth"]) is int
    assert type(record["learning_rate"]) is float
    json.dumps({"model_record": record}, indent=2, sort_keys=True)
