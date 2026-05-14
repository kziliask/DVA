from __future__ import annotations

from dva.model.taxi_features import resolve_taxi_training_features


def test_poc_taxi_feature_set_matches_training_columns() -> None:
    assert resolve_taxi_training_features("poc_v1") == (
        "hour",
        "day_of_week",
        "month",
        "pickup_count",
        "pickup_count_lag_1",
        "temp_c",
        "precip_mm",
    )
