from __future__ import annotations


DEFAULT_TAXI_TRAINING_FEATURE_SET = "poc_v1"
TAXI_TRAINING_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "poc_v1": (
        "hour",
        "day_of_week",
        "month",
        "pickup_count",
        "pickup_count_lag_1",
        "temp_c",
        "precip_mm",
    ),
}
TAXI_CATEGORICAL_FEATURES = ("borough", "service_zone")
TAXI_TARGET_COLUMN = "target_pickup_count_next_hour"
TAXI_TIMESTAMP_COLUMN = "timestamp_hour"
TAXI_ZONE_ID_COLUMN = "zone_id"


def resolve_taxi_training_features(
    feature_set: str = DEFAULT_TAXI_TRAINING_FEATURE_SET,
) -> tuple[str, ...]:
    try:
        return TAXI_TRAINING_FEATURE_SETS[feature_set]
    except KeyError as exc:
        valid_feature_sets = ", ".join(sorted(TAXI_TRAINING_FEATURE_SETS))
        raise KeyError(
            f"Unknown taxi feature set {feature_set!r}. "
            f"Available feature sets: {valid_feature_sets}."
        ) from exc
