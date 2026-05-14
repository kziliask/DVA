from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


DEFAULT_TOP_K = 5
DEFAULT_RBO_DEPTH = 10
DEFAULT_RBO_P = 0.9
DEFAULT_POLICY_ATOL = 1e-6
DECISION_INSERTION_AUC_CLIPPING = "clip_to_[0,v_dec(N)]"
DECISION_DELETION_AUC_CLIPPING = DECISION_INSERTION_AUC_CLIPPING


@dataclass(frozen=True, slots=True)
class DailyDispatchPolicy:
    date: str
    charge: tuple[float, ...]
    discharge: tuple[float, ...]
    mode: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DecisionActivationMetrics:
    activation_rate: np.ndarray
    activated_value_sum: np.ndarray
    activated_value: np.ndarray


def build_attribution_ranking(
    attributions: Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
) -> tuple[str, ...]:
    attribution_array = _validate_feature_aligned_vector(attributions, feature_names)
    indexed_scores = list(enumerate(np.abs(attribution_array)))
    ranked_indices = sorted(
        indexed_scores,
        key=lambda item: (-float(item[1]), item[0]),
    )
    return tuple(feature_names[index] for index, _ in ranked_indices)


def rank_features_from_scores(
    scores_by_feature: Mapping[str, float],
    feature_names: Sequence[str],
) -> tuple[str, ...]:
    if set(scores_by_feature) != set(feature_names):
        raise ValueError("scores_by_feature must contain exactly one score per feature.")

    indexed_scores = [
        (index, float(scores_by_feature[feature_name]))
        for index, feature_name in enumerate(feature_names)
    ]
    ranked_indices = sorted(
        indexed_scores,
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(feature_names[index] for index, _ in ranked_indices)


def compute_decision_activation_metrics(
    coalition_values: Sequence[float] | np.ndarray,
    decisions: Sequence[Any],
    feature_count: int,
    decision_changed: Callable[[Any, Any], bool],
) -> DecisionActivationMetrics:
    coalition_array = _validate_scalar_coalition_values(
        coalition_values,
        feature_count=feature_count,
    )
    expected_coalition_count = 1 << feature_count
    if len(decisions) != expected_coalition_count:
        raise ValueError(
            "decisions must contain one decision per coalition. "
            f"Expected {expected_coalition_count}, got {len(decisions)}."
        )

    activation_rate = np.zeros(feature_count, dtype=float)
    activated_value_sum = np.zeros(feature_count, dtype=float)
    weights = _activation_subset_weights(feature_count)
    for feature_idx in range(feature_count):
        feature_bit = 1 << feature_idx
        for coalition_mask in range(expected_coalition_count):
            if coalition_mask & feature_bit:
                continue
            coalition_with_feature = coalition_mask | feature_bit
            activated = float(
                decision_changed(
                    decisions[coalition_mask],
                    decisions[coalition_with_feature],
                )
            )
            if activated == 0.0:
                continue
            coalition_size = coalition_mask.bit_count()
            marginal_value = (
                coalition_array[coalition_with_feature]
                - coalition_array[coalition_mask]
            )
            weight = weights[coalition_size]
            activation_rate[feature_idx] += weight
            activated_value_sum[feature_idx] += weight * float(marginal_value)

    activated_value = np.divide(
        activated_value_sum,
        activation_rate,
        out=np.zeros_like(activated_value_sum),
        where=activation_rate > 0.0,
    )
    return DecisionActivationMetrics(
        activation_rate=activation_rate,
        activated_value_sum=activated_value_sum,
        activated_value=activated_value,
    )


def compute_decision_insertion_auc(
    attributions: Sequence[float] | np.ndarray,
    coalition_values: Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
) -> float | None:
    attribution_array = _validate_feature_aligned_vector(attributions, feature_names)
    coalition_array = _validate_scalar_coalition_values(
        coalition_values,
        feature_count=len(feature_names),
    )
    full_mask = len(coalition_array) - 1
    full_value = float(coalition_array[full_mask])
    if full_value <= 0:
        return None

    ranking = build_attribution_ranking(attribution_array, feature_names)
    feature_index = {feature_name: index for index, feature_name in enumerate(feature_names)}

    running_mask = 0
    strict_prefix_values: list[float] = []
    normalized_sum = float(np.clip(coalition_array[0], 0.0, full_value) / full_value)
    for prefix_index, feature_name in enumerate(ranking, start=1):
        running_mask |= 1 << feature_index[feature_name]
        prefix_value = float(coalition_array[running_mask])
        if prefix_index < len(feature_names):
            strict_prefix_values.append(prefix_value)
        normalized_sum += float(
            np.clip(prefix_value, 0.0, full_value) / full_value
        )

    if strict_prefix_values and max(strict_prefix_values) <= 0:
        return 0.0

    return normalized_sum / (len(feature_names) + 1)


def compute_decision_deletion_auc(
    attributions: Sequence[float] | np.ndarray,
    coalition_values: Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
) -> float | None:
    attribution_array = _validate_feature_aligned_vector(attributions, feature_names)
    coalition_array = _validate_scalar_coalition_values(
        coalition_values,
        feature_count=len(feature_names),
    )
    full_mask = len(coalition_array) - 1
    full_value = float(coalition_array[full_mask])
    if full_value <= 0:
        return None

    ranking = build_attribution_ranking(attribution_array, feature_names)
    feature_index = {feature_name: index for index, feature_name in enumerate(feature_names)}

    running_mask = full_mask
    strict_suffix_values: list[float] = []
    normalized_sum = float(np.clip(coalition_array[full_mask], 0.0, full_value) / full_value)
    for prefix_index, feature_name in enumerate(ranking, start=1):
        running_mask &= ~(1 << feature_index[feature_name])
        retained_value = float(coalition_array[running_mask])
        if prefix_index < len(feature_names):
            strict_suffix_values.append(retained_value)
        normalized_sum += float(
            np.clip(retained_value, 0.0, full_value) / full_value
        )

    if strict_suffix_values and max(strict_suffix_values) <= 0:
        return 0.0

    return normalized_sum / (len(feature_names) + 1)


def compute_exact_decision_infidelity(
    attributions: Sequence[float] | np.ndarray,
    coalition_values: Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
) -> float:
    attribution_array = _validate_feature_aligned_vector(attributions, feature_names)
    coalition_array = _validate_scalar_coalition_values(
        coalition_values,
        feature_count=len(feature_names),
    )
    full_mask = len(coalition_array) - 1
    squared_errors = []
    for removed_mask in range(1, full_mask + 1):
        retained_mask = full_mask ^ removed_mask
        actual_loss = float(coalition_array[full_mask] - coalition_array[retained_mask])
        predicted_loss = 0.0
        for feature_index, attribution_value in enumerate(attribution_array):
            if removed_mask & (1 << feature_index):
                predicted_loss += float(attribution_value)
        squared_errors.append((actual_loss - predicted_loss) ** 2)
    return float(np.mean(squared_errors))


def build_metric_summary(
    values_by_date: Mapping[str, float | None],
) -> dict[str, Any]:
    ordered_dates = list(values_by_date)
    resolved_values = [values_by_date[date] for date in ordered_dates]
    valid_values = np.asarray(
        [value for value in resolved_values if value is not None],
        dtype=float,
    )
    total_days = len(ordered_dates)
    valid_days = int(len(valid_values))
    return {
        "values_by_date": {
            date: (None if value is None else float(value))
            for date, value in values_by_date.items()
        },
        "mean": float(np.mean(valid_values)) if valid_days else None,
        "median": float(np.median(valid_values)) if valid_days else None,
        "std": float(np.std(valid_values)) if valid_days else None,
        "valid_days": valid_days,
        "total_days": total_days,
        "coverage": (float(valid_days / total_days) if total_days else None),
    }


def extract_local_attributions(
    daily_shap: pd.DataFrame,
    feature_names: Sequence[str],
    explainer_family: str,
    *,
    dates: Sequence[str] | None = None,
) -> pd.DataFrame:
    columns = [f"{explainer_family}_shap_{feature_name}" for feature_name in feature_names]
    missing_columns = sorted(set(columns) - set(daily_shap.columns))
    if missing_columns:
        raise KeyError(
            "daily_shap is missing required explainer columns: "
            + ", ".join(missing_columns)
        )

    frame = daily_shap.loc[:, ["date", *columns]].copy()
    frame["date"] = frame["date"].astype(str)
    if dates is None:
        return frame

    wanted_dates = [str(date) for date in dates]
    filtered = frame[frame["date"].isin(wanted_dates)].copy()
    if len(filtered) != len(wanted_dates):
        missing_dates = sorted(set(wanted_dates) - set(filtered["date"]))
        raise ValueError("Missing requested dates in daily_shap: " + ", ".join(missing_dates))
    return filtered


def compute_global_importance(
    daily_shap: pd.DataFrame,
    feature_names: Sequence[str],
    explainer_family: str,
    *,
    dates: Sequence[str] | None = None,
) -> dict[str, float]:
    local_frame = extract_local_attributions(
        daily_shap,
        feature_names,
        explainer_family,
        dates=dates,
    )
    importance_by_feature = {}
    for feature_name in feature_names:
        column_name = f"{explainer_family}_shap_{feature_name}"
        importance_by_feature[feature_name] = float(local_frame[column_name].abs().mean())
    return importance_by_feature


def compute_top_k_jaccard(
    left_ranking: Sequence[str],
    right_ranking: Sequence[str],
    *,
    k: int = DEFAULT_TOP_K,
) -> float | None:
    effective_k = min(k, len(left_ranking), len(right_ranking))
    if effective_k <= 0:
        return None
    left_top = set(left_ranking[:effective_k])
    right_top = set(right_ranking[:effective_k])
    union = left_top | right_top
    if not union:
        return None
    return float(len(left_top & right_top) / len(union))


def compute_truncated_rbo(
    left_ranking: Sequence[str],
    right_ranking: Sequence[str],
    *,
    depth: int = DEFAULT_RBO_DEPTH,
    p: float = DEFAULT_RBO_P,
) -> float | None:
    if not 0 < p < 1:
        raise ValueError("p must lie strictly between 0 and 1.")
    effective_depth = min(depth, len(left_ranking), len(right_ranking))
    if effective_depth <= 0:
        return None

    left_seen: set[str] = set()
    right_seen: set[str] = set()
    prefix_overlaps: list[float] = []

    for prefix_depth in range(1, effective_depth + 1):
        left_seen.add(str(left_ranking[prefix_depth - 1]))
        right_seen.add(str(right_ranking[prefix_depth - 1]))
        prefix_overlaps.append(len(left_seen & right_seen) / prefix_depth)

    weighted_overlap_sum = sum(
        overlap * (p**prefix_depth)
        for prefix_depth, overlap in enumerate(prefix_overlaps, start=1)
    )
    return float(
        ((1 - p) / p) * weighted_overlap_sum
        + prefix_overlaps[-1] * (p**effective_depth)
    )


def compute_rank_spearman_from_rankings(
    left_ranking: Sequence[str],
    right_ranking: Sequence[str],
) -> float | None:
    left_positions, right_positions = _resolve_rank_position_vectors(
        left_ranking,
        right_ranking,
    )
    return compute_spearman_rank_correlation(left_positions, right_positions)


def compute_rank_kendall_tau_from_rankings(
    left_ranking: Sequence[str],
    right_ranking: Sequence[str],
) -> float | None:
    left_positions, right_positions = _resolve_rank_position_vectors(
        left_ranking,
        right_ranking,
    )
    return compute_kendall_tau_correlation(left_positions, right_positions)


def compute_normalized_importance_vector(
    scores_by_feature: Mapping[str, float],
    feature_names: Sequence[str],
) -> dict[str, float] | None:
    if set(scores_by_feature) != set(feature_names):
        raise ValueError("scores_by_feature must contain exactly one score per feature.")

    total = float(sum(float(scores_by_feature[feature_name]) for feature_name in feature_names))
    if total <= 0:
        return None

    return {
        feature_name: float(scores_by_feature[feature_name] / total)
        for feature_name in feature_names
    }


def compute_normalized_importance_l1(
    left_scores: Mapping[str, float],
    right_scores: Mapping[str, float],
    feature_names: Sequence[str],
) -> float | None:
    left_normalized = compute_normalized_importance_vector(left_scores, feature_names)
    right_normalized = compute_normalized_importance_vector(right_scores, feature_names)
    if left_normalized is None or right_normalized is None:
        return None

    return float(
        sum(
            abs(left_normalized[feature_name] - right_normalized[feature_name])
            for feature_name in feature_names
        )
    )


def build_daily_dispatch_policy_lookup(
    daily_full_dispatch: pd.DataFrame,
) -> dict[str, DailyDispatchPolicy]:
    required_columns = {
        "date",
        "hour",
        "charge",
        "discharge",
        "mode",
    }
    missing_columns = sorted(required_columns - set(daily_full_dispatch.columns))
    if missing_columns:
        raise KeyError(
            "daily_full_dispatch is missing required columns: "
            + ", ".join(missing_columns)
        )

    policy_lookup: dict[str, DailyDispatchPolicy] = {}
    normalized_frame = daily_full_dispatch.copy()
    normalized_frame["date"] = normalized_frame["date"].astype(str)
    normalized_frame = normalized_frame.sort_values(["date", "hour"]).reset_index(drop=True)
    for date, date_frame in normalized_frame.groupby("date", sort=False):
        date_str = cast(str, date)
        charge = tuple(float(value) for value in date_frame["charge"])
        discharge = tuple(float(value) for value in date_frame["discharge"])
        mode = tuple(int(value) for value in date_frame["mode"])
        policy_lookup[date_str] = DailyDispatchPolicy(
            date=date_str,
            charge=charge,
            discharge=discharge,
            mode=mode,
        )
    return policy_lookup


def identify_invariant_policy_days(
    left_daily_full_dispatch: pd.DataFrame,
    right_daily_full_dispatch: pd.DataFrame,
    *,
    atol: float = DEFAULT_POLICY_ATOL,
) -> list[str]:
    left_lookup = build_daily_dispatch_policy_lookup(left_daily_full_dispatch)
    right_lookup = build_daily_dispatch_policy_lookup(right_daily_full_dispatch)
    if set(left_lookup) != set(right_lookup):
        raise ValueError("Both settings must contain the same dates in daily_full_dispatch.")

    invariant_dates = []
    for date in sorted(left_lookup):
        left_policy = left_lookup[date]
        right_policy = right_lookup[date]
        if _policies_match(left_policy, right_policy, atol=atol):
            invariant_dates.append(date)
    return invariant_dates


def compute_spearman_rank_correlation(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> float | None:
    prepared = _prepare_correlation_inputs(left, right)
    if prepared is None:
        return None

    left_series = pd.Series(prepared[0], dtype=float)
    right_series = pd.Series(prepared[1], dtype=float)
    left_ranks = left_series.rank(method="average")
    right_ranks = right_series.rank(method="average")
    correlation = left_ranks.corr(right_ranks)
    if pd.isna(correlation):
        return None
    return float(correlation)


def compute_kendall_tau_correlation(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> float | None:
    """Compute Kendall's tau-b correlation, including correction for ties."""
    prepared = _prepare_correlation_inputs(left, right)
    if prepared is None:
        return None

    correlation = kendalltau(
        prepared[0],
        prepared[1],
        variant="b",
        nan_policy="omit",
    ).statistic
    if correlation is None or pd.isna(correlation):
        return None
    return float(correlation)


def _resolve_rank_position_vectors(
    left_ranking: Sequence[str],
    right_ranking: Sequence[str],
) -> tuple[list[float], list[float]]:
    if len(left_ranking) != len(right_ranking):
        raise ValueError("Rankings must have the same length.")
    if set(left_ranking) != set(right_ranking):
        raise ValueError("Rankings must contain the same items.")
    if len(left_ranking) <= 1:
        return [], []

    left_positions = {feature_name: index + 1 for index, feature_name in enumerate(left_ranking)}
    right_positions = {
        feature_name: index + 1 for index, feature_name in enumerate(right_ranking)
    }
    ordered_features = list(left_ranking)
    return (
        [float(left_positions[feature_name]) for feature_name in ordered_features],
        [float(right_positions[feature_name]) for feature_name in ordered_features],
    )


def _prepare_correlation_inputs(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        raise ValueError(
            "left and right must share the same shape for correlation computation. "
            f"Got {left_array.shape} and {right_array.shape}."
        )
    if left_array.ndim != 1:
        raise ValueError("left and right must be one-dimensional sequences.")

    valid_mask = ~(np.isnan(left_array) | np.isnan(right_array))
    left_valid = left_array[valid_mask]
    right_valid = right_array[valid_mask]
    if len(left_valid) <= 1:
        return None
    if np.unique(left_valid).size <= 1 or np.unique(right_valid).size <= 1:
        return None
    return left_valid, right_valid


def _policies_match(
    left_policy: DailyDispatchPolicy,
    right_policy: DailyDispatchPolicy,
    *,
    atol: float,
) -> bool:
    return (
        left_policy.mode == right_policy.mode
        and np.allclose(left_policy.charge, right_policy.charge, atol=atol, rtol=0.0)
        and np.allclose(
            left_policy.discharge,
            right_policy.discharge,
            atol=atol,
            rtol=0.0,
        )
    )


def _validate_feature_aligned_vector(
    values: Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    expected_shape = (len(feature_names),)
    if array.shape != expected_shape:
        raise ValueError(
            "values must contain one entry per feature. "
            f"Expected shape {expected_shape}, got {array.shape}."
        )
    return array


def _validate_scalar_coalition_values(
    coalition_values: Sequence[float] | np.ndarray,
    *,
    feature_count: int,
) -> np.ndarray:
    array = np.asarray(coalition_values, dtype=float)
    expected_shape = (1 << feature_count,)
    if array.shape != expected_shape:
        raise ValueError(
            "coalition_values must contain one scalar value per coalition. "
            f"Expected shape {expected_shape}, got {array.shape}."
        )
    return array


def _activation_subset_weights(feature_count: int) -> tuple[float, ...]:
    denominator = math.factorial(feature_count)
    return tuple(
        math.factorial(subset_size)
        * math.factorial(feature_count - subset_size - 1)
        / denominator
        for subset_size in range(feature_count)
    )
