from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dva.analysis.run_ems_decision_shap_approximation_experiment import (
    DEFAULT_SAMPLE_BUDGETS,
    DEFAULT_SEED_COUNT,
    _approximation_oracle_calls,
    _compute_approximation_metrics,
    _resolve_approximation_seeds,
    _summarize_metrics,
    build_parser,
)


def test_approximation_experiment_parser_defaults_to_requested_grid() -> None:
    args = build_parser().parse_args([])

    assert args.coverage_radius_km == pytest.approx(1.0)
    assert args.sample_budget == list(DEFAULT_SAMPLE_BUDGETS)
    assert args.seed_count == DEFAULT_SEED_COUNT
    assert args.method is None


def test_approximation_experiment_resolves_explicit_or_counted_seeds() -> None:
    parser = build_parser()

    explicit_args = parser.parse_args(["--approximation-seed", "7", "8", "7"])
    counted_args = parser.parse_args(["--seed-start", "20", "--seed-count", "3"])

    assert _resolve_approximation_seeds(explicit_args) == (7, 8)
    assert _resolve_approximation_seeds(counted_args) == (20, 21, 22)


def test_compute_approximation_metrics_uses_pooled_nmae_denominator() -> None:
    exact = np.array(
        [
            [1.0, -3.0],
            [2.0, 4.0],
        ],
        dtype=float,
    )
    approximation = np.array(
        [
            [2.0, -1.0],
            [1.0, 7.0],
        ],
        dtype=float,
    )

    metrics = _compute_approximation_metrics(
        approximation,
        exact,
        ("a", "b"),
        exact_denominator=float(np.abs(exact).sum()),
    )

    assert metrics["absolute_error_sum"] == pytest.approx(7.0)
    assert metrics["exact_abs_shap_sum"] == pytest.approx(10.0)
    assert metrics["nmae"] == pytest.approx(0.7)
    assert metrics["global_top1_match"] == 1
    assert metrics["global_kendall_tau_b"] == pytest.approx(1.0)


def test_approximation_oracle_calls_count_sampled_unique_masks() -> None:
    permutation_calls = _approximation_oracle_calls(
        "permutation",
        feature_count=3,
        sample_budget=2,
        seed=0,
        hour_count=2,
    )
    kernel_calls = _approximation_oracle_calls(
        "kernel",
        feature_count=3,
        sample_budget=2,
        seed=0,
        hour_count=2,
    )
    replacement_kernel_calls = _approximation_oracle_calls(
        "kernel",
        feature_count=3,
        sample_budget=7,
        seed=0,
        hour_count=2,
    )

    assert 2 <= permutation_calls["unique"] <= 2 * (1 << 3)
    assert permutation_calls["noncached"] == 2 * 2 * (3 + 1)
    assert kernel_calls["unique"] == 2 * (2 + 2)
    assert kernel_calls["noncached"] == 2 * (2 + 2)
    assert 2 * 3 <= replacement_kernel_calls["unique"] <= 2 * (1 << 3)
    assert replacement_kernel_calls["noncached"] == 2 * (7 + 2)


def test_summarize_metrics_bootstraps_by_method_and_budget() -> None:
    raw = pd.DataFrame(
        [
            {
                "method": "permutation",
                "sample_budget": 16,
                "nmae": 0.4,
                "runtime_seconds": 1.0,
                "actual_runner_runtime_seconds": 0.1,
                "transform_runtime_seconds": 0.1,
                "estimated_oracle_runtime_seconds": 0.9,
                "estimated_standalone_runtime_seconds": 1.0,
                "estimated_cached_oracle_runtime_seconds": 0.9,
                "estimated_noncached_oracle_runtime_seconds": 1.2,
                "estimated_standalone_cached_runtime_seconds": 1.0,
                "estimated_standalone_noncached_runtime_seconds": 1.3,
                "global_top1_match": 1,
                "global_kendall_tau_b": 0.5,
                "oracle_calls": 10,
                "standalone_cached_oracle_calls": 10,
                "standalone_unique_oracle_calls": 10,
                "standalone_noncached_oracle_calls": 12,
                "actual_runner_oracle_calls": 0,
            },
            {
                "method": "permutation",
                "sample_budget": 16,
                "nmae": 0.2,
                "runtime_seconds": 2.0,
                "actual_runner_runtime_seconds": 0.2,
                "transform_runtime_seconds": 0.2,
                "estimated_oracle_runtime_seconds": 1.8,
                "estimated_standalone_runtime_seconds": 2.0,
                "estimated_cached_oracle_runtime_seconds": 1.8,
                "estimated_noncached_oracle_runtime_seconds": 2.4,
                "estimated_standalone_cached_runtime_seconds": 2.0,
                "estimated_standalone_noncached_runtime_seconds": 2.6,
                "global_top1_match": 0,
                "global_kendall_tau_b": 0.25,
                "oracle_calls": 11,
                "standalone_cached_oracle_calls": 11,
                "standalone_unique_oracle_calls": 11,
                "standalone_noncached_oracle_calls": 12,
                "actual_runner_oracle_calls": 0,
            },
        ]
    )

    summary = _summarize_metrics(raw, bootstrap_draws=100, bootstrap_seed=0)
    row = summary.iloc[0]

    assert row["method"] == "permutation"
    assert row["sample_budget"] == 16
    assert row["n_runs"] == 2
    assert row["nmae_mean"] == pytest.approx(0.3)
    assert row["runtime_seconds_mean"] == pytest.approx(1.5)
    assert row["actual_runner_runtime_seconds_mean"] == pytest.approx(0.15)
    assert row["standalone_cached_oracle_calls_mean"] == pytest.approx(10.5)
    assert row["standalone_noncached_oracle_calls_mean"] == pytest.approx(12.0)
    assert row["global_top1_match_mean"] == pytest.approx(0.5)
