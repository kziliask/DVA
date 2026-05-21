from __future__ import annotations

import pytest

from dva.analysis.additional_decision_ranking_baselines import (
    compute_additional_baselines_from_coalitions,
)
from dva.analysis.paired_bootstrap import BootstrapConfig


def test_additional_baselines_compute_local_and_aggregate_auc() -> None:
    result = compute_additional_baselines_from_coalitions(
        dataset="Toy",
        unit_ids=("a", "b"),
        player_names=("x1", "x2"),
        decision_characteristic_by_unit={
            "a": [0.0, 3.0, 1.0, 4.0],
            "b": [0.0, 1.0, 4.0, 5.0],
        },
        pfi_scores={"x1": 2.0, "x2": 1.0},
        pfi_local_scores_by_unit={
            "a": {"x1": 2.0, "x2": 0.5},
            "b": {"x1": 2.0, "x2": 1.5},
        },
        bootstrap_config=BootstrapConfig(n_bootstrap=100, seed=0),
    )

    local = result.local_rows
    assert set(local["method"]) == {
        "Leave-one-feature-out",
        "Downstream permutation feature importance",
        "Greedy decision insertion",
    }
    assert len(local) == 6
    assert local["insertion_auc"].between(0.0, 1.0).all()

    aggregate = result.aggregate_rows.set_index("method")
    assert aggregate.loc["Greedy decision insertion", "ranking"] == '["x2", "x1"]'
    assert aggregate.loc["Leave-one-feature-out", "n_units"] == pytest.approx(2)


def test_paper_original_uses_signed_global_pfi_ranking() -> None:
    result = compute_additional_baselines_from_coalitions(
        dataset="Toy",
        unit_ids=("a", "b"),
        player_names=("x1", "x2"),
        decision_characteristic_by_unit={
            "a": [0.0, 2.0, 4.0, 5.0],
            "b": [0.0, 2.0, 4.0, 5.0],
        },
        pfi_local_scores_by_unit={
            "a": {"x1": -100.0, "x2": 1.0},
            "b": {"x1": -100.0, "x2": 1.0},
        },
        ranking_mode="paper_original",
        bootstrap_config=BootstrapConfig(n_bootstrap=100, seed=0),
    )

    pfi_rows = result.local_rows.loc[
        result.local_rows["method"] == "Downstream permutation feature importance"
    ]
    assert set(pfi_rows["score_scope"]) == {"global_signed"}
    assert set(pfi_rows["ranking"]) == {'["x2", "x1"]'}
    assert set(pfi_rows["local_score_x1"]) == {-100.0}
