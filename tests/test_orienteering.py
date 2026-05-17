from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dva.model.orienteering import (
    solve_orienteering,
    solve_orienteering_heuristic,
    solve_orienteering_ortools,
)


def test_solve_orienteering_finds_best_closed_tour() -> None:
    zone_ids = [0, 1, 2, 3]
    distance_matrix = pd.DataFrame(
        [
            [0.0, 1.0, 1.5, 4.0],
            [1.0, 0.0, 1.0, 3.0],
            [1.5, 1.0, 0.0, 3.0],
            [4.0, 3.0, 3.0, 0.0],
        ],
        index=zone_ids,
        columns=zone_ids,
    )
    zone_scores = {1: 10.0, 2: 8.0, 3: 5.0}

    result = solve_orienteering(
        zone_scores=zone_scores,
        max_distance_budget=4.0,
        distance_matrix=distance_matrix,
        start_zone_id=0,
    )

    assert result.optimal
    assert result.method == "exact"
    assert result.route_zone_ids[0] == 0
    assert result.route_zone_ids[-1] == 0
    assert set(result.visited_zone_ids) == {1, 2}
    assert result.collected_score == pytest.approx(18.0)
    assert result.total_distance <= 4.0 + 1e-9


def test_solve_orienteering_supports_distinct_start_and_end() -> None:
    zone_ids = [0, 1, 2, 3]
    distance_matrix = pd.DataFrame(
        [
            [0.0, 1.0, 1.0, 4.0],
            [1.0, 0.0, 10.0, 1.0],
            [1.0, 10.0, 0.0, 2.0],
            [4.0, 1.0, 2.0, 0.0],
        ],
        index=zone_ids,
        columns=zone_ids,
    )

    result = solve_orienteering(
        zone_scores={1: 9.0, 2: 5.0},
        max_distance_budget=2.1,
        distance_matrix=distance_matrix,
        start_zone_id=0,
        end_zone_id=3,
    )

    assert result.optimal
    assert result.method == "exact"
    assert result.route_zone_ids == (0, 1, 3)
    assert result.visited_zone_ids == (1,)
    assert result.collected_score == pytest.approx(9.0)
    assert result.total_distance == pytest.approx(2.0)


def test_solve_orienteering_raises_when_no_feasible_path_exists() -> None:
    distance_matrix = pd.DataFrame(
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
        ),
        index=[0, 1],
        columns=[0, 1],
    )

    with pytest.raises(RuntimeError, match="did not produce a feasible solution"):
        solve_orienteering(
            zone_scores={},
            max_distance_budget=0.5,
            distance_matrix=distance_matrix,
            start_zone_id=0,
            end_zone_id=1,
        )


def test_solve_orienteering_heuristic_inserts_best_gain_per_distance_nodes() -> None:
    distance_matrix = pd.DataFrame(
        [
            [0.0, 1.0, 1.0, 2.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [2.0, 1.0, 1.0, 0.0],
        ],
        index=[0, 1, 2, 3],
        columns=[0, 1, 2, 3],
    )

    result = solve_orienteering(
        zone_scores={1: 12.0, 2: 8.0, 3: 7.0},
        max_distance_budget=4.0,
        distance_matrix=distance_matrix,
        start_zone_id=0,
        method="heuristic",
    )

    assert result.method == "heuristic"
    assert result.route_zone_ids == (0, 1, 3, 2, 0)
    assert result.collected_score == pytest.approx(27.0)
    assert result.total_distance == pytest.approx(4.0)
    assert not result.optimal
    assert result.solver_status is None


def test_solve_orienteering_heuristic_function_supports_open_path() -> None:
    distance_matrix = pd.DataFrame(
        [
            [0.0, 1.0, 3.0, 3.0],
            [1.0, 0.0, 1.0, 1.0],
            [3.0, 1.0, 0.0, 1.0],
            [3.0, 1.0, 1.0, 0.0],
        ],
        index=[0, 1, 2, 3],
        columns=[0, 1, 2, 3],
    )

    result = solve_orienteering_heuristic(
        zone_scores={1: 6.0, 2: 5.0},
        max_distance_budget=2.1,
        distance_matrix=distance_matrix,
        start_zone_id=0,
        end_zone_id=3,
    )

    assert result.method == "heuristic"
    assert result.route_zone_ids == (0, 1, 3)
    assert result.collected_score == pytest.approx(6.0)
    assert result.total_distance == pytest.approx(2.0)


def test_solve_orienteering_ortools_finds_high_prize_closed_tour() -> None:
    zone_ids = [0, 1, 2, 3]
    distance_matrix = pd.DataFrame(
        [
            [0.0, 1.0, 1.5, 4.0],
            [1.0, 0.0, 1.0, 3.0],
            [1.5, 1.0, 0.0, 3.0],
            [4.0, 3.0, 3.0, 0.0],
        ],
        index=zone_ids,
        columns=zone_ids,
    )

    result = solve_orienteering(
        zone_scores={1: 10.0, 2: 8.0, 3: 5.0},
        max_distance_budget=4.0,
        distance_matrix=distance_matrix,
        start_zone_id=0,
        method="ortools",
        solver_params={"time_limit_s": 1},
    )

    assert result.method == "ortools"
    assert result.route_zone_ids[0] == 0
    assert result.route_zone_ids[-1] == 0
    assert set(result.visited_zone_ids) == {1, 2}
    assert result.collected_score == pytest.approx(18.0)
    assert result.total_distance <= 4.0 + 1e-9
    assert not result.optimal


def test_solve_orienteering_ortools_supports_open_path_and_fractional_scores() -> None:
    distance_matrix = pd.DataFrame(
        [
            [0.0, 1.0, 1.0, 4.0],
            [1.0, 0.0, 10.0, 1.0],
            [1.0, 10.0, 0.0, 2.0],
            [4.0, 1.0, 2.0, 0.0],
        ],
        index=[0, 1, 2, 3],
        columns=[0, 1, 2, 3],
    )

    result = solve_orienteering_ortools(
        zone_scores={1: 0.9, 2: 0.5},
        max_distance_budget=2.1,
        distance_matrix=distance_matrix,
        start_zone_id=0,
        end_zone_id=3,
        time_limit_s=1,
    )

    assert result.method == "ortools"
    assert result.route_zone_ids == (0, 1, 3)
    assert result.visited_zone_ids == (1,)
    assert result.collected_score == pytest.approx(0.9)
    assert result.total_distance == pytest.approx(2.0)
