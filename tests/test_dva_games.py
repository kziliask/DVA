from __future__ import annotations

import numpy as np
import pytest

from dva.case_studies.caiso.designs import (
    CAISO_ACTUAL_DESIGN,
    CAISO_BASELINE_DESIGNS,
    CAISO_INFO_PLAYERS,
    parameter_player_spec_for_baseline,
)
from dva.case_studies.ems.designs import (
    EMS_DESIGN_ACTUALS,
    EMS_DESIGN_BASELINE,
    EMS_INFO_RADII_KM,
    EMS_INFO_STAGING_AREAS,
)
from dva.games import (
    DVAGame,
    PlayerKind,
    ValueMode,
    build_design_players,
    build_info_players,
    build_joint_players,
    classify_interaction,
    materialize_dvi_interactions,
    positive_ante_value_gain,
)


def test_dva_game_player_definitions_and_positive_ante_gain() -> None:
    info_players = build_info_players(CAISO_INFO_PLAYERS)
    design_players = build_design_players(
        {"throughput_penalty": 5.0, "energy_capacity": 4.0},
        {"throughput_penalty": 10.0, "energy_capacity": 2.0},
    )
    joint_players = build_joint_players(info_players.names, design_players)

    assert info_players.count == 8
    assert all(player.kind == PlayerKind.INFO for player in info_players.players)
    assert tuple(player.name for player in design_players.players) == (
        "throughput_penalty",
        "energy_capacity",
    )
    assert joint_players.count == 10
    assert positive_ante_value_gain(7.0, 3.0) == pytest.approx(4.0)
    np.testing.assert_allclose(
        positive_ante_value_gain(np.array([2.0, 5.0]), np.array([1.0, 1.5])),
        np.array([1.0, 3.5]),
    )


def test_dva_game_characteristic_values_use_empty_coalition_baseline() -> None:
    game = DVAGame(
        name="toy",
        players=build_info_players(("a", "b")),
        mode=ValueMode.ANTE,
        value_function=lambda mask, mode: float(mask + (10 if mode == ValueMode.ANTE else 0)),
    )

    np.testing.assert_allclose(game.coalition_values(), np.array([10.0, 11.0, 12.0, 13.0]))
    np.testing.assert_allclose(game.characteristic_values(), np.array([0.0, 1.0, 2.0, 3.0]))


def test_cross_dvi_labeling() -> None:
    design_players = build_design_players({"solver": "greedy"}, {"solver": "gurobi"})
    players = build_joint_players(("hour", "temp_c"), design_players)

    assert classify_interaction(players, frozenset({0, 1})) == "Info-Info"
    assert classify_interaction(players, frozenset({2})) == "Design-Design"
    assert classify_interaction(players, frozenset({1, 2})) == "Cross-DVI"

    interactions = materialize_dvi_interactions(
        players,
        {
            frozenset({0, 1}): 1.0,
            frozenset({0, 2}): 2.0,
        },
    )

    assert [interaction.interaction_type for interaction in interactions] == [
        "Info-Info",
        "Cross-DVI",
    ]


def test_caiso_design_player_baselines() -> None:
    assert CAISO_ACTUAL_DESIGN.parameters().energy_capacity == pytest.approx(4.0)
    assert CAISO_ACTUAL_DESIGN.parameters().throughput_penalty == pytest.approx(5.0)
    assert CAISO_ACTUAL_DESIGN.parameters().charge_efficiency == pytest.approx(0.95)

    conservative = CAISO_BASELINE_DESIGNS["conservative"]
    optimistic = CAISO_BASELINE_DESIGNS["optimistic"]
    assert conservative.parameters().energy_capacity == pytest.approx(2.0)
    assert conservative.parameters().throughput_penalty == pytest.approx(10.0)
    assert conservative.parameters().charge_efficiency == pytest.approx(0.8)
    assert optimistic.parameters().energy_capacity == pytest.approx(24.0)
    assert optimistic.parameters().throughput_penalty == pytest.approx(0.0)
    assert optimistic.parameters().charge_efficiency == pytest.approx(1.0)

    spec = parameter_player_spec_for_baseline(conservative)
    assert spec.throughput_penalty_baseline == pytest.approx(10.0)
    assert spec.energy_capacity_baseline == pytest.approx(2.0)
    assert spec.charge_efficiency_baseline == pytest.approx(0.8)


def test_ems_design_player_baselines() -> None:
    assert EMS_INFO_RADII_KM == (1.0, 2.0, 3.0)
    assert EMS_INFO_STAGING_AREAS == (3, 5, 8)
    assert EMS_DESIGN_BASELINE.solver == "gurobi"
    assert EMS_DESIGN_BASELINE.radius_km == pytest.approx(3.0)
    assert EMS_DESIGN_BASELINE.staging_areas == 8
    assert EMS_DESIGN_ACTUALS["naive"].solver == "naive_greedy"
    assert EMS_DESIGN_ACTUALS["greedy"].solver == "greedy_max_cover"
    assert EMS_DESIGN_ACTUALS["naive"].radius_km == pytest.approx(1.0)
    assert EMS_DESIGN_ACTUALS["greedy"].staging_areas == 3
