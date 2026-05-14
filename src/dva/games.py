from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import numpy as np


class ValueMode(StrEnum):
    POST = "post"
    ANTE = "ante"


class PlayerKind(StrEnum):
    INFO = "info"
    DESIGN = "design"


@dataclass(frozen=True, slots=True)
class Player:
    name: str
    kind: PlayerKind
    baseline: Any = None
    actual: Any = None


@dataclass(frozen=True, slots=True)
class PlayerSet:
    players: tuple[Player, ...]

    def __init__(self, players: Sequence[Player]) -> None:
        object.__setattr__(self, "players", tuple(players))
        names = [player.name for player in self.players]
        if len(names) != len(set(names)):
            raise ValueError("Player names must be unique.")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(player.name for player in self.players)

    @property
    def count(self) -> int:
        return len(self.players)

    @property
    def coalition_count(self) -> int:
        return 1 << self.count

    def kinds_for_subset(self, subset: frozenset[int]) -> frozenset[PlayerKind]:
        return frozenset(self.players[index].kind for index in subset)


class CoalitionValueFunction(Protocol):
    def __call__(self, coalition_mask: int, mode: ValueMode) -> float | np.ndarray:
        ...


@dataclass(frozen=True, slots=True)
class DVAGame:
    name: str
    players: PlayerSet
    value_function: CoalitionValueFunction
    mode: ValueMode = ValueMode.POST

    def coalition_values(self) -> np.ndarray:
        values = [
            self.value_function(mask, self.mode)
            for mask in range(self.players.coalition_count)
        ]
        return np.asarray(values, dtype=float)

    def characteristic_values(self) -> np.ndarray:
        values = self.coalition_values()
        return values - values[0]


@dataclass(frozen=True, slots=True)
class InfoDVAGame(DVAGame):
    pass


@dataclass(frozen=True, slots=True)
class DesignDVAGame(DVAGame):
    pass


@dataclass(frozen=True, slots=True)
class JointDVAGame(DVAGame):
    pass


@dataclass(frozen=True, slots=True)
class DVIInteraction:
    players: tuple[str, ...]
    indices: tuple[int, ...]
    interaction_type: str
    value: Any


def build_info_players(names: Sequence[str]) -> PlayerSet:
    return PlayerSet(Player(str(name), PlayerKind.INFO) for name in names)


def build_design_players(
    actual: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> PlayerSet:
    missing = sorted(set(actual) ^ set(baseline))
    if missing:
        raise ValueError(
            "actual and baseline design mappings must contain the same keys: "
            + ", ".join(missing)
        )
    return PlayerSet(
        Player(str(name), PlayerKind.DESIGN, baseline=baseline[name], actual=actual[name])
        for name in actual
    )


def build_joint_players(info_names: Sequence[str], design_players: PlayerSet) -> PlayerSet:
    return PlayerSet(
        [
            *(Player(str(name), PlayerKind.INFO) for name in info_names),
            *design_players.players,
        ]
    )


def classify_interaction(players: PlayerSet, subset: frozenset[int]) -> str:
    kinds = players.kinds_for_subset(subset)
    if kinds == frozenset({PlayerKind.INFO}):
        return "Info-Info"
    if kinds == frozenset({PlayerKind.DESIGN}):
        return "Design-Design"
    return "Cross-DVI"


def materialize_dvi_interactions(
    players: PlayerSet,
    interaction_values: Mapping[frozenset[int], Any],
) -> tuple[DVIInteraction, ...]:
    rows: list[DVIInteraction] = []
    for subset, value in sorted(
        interaction_values.items(),
        key=lambda item: (len(item[0]), tuple(sorted(item[0]))),
    ):
        indices = tuple(sorted(subset))
        rows.append(
            DVIInteraction(
                players=tuple(players.names[index] for index in indices),
                indices=indices,
                interaction_type=classify_interaction(players, subset),
                value=value,
            )
        )
    return tuple(rows)


def positive_ante_value_gain(
    coalition_value: float | np.ndarray,
    empty_value: float | np.ndarray,
) -> float | np.ndarray:
    return np.asarray(coalition_value) - np.asarray(empty_value)


__all__ = [
    "DVAGame",
    "DVIInteraction",
    "DesignDVAGame",
    "InfoDVAGame",
    "JointDVAGame",
    "Player",
    "PlayerKind",
    "PlayerSet",
    "ValueMode",
    "build_design_players",
    "build_info_players",
    "build_joint_players",
    "classify_interaction",
    "materialize_dvi_interactions",
    "positive_ante_value_gain",
]
