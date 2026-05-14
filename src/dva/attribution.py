from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from dva.analysis.caiso_shap import (
    compute_exact_faith_shap_values,
    compute_exact_interaction_values,
    compute_exact_shapley_taylor_values,
    compute_exact_shapley_values,
    compute_mobius_transform,
)
from dva.analysis.ems_exact_shap import (
    compute_kernel_shapley_values,
    compute_permutation_shapley_values,
)


def compute_perturbation_shapley_values(
    coalition_values: np.ndarray,
    player_count: int,
    *,
    sample_count: int,
    random_state: int | Sequence[int] | None = None,
) -> np.ndarray:
    """Estimate Shapley values by sampled perturbation/permutation paths."""

    return compute_permutation_shapley_values(
        coalition_values,
        feature_count=player_count,
        sample_count=sample_count,
        random_state=random_state,
    )


def compute_dvi_values(
    coalition_values: np.ndarray,
    player_count: int,
    *,
    order: int = 2,
    method: str = "faith_shap",
) -> dict[frozenset[int], Any]:
    """Compute Decision Value Interactions with Faith-SHAP by default."""

    return compute_exact_interaction_values(
        coalition_values,
        player_count=player_count,
        order=order,
        method=method,
    )


__all__ = [
    "compute_dvi_values",
    "compute_exact_faith_shap_values",
    "compute_exact_interaction_values",
    "compute_exact_shapley_taylor_values",
    "compute_exact_shapley_values",
    "compute_kernel_shapley_values",
    "compute_mobius_transform",
    "compute_permutation_shapley_values",
    "compute_perturbation_shapley_values",
]
