from __future__ import annotations

from dataclasses import dataclass

from dva.analysis.ems_exact_shap import (
    EMS_COVERAGE_SOLVER_GREEDY_MAX_COVER,
    EMS_COVERAGE_SOLVER_GUROBI,
    EMS_COVERAGE_SOLVER_NAIVE_GREEDY,
)


@dataclass(frozen=True, slots=True)
class EmsDesign:
    name: str
    solver: str
    radius_km: float
    staging_areas: int


EMS_INFO_RADII_KM = (1.0, 2.0, 3.0)
EMS_INFO_STAGING_AREAS = (3, 5, 8)
EMS_DESIGN_ACTUALS = {
    "naive": EmsDesign(
        name="naive",
        solver=EMS_COVERAGE_SOLVER_NAIVE_GREEDY,
        radius_km=1.0,
        staging_areas=3,
    ),
    "greedy": EmsDesign(
        name="greedy",
        solver=EMS_COVERAGE_SOLVER_GREEDY_MAX_COVER,
        radius_km=1.0,
        staging_areas=3,
    ),
}
EMS_DESIGN_BASELINE = EmsDesign(
    name="exact_radius_3_staging_8",
    solver=EMS_COVERAGE_SOLVER_GUROBI,
    radius_km=3.0,
    staging_areas=8,
)


__all__ = [
    "EMS_DESIGN_ACTUALS",
    "EMS_DESIGN_BASELINE",
    "EMS_INFO_RADII_KM",
    "EMS_INFO_STAGING_AREAS",
    "EmsDesign",
]
