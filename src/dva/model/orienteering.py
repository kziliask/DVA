from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import pyomo.environ as pyo
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from dva.optimization import (
    DEFAULT_OPTIMIZATION_SOLVER,
    normalize_optimization_solver,
    pyomo_mip_gap,
    pyomo_solver_status,
    require_pyomo_solution,
    solve_pyomo_model,
)


DEFAULT_START_ZONE_ID = 161
DEFAULT_MAX_DISTANCE_BUDGET = 10.0
DEFAULT_ZONE_DISTANCE_MATRIX_PATH = Path("data/nyc_data/zone_centroid_distance_matrix_km.parquet")
_SOURCE_NODE = "__source__"
_SINK_NODE = "__sink__"
_NUMERICAL_TOLERANCE = 1e-9
DEFAULT_ORTOOLS_TIME_LIMIT_S = 0.05
DEFAULT_ORTOOLS_LOCAL_SEARCH_METAHEURISTIC = "GREEDY_DESCENT"
_DEFAULT_ORTOOLS_DISTANCE_SCALE = 1_000_000
_DEFAULT_ORTOOLS_SCORE_SCALE = 1_000_000

type NodeKey = int | str
type OrienteeringSolveMethod = Literal["exact", "heuristic", "ortools"]


@dataclass(slots=True)
class OrienteeringModel:
    model: pyo.ConcreteModel
    zone_scores: dict[int, float]
    distance_matrix: pd.DataFrame
    start_zone_id: int
    end_zone_id: int
    max_distance_budget: float
    customer_zone_ids: tuple[int, ...]
    arc_distances: dict[tuple[NodeKey, NodeKey], float]
    outgoing_arcs: dict[NodeKey, tuple[NodeKey, ...]]
    incoming_arcs: dict[NodeKey, tuple[NodeKey, ...]]
    x: pyo.Var
    y: pyo.Var
    optimization_solver: str = DEFAULT_OPTIMIZATION_SOLVER
    solver_params: dict[str, float | int | str] | None = None
    log_to_console: bool = False
    solve_status: str | None = None
    optimal: bool = False
    mip_gap: float | None = None


@dataclass(frozen=True, slots=True)
class OrienteeringResult:
    method: OrienteeringSolveMethod
    objective_value: float
    collected_score: float
    total_distance: float
    max_distance_budget: float
    start_zone_id: int
    end_zone_id: int
    route_zone_ids: tuple[int, ...]
    visited_zone_ids: tuple[int, ...]
    route_arcs: tuple[tuple[int, int], ...]
    visited_scores: tuple[tuple[int, float], ...]
    solver_status: int | str | None
    optimal: bool
    mip_gap: float | None

    @property
    def remaining_budget(self) -> float:
        return self.max_distance_budget - self.total_distance


def load_zone_distance_matrix(
    path: str | Path = DEFAULT_ZONE_DISTANCE_MATRIX_PATH,
) -> pd.DataFrame:
    matrix_path = Path(path)
    if not matrix_path.exists():
        raise FileNotFoundError(f"Distance matrix not found at {matrix_path}.")

    if matrix_path.suffix == ".parquet":
        distance_matrix = pd.read_parquet(matrix_path)
        if "zone_id" in distance_matrix.columns:
            distance_matrix = distance_matrix.set_index("zone_id")
    elif matrix_path.suffix == ".csv":
        distance_matrix = pd.read_csv(matrix_path)
        if "zone_id" not in distance_matrix.columns:
            raise ValueError("CSV distance matrix must include a zone_id column.")
        distance_matrix = distance_matrix.set_index("zone_id")
    else:
        raise ValueError("Distance matrix must be a parquet or CSV file.")

    distance_matrix.index = distance_matrix.index.astype(int)
    distance_matrix.columns = [int(column) for column in distance_matrix.columns]
    distance_matrix = distance_matrix.sort_index().sort_index(axis=1)

    if distance_matrix.isna().any().any():
        raise ValueError("Distance matrix contains missing values.")
    return distance_matrix.astype(float)


def build_orienteering_model(
    zone_scores: Mapping[int, float] | pd.Series,
    max_distance_budget: float = DEFAULT_MAX_DISTANCE_BUDGET,
    *,
    distance_matrix: pd.DataFrame | None = None,
    start_zone_id: int = DEFAULT_START_ZONE_ID,
    end_zone_id: int | None = None,
    name: str = "orienteering",
    log_to_console: bool = False,
    solver_params: Mapping[str, float | int | str] | None = None,
    optimization_solver: str = DEFAULT_OPTIMIZATION_SOLVER,
) -> OrienteeringModel:
    normalized_scores = _normalize_zone_scores(zone_scores)
    if max_distance_budget < 0:
        raise ValueError("max_distance_budget must be non-negative.")

    resolved_end_zone_id = start_zone_id if end_zone_id is None else int(end_zone_id)
    resolved_distance_matrix = (
        load_zone_distance_matrix() if distance_matrix is None else _normalize_distance_matrix(distance_matrix)
    )
    candidate_zone_ids = tuple(sorted(set(normalized_scores) | {int(start_zone_id), resolved_end_zone_id}))
    _validate_distance_matrix_coverage(
        distance_matrix=resolved_distance_matrix,
        required_zone_ids=candidate_zone_ids,
    )

    customer_zone_ids = tuple(
        zone_id for zone_id in candidate_zone_ids if zone_id not in {int(start_zone_id), resolved_end_zone_id}
    )
    arc_distances = _build_arc_distances(
        distance_matrix=resolved_distance_matrix,
        customer_zone_ids=customer_zone_ids,
        start_zone_id=int(start_zone_id),
        end_zone_id=resolved_end_zone_id,
    )
    outgoing_arcs, incoming_arcs = _build_adjacency_maps(arc_distances)

    model = _build_orienteering_pyomo_model(
        name=name,
        normalized_scores=normalized_scores,
        max_distance_budget=max_distance_budget,
        customer_zone_ids=customer_zone_ids,
        arc_distances=arc_distances,
        outgoing_arcs=outgoing_arcs,
        incoming_arcs=incoming_arcs,
    )

    orienteering_model = OrienteeringModel(
        model=model,
        zone_scores=normalized_scores,
        distance_matrix=resolved_distance_matrix.loc[candidate_zone_ids, candidate_zone_ids].copy(),
        start_zone_id=int(start_zone_id),
        end_zone_id=resolved_end_zone_id,
        max_distance_budget=float(max_distance_budget),
        customer_zone_ids=customer_zone_ids,
        arc_distances=arc_distances,
        outgoing_arcs=outgoing_arcs,
        incoming_arcs=incoming_arcs,
        x=model.x,
        y=model.y,
        optimization_solver=normalize_optimization_solver(optimization_solver),
        solver_params=dict(solver_params or {}),
        log_to_console=log_to_console,
    )
    return orienteering_model


def solve_orienteering(
    zone_scores: Mapping[int, float] | pd.Series,
    max_distance_budget: float = DEFAULT_MAX_DISTANCE_BUDGET,
    *,
    distance_matrix: pd.DataFrame | None = None,
    start_zone_id: int = DEFAULT_START_ZONE_ID,
    end_zone_id: int | None = None,
    method: OrienteeringSolveMethod = "exact",
    name: str = "orienteering",
    log_to_console: bool = False,
    solver_params: Mapping[str, float | int | str] | None = None,
    optimization_solver: str = DEFAULT_OPTIMIZATION_SOLVER,
) -> OrienteeringResult:
    if method == "heuristic":
        return solve_orienteering_heuristic(
            zone_scores=zone_scores,
            max_distance_budget=max_distance_budget,
            distance_matrix=distance_matrix,
            start_zone_id=start_zone_id,
            end_zone_id=end_zone_id,
        )
    if method == "ortools":
        ortools_params = _parse_ortools_solver_params(solver_params)
        return solve_orienteering_ortools(
            zone_scores=zone_scores,
            max_distance_budget=max_distance_budget,
            distance_matrix=distance_matrix,
            start_zone_id=start_zone_id,
            end_zone_id=end_zone_id,
            log_to_console=log_to_console,
            **ortools_params,
        )
    if method != "exact":
        raise ValueError(f"Unsupported orienteering solve method: {method!r}")

    orienteering_model = build_orienteering_model(
        zone_scores=zone_scores,
        max_distance_budget=max_distance_budget,
        distance_matrix=distance_matrix,
        start_zone_id=start_zone_id,
        end_zone_id=end_zone_id,
        name=name,
        log_to_console=log_to_console,
        solver_params=solver_params,
        optimization_solver=optimization_solver,
    )
    _optimize_orienteering_model(orienteering_model)
    return _extract_orienteering_result(orienteering_model, method="exact")


def solve_orienteering_heuristic(
    zone_scores: Mapping[int, float] | pd.Series,
    max_distance_budget: float = DEFAULT_MAX_DISTANCE_BUDGET,
    *,
    distance_matrix: pd.DataFrame | None = None,
    start_zone_id: int = DEFAULT_START_ZONE_ID,
    end_zone_id: int | None = None,
) -> OrienteeringResult:
    normalized_scores = _normalize_zone_scores(zone_scores)
    if max_distance_budget < 0:
        raise ValueError("max_distance_budget must be non-negative.")

    resolved_end_zone_id = start_zone_id if end_zone_id is None else int(end_zone_id)
    resolved_distance_matrix = (
        load_zone_distance_matrix() if distance_matrix is None else _normalize_distance_matrix(distance_matrix)
    )
    candidate_zone_ids = tuple(sorted(set(normalized_scores) | {int(start_zone_id), resolved_end_zone_id}))
    _validate_distance_matrix_coverage(
        distance_matrix=resolved_distance_matrix,
        required_zone_ids=candidate_zone_ids,
    )

    route_zone_ids = [int(start_zone_id), resolved_end_zone_id]
    route_distance = float(resolved_distance_matrix.loc[int(start_zone_id), resolved_end_zone_id])
    remaining_zone_ids = {
        zone_id
        for zone_id in candidate_zone_ids
        if zone_id not in {int(start_zone_id), resolved_end_zone_id}
        and normalized_scores.get(zone_id, 0.0) > _NUMERICAL_TOLERANCE
    }

    while remaining_zone_ids:
        best_candidate: tuple[float, float, float, int, int] | None = None
        for zone_id in sorted(remaining_zone_ids):
            score = normalized_scores[zone_id]
            for insert_after_idx in range(len(route_zone_ids) - 1):
                origin_zone_id = route_zone_ids[insert_after_idx]
                destination_zone_id = route_zone_ids[insert_after_idx + 1]
                marginal_distance = (
                    float(resolved_distance_matrix.loc[origin_zone_id, zone_id])
                    + float(resolved_distance_matrix.loc[zone_id, destination_zone_id])
                    - float(resolved_distance_matrix.loc[origin_zone_id, destination_zone_id])
                )
                if route_distance + marginal_distance > max_distance_budget + _NUMERICAL_TOLERANCE:
                    continue

                if marginal_distance <= _NUMERICAL_TOLERANCE:
                    gain_ratio = float("inf")
                else:
                    gain_ratio = score / marginal_distance

                candidate = (gain_ratio, score, -marginal_distance, zone_id, insert_after_idx)
                if best_candidate is None or candidate > best_candidate:
                    best_candidate = candidate

        if best_candidate is None:
            break

        _, _, neg_marginal_distance, chosen_zone_id, insert_after_idx = best_candidate
        marginal_distance = -neg_marginal_distance
        route_zone_ids.insert(insert_after_idx + 1, chosen_zone_id)
        route_distance += marginal_distance
        remaining_zone_ids.remove(chosen_zone_id)

    if route_distance > max_distance_budget + _NUMERICAL_TOLERANCE:
        raise RuntimeError("Orienteering heuristic did not produce a feasible solution.")

    return _build_orienteering_result(
        method="heuristic",
        zone_scores=normalized_scores,
        distance_matrix=resolved_distance_matrix.loc[candidate_zone_ids, candidate_zone_ids].copy(),
        start_zone_id=int(start_zone_id),
        end_zone_id=resolved_end_zone_id,
        max_distance_budget=float(max_distance_budget),
        route_zone_ids=tuple(route_zone_ids),
        solver_status=None,
        optimal=False,
        mip_gap=None,
    )


def solve_orienteering_ortools(
    zone_scores: Mapping[int, float] | pd.Series,
    max_distance_budget: float = DEFAULT_MAX_DISTANCE_BUDGET,
    *,
    distance_matrix: pd.DataFrame | None = None,
    start_zone_id: int = DEFAULT_START_ZONE_ID,
    end_zone_id: int | None = None,
    time_limit_s: float = DEFAULT_ORTOOLS_TIME_LIMIT_S,
    distance_scale: int = _DEFAULT_ORTOOLS_DISTANCE_SCALE,
    score_scale: int = _DEFAULT_ORTOOLS_SCORE_SCALE,
    first_solution_strategy: str | int = "PATH_CHEAPEST_ARC",
    local_search_metaheuristic: str | int = DEFAULT_ORTOOLS_LOCAL_SEARCH_METAHEURISTIC,
    log_to_console: bool = False,
) -> OrienteeringResult:
    normalized_scores = _normalize_zone_scores(zone_scores)
    if max_distance_budget < 0:
        raise ValueError("max_distance_budget must be non-negative.")
    if time_limit_s <= 0:
        raise ValueError("time_limit_s must be strictly positive.")
    if distance_scale <= 0:
        raise ValueError("distance_scale must be strictly positive.")
    if score_scale <= 0:
        raise ValueError("score_scale must be strictly positive.")

    resolved_start_zone_id = int(start_zone_id)
    resolved_end_zone_id = resolved_start_zone_id if end_zone_id is None else int(end_zone_id)
    resolved_distance_matrix = (
        load_zone_distance_matrix() if distance_matrix is None else _normalize_distance_matrix(distance_matrix)
    )
    positive_score_zone_ids = {
        zone_id for zone_id, score in normalized_scores.items() if score > _NUMERICAL_TOLERANCE
    }
    candidate_zone_ids = tuple(sorted(positive_score_zone_ids | {resolved_start_zone_id, resolved_end_zone_id}))
    _validate_distance_matrix_coverage(
        distance_matrix=resolved_distance_matrix,
        required_zone_ids=candidate_zone_ids,
    )

    node_index_by_zone_id = {zone_id: node_idx for node_idx, zone_id in enumerate(candidate_zone_ids)}
    start_node_idx = node_index_by_zone_id[resolved_start_zone_id]
    end_node_idx = node_index_by_zone_id[resolved_end_zone_id]
    customer_zone_ids = tuple(
        zone_id
        for zone_id in candidate_zone_ids
        if zone_id not in {resolved_start_zone_id, resolved_end_zone_id}
    )

    integer_distances = _build_integer_distance_matrix(
        distance_matrix=resolved_distance_matrix,
        zone_ids=candidate_zone_ids,
        scale=distance_scale,
    )
    max_distance_int = _scale_distance_budget(max_distance_budget, distance_scale)

    if resolved_start_zone_id == resolved_end_zone_id:
        manager = pywrapcp.RoutingIndexManager(len(candidate_zone_ids), 1, start_node_idx)
    else:
        manager = pywrapcp.RoutingIndexManager(len(candidate_zone_ids), 1, [start_node_idx], [end_node_idx])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        origin_idx = manager.IndexToNode(from_index)
        destination_idx = manager.IndexToNode(to_index)
        return integer_distances[origin_idx][destination_idx]

    transit_cb = routing.RegisterTransitCallback(distance_callback)
    zero_cost_cb = routing.RegisterTransitCallback(lambda _from_index, _to_index: 0)
    routing.SetArcCostEvaluatorOfAllVehicles(zero_cost_cb)
    routing.AddDimension(
        transit_cb,
        0,
        max_distance_int,
        True,
        "Distance",
    )

    for zone_id in customer_zone_ids:
        prize = _scale_score_penalty(normalized_scores.get(zone_id, 0.0), score_scale)
        routing.AddDisjunction([manager.NodeToIndex(node_index_by_zone_id[zone_id])], prize)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = _routing_enum_value(
        routing_enums_pb2.FirstSolutionStrategy,
        first_solution_strategy,
        param_name="first_solution_strategy",
    )
    search_params.local_search_metaheuristic = _routing_enum_value(
        routing_enums_pb2.LocalSearchMetaheuristic,
        local_search_metaheuristic,
        param_name="local_search_metaheuristic",
    )
    seconds = int(time_limit_s)
    nanos = int(round((float(time_limit_s) - seconds) * 1_000_000_000))
    if nanos == 1_000_000_000:
        seconds += 1
        nanos = 0
    search_params.time_limit.seconds = seconds
    search_params.time_limit.nanos = nanos
    search_params.log_search = log_to_console

    solution = routing.SolveWithParameters(search_params)
    if solution is None:
        status = routing.status() if hasattr(routing, "status") else None
        raise RuntimeError(
            "OR-Tools orienteering model did not produce a feasible solution. "
            f"Routing status code: {status}.",
        )

    route_zone_ids = _extract_ortools_route(
        manager=manager,
        routing=routing,
        solution=solution,
        vehicle_id=0,
        zone_ids=candidate_zone_ids,
    )
    result = _build_orienteering_result(
        method="ortools",
        zone_scores=normalized_scores,
        distance_matrix=resolved_distance_matrix.loc[candidate_zone_ids, candidate_zone_ids].copy(),
        start_zone_id=resolved_start_zone_id,
        end_zone_id=resolved_end_zone_id,
        max_distance_budget=float(max_distance_budget),
        route_zone_ids=route_zone_ids,
        solver_status=int(routing.status()) if hasattr(routing, "status") else None,
        optimal=False,
        mip_gap=None,
    )
    if result.total_distance > max_distance_budget + _NUMERICAL_TOLERANCE:
        raise RuntimeError("OR-Tools orienteering route exceeded the distance budget after float validation.")
    return result


def _normalize_zone_scores(
    zone_scores: Mapping[int, float] | pd.Series,
) -> dict[int, float]:
    if isinstance(zone_scores, pd.Series):
        items = zone_scores.items()
    else:
        items = zone_scores.items()

    normalized: dict[int, float] = {}
    for zone_id, score in items:
        normalized_zone_id = int(cast(int | str, zone_id))
        normalized_score = float(score)
        if normalized_score < 0:
            raise ValueError("zone_scores must be non-negative.")
        normalized[normalized_zone_id] = normalized_score
    return normalized


def _normalize_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    normalized = distance_matrix.copy()
    normalized.index = normalized.index.astype(int)
    normalized.columns = [int(column) for column in normalized.columns]
    normalized = normalized.sort_index().sort_index(axis=1)
    if normalized.isna().any().any():
        raise ValueError("distance_matrix contains missing values.")
    if (normalized < -_NUMERICAL_TOLERANCE).any().any():
        raise ValueError("distance_matrix must be non-negative.")
    return normalized.astype(float)


def _validate_distance_matrix_coverage(
    *,
    distance_matrix: pd.DataFrame,
    required_zone_ids: tuple[int, ...],
) -> None:
    missing_index_ids = sorted(set(required_zone_ids) - set(distance_matrix.index))
    missing_column_ids = sorted(set(required_zone_ids) - set(distance_matrix.columns))
    if missing_index_ids or missing_column_ids:
        raise ValueError(
            "distance_matrix is missing required zone IDs. "
            f"Missing rows: {missing_index_ids}. Missing columns: {missing_column_ids}.",
        )


def _build_arc_distances(
    *,
    distance_matrix: pd.DataFrame,
    customer_zone_ids: tuple[int, ...],
    start_zone_id: int,
    end_zone_id: int,
) -> dict[tuple[NodeKey, NodeKey], float]:
    arc_distances: dict[tuple[NodeKey, NodeKey], float] = {
        (_SOURCE_NODE, _SINK_NODE): float(distance_matrix.loc[start_zone_id, end_zone_id]),
    }

    for zone_id in customer_zone_ids:
        arc_distances[_SOURCE_NODE, zone_id] = float(distance_matrix.loc[start_zone_id, zone_id])
        arc_distances[zone_id, _SINK_NODE] = float(distance_matrix.loc[zone_id, end_zone_id])

    for origin_zone_id in customer_zone_ids:
        for destination_zone_id in customer_zone_ids:
            if origin_zone_id == destination_zone_id:
                continue
            arc_distances[origin_zone_id, destination_zone_id] = float(
                distance_matrix.loc[origin_zone_id, destination_zone_id]
            )

    return arc_distances


def _build_integer_distance_matrix(
    *,
    distance_matrix: pd.DataFrame,
    zone_ids: tuple[int, ...],
    scale: int,
) -> list[list[int]]:
    integer_distances: list[list[int]] = []
    for origin_zone_id in zone_ids:
        row: list[int] = []
        for destination_zone_id in zone_ids:
            distance = float(distance_matrix.loc[origin_zone_id, destination_zone_id])
            row.append(_scale_distance(distance, scale))
        integer_distances.append(row)
    return integer_distances


def _scale_distance(distance: float, scale: int) -> int:
    if distance < -_NUMERICAL_TOLERANCE:
        raise ValueError("distance_matrix must be non-negative.")
    return int(max(0, math.ceil(distance * scale - _NUMERICAL_TOLERANCE)))


def _scale_distance_budget(max_distance_budget: float, scale: int) -> int:
    return int(math.floor(max_distance_budget * scale + _NUMERICAL_TOLERANCE))


def _scale_score_penalty(score: float, scale: int) -> int:
    penalty = int(round(score * scale))
    if score > _NUMERICAL_TOLERANCE and penalty <= 0:
        return 1
    return max(0, penalty)


def _parse_ortools_solver_params(
    solver_params: Mapping[str, float | int | str] | None,
) -> dict[str, Any]:
    if solver_params is None:
        return {}

    allowed_params = {
        "time_limit_s",
        "distance_scale",
        "score_scale",
        "first_solution_strategy",
        "local_search_metaheuristic",
    }
    unknown_params = sorted(set(solver_params) - allowed_params)
    if unknown_params:
        raise ValueError(f"Unsupported OR-Tools solver_params: {unknown_params}.")

    parsed_params: dict[str, Any] = {}
    if "time_limit_s" in solver_params:
        parsed_params["time_limit_s"] = float(solver_params["time_limit_s"])
    if "distance_scale" in solver_params:
        parsed_params["distance_scale"] = int(solver_params["distance_scale"])
    if "score_scale" in solver_params:
        parsed_params["score_scale"] = int(solver_params["score_scale"])
    if "first_solution_strategy" in solver_params:
        parsed_params["first_solution_strategy"] = solver_params["first_solution_strategy"]
    if "local_search_metaheuristic" in solver_params:
        parsed_params["local_search_metaheuristic"] = solver_params["local_search_metaheuristic"]
    return parsed_params


def _routing_enum_value(
    enum_message: Any,
    value: str | int,
    *,
    param_name: str,
) -> int:
    if isinstance(value, str):
        normalized_value = value.upper()
        values_by_name = enum_message.DESCRIPTOR.enum_types_by_name["Value"].values_by_name
        if normalized_value not in values_by_name:
            supported_values = ", ".join(sorted(values_by_name))
            raise ValueError(f"Unsupported {param_name}: {value!r}. Supported values: {supported_values}.")
        return int(values_by_name[normalized_value].number)
    return int(value)


def _extract_ortools_route(
    *,
    manager: pywrapcp.RoutingIndexManager,
    routing: pywrapcp.RoutingModel,
    solution: pywrapcp.Assignment,
    vehicle_id: int,
    zone_ids: tuple[int, ...],
) -> tuple[int, ...]:
    route_zone_ids: list[int] = []
    index = routing.Start(vehicle_id)
    seen_indices: set[int] = set()

    while not routing.IsEnd(index):
        if index in seen_indices:
            raise RuntimeError("OR-Tools route contains a repeated routing index.")
        seen_indices.add(index)
        route_zone_ids.append(zone_ids[manager.IndexToNode(index)])
        index = solution.Value(routing.NextVar(index))

    route_zone_ids.append(zone_ids[manager.IndexToNode(index)])
    return tuple(route_zone_ids)


def _build_adjacency_maps(
    arc_distances: Mapping[tuple[NodeKey, NodeKey], float],
) -> tuple[dict[NodeKey, tuple[NodeKey, ...]], dict[NodeKey, tuple[NodeKey, ...]]]:
    outgoing: dict[NodeKey, list[NodeKey]] = {}
    incoming: dict[NodeKey, list[NodeKey]] = {}

    for origin_node, destination_node in arc_distances:
        outgoing.setdefault(origin_node, []).append(destination_node)
        incoming.setdefault(destination_node, []).append(origin_node)
        outgoing.setdefault(destination_node, [])
        incoming.setdefault(origin_node, [])

    outgoing_tuples = {
        node: tuple(sorted(neighbors, key=lambda neighbor: (isinstance(neighbor, str), neighbor)))
        for node, neighbors in outgoing.items()
    }
    incoming_tuples = {
        node: tuple(sorted(neighbors, key=lambda neighbor: (isinstance(neighbor, str), neighbor)))
        for node, neighbors in incoming.items()
    }
    return outgoing_tuples, incoming_tuples


def _build_orienteering_pyomo_model(
    *,
    name: str,
    normalized_scores: Mapping[int, float],
    max_distance_budget: float,
    customer_zone_ids: tuple[int, ...],
    arc_distances: Mapping[tuple[NodeKey, NodeKey], float],
    outgoing_arcs: Mapping[NodeKey, tuple[NodeKey, ...]],
    incoming_arcs: Mapping[NodeKey, tuple[NodeKey, ...]],
) -> pyo.ConcreteModel:
    model = pyo.ConcreteModel(name=name)
    model.ARCS = pyo.Set(dimen=2, initialize=list(arc_distances))
    model.CUSTOMERS = pyo.Set(initialize=customer_zone_ids)
    model.CUSTOMER_ARCS = pyo.Set(
        dimen=2,
        initialize=[
            (origin, destination)
            for origin, destination in arc_distances
            if isinstance(origin, int) and isinstance(destination, int)
        ],
    )

    model.x = pyo.Var(model.ARCS, domain=pyo.Binary)
    model.y = pyo.Var(model.CUSTOMERS, domain=pyo.Binary)
    if len(customer_zone_ids) > 0:
        model.order = pyo.Var(
            model.CUSTOMERS,
            domain=pyo.NonNegativeReals,
            bounds=(0.0, float(len(customer_zone_ids))),
        )

    model.objective = pyo.Objective(
        expr=sum(
            normalized_scores.get(zone_id, 0.0) * model.y[zone_id]
            for zone_id in customer_zone_ids
        ),
        sense=pyo.maximize,
    )
    model.source_departure = pyo.Constraint(
        expr=sum(model.x[_SOURCE_NODE, successor] for successor in outgoing_arcs[_SOURCE_NODE]) == 1
    )
    model.sink_arrival = pyo.Constraint(
        expr=sum(model.x[predecessor, _SINK_NODE] for predecessor in incoming_arcs[_SINK_NODE]) == 1
    )

    model.in_degree = pyo.Constraint(
        model.CUSTOMERS,
        rule=lambda route_model, zone_id: (
            sum(route_model.x[predecessor, zone_id] for predecessor in incoming_arcs[zone_id])
            == route_model.y[zone_id]
        ),
    )
    model.out_degree = pyo.Constraint(
        model.CUSTOMERS,
        rule=lambda route_model, zone_id: (
            sum(route_model.x[zone_id, successor] for successor in outgoing_arcs[zone_id])
            == route_model.y[zone_id]
        ),
    )
    model.distance_budget = pyo.Constraint(
        expr=sum(distance * model.x[arc] for arc, distance in arc_distances.items())
        <= max_distance_budget
    )

    if len(customer_zone_ids) > 0:
        customer_count = len(customer_zone_ids)
        model.order_selected_lower = pyo.Constraint(
            model.CUSTOMERS,
            rule=lambda route_model, zone_id: route_model.order[zone_id]
            >= route_model.y[zone_id],
        )
        model.order_selected_upper = pyo.Constraint(
            model.CUSTOMERS,
            rule=lambda route_model, zone_id: route_model.order[zone_id]
            <= customer_count * route_model.y[zone_id],
        )
        model.subtour_elimination = pyo.Constraint(
            model.CUSTOMER_ARCS,
            rule=lambda route_model, origin, destination: (
                route_model.order[origin]
                - route_model.order[destination]
                + customer_count * route_model.x[origin, destination]
                <= customer_count - 1
            ),
        )

    return model


def _optimize_orienteering_model(
    orienteering_model: OrienteeringModel,
) -> None:
    solve_result = solve_pyomo_model(
        orienteering_model.model,
        solver_name=orienteering_model.optimization_solver,
        solver_params=orienteering_model.solver_params,
        log_to_console=orienteering_model.log_to_console,
    )
    require_pyomo_solution(
        solve_result,
        problem_label="Orienteering model",
    )
    orienteering_model.solve_status = pyomo_solver_status(solve_result)
    orienteering_model.optimal = solve_result.optimal
    orienteering_model.mip_gap = None if solve_result.optimal else pyomo_mip_gap(solve_result)

def _extract_orienteering_result(
    orienteering_model: OrienteeringModel,
    *,
    method: OrienteeringSolveMethod,
) -> OrienteeringResult:
    active_successor = {
        origin_node: destination_node
        for (origin_node, destination_node), variable in orienteering_model.x.items()
        if float(pyo.value(variable)) > 0.5
    }
    route_zone_ids, route_arcs = _extract_route(
        active_successor=active_successor,
        start_zone_id=orienteering_model.start_zone_id,
        end_zone_id=orienteering_model.end_zone_id,
    )

    return _build_orienteering_result(
        method=method,
        zone_scores=orienteering_model.zone_scores,
        distance_matrix=orienteering_model.distance_matrix,
        start_zone_id=orienteering_model.start_zone_id,
        end_zone_id=orienteering_model.end_zone_id,
        max_distance_budget=orienteering_model.max_distance_budget,
        route_zone_ids=route_zone_ids,
        solver_status=orienteering_model.solve_status,
        optimal=orienteering_model.optimal,
        mip_gap=orienteering_model.mip_gap,
    )


def _build_orienteering_result(
    *,
    method: OrienteeringSolveMethod,
    zone_scores: Mapping[int, float],
    distance_matrix: pd.DataFrame,
    start_zone_id: int,
    end_zone_id: int,
    max_distance_budget: float,
    route_zone_ids: tuple[int, ...],
    solver_status: int | str | None,
    optimal: bool,
    mip_gap: float | None,
) -> OrienteeringResult:
    route_arcs = tuple(zip(route_zone_ids[:-1], route_zone_ids[1:], strict=True))
    visited_zone_ids = route_zone_ids[1:-1]
    visited_scores = tuple((zone_id, zone_scores.get(zone_id, 0.0)) for zone_id in visited_zone_ids)
    collected_score = float(sum(score for _, score in visited_scores))
    total_distance = float(
        sum(distance_matrix.loc[origin_zone_id, destination_zone_id] for origin_zone_id, destination_zone_id in route_arcs)
    )

    return OrienteeringResult(
        method=method,
        objective_value=collected_score,
        collected_score=collected_score,
        total_distance=total_distance,
        max_distance_budget=max_distance_budget,
        start_zone_id=start_zone_id,
        end_zone_id=end_zone_id,
        route_zone_ids=route_zone_ids,
        visited_zone_ids=visited_zone_ids,
        route_arcs=route_arcs,
        visited_scores=visited_scores,
        solver_status=solver_status,
        optimal=optimal,
        mip_gap=mip_gap,
    )


def _extract_route(
    *,
    active_successor: Mapping[NodeKey, NodeKey],
    start_zone_id: int,
    end_zone_id: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    route_zone_ids: list[int] = [start_zone_id]
    route_arcs: list[tuple[int, int]] = []
    current_node: NodeKey = _SOURCE_NODE
    seen_nodes: set[NodeKey] = {_SOURCE_NODE}

    while current_node != _SINK_NODE:
        if current_node not in active_successor:
            raise RuntimeError("Solved route is incomplete: missing successor arc.")
        next_node = active_successor[current_node]
        physical_origin = start_zone_id if current_node == _SOURCE_NODE else int(current_node)
        physical_destination = end_zone_id if next_node == _SINK_NODE else int(next_node)
        route_arcs.append((physical_origin, physical_destination))

        if next_node == _SINK_NODE:
            route_zone_ids.append(end_zone_id)
            break

        if next_node in seen_nodes:
            raise RuntimeError("Solved route contains a repeated node in the main path.")
        seen_nodes.add(next_node)
        route_zone_ids.append(int(next_node))
        current_node = next_node

    return tuple(route_zone_ids), tuple(route_arcs)
