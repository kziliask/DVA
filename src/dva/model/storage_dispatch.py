from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from pyepo.model.grb.grbmodel import optGrbModel

RELAXED_THROUGHPUT_PENALTY_THRESHOLD = 1.0


@dataclass(frozen=True, slots=True)
class StorageDispatchParameters:
    energy_capacity: float
    power_limit: float
    charge_efficiency: float
    discharge_efficiency: float
    throughput_penalty: float = 1.0
    initial_state_of_charge: float = 0.0
    terminal_state_of_charge: float | None = None


@dataclass(slots=True)
class StorageDispatchModel:
    model: gp.Model
    prices: tuple[float, ...]
    parameters: StorageDispatchParameters
    charge: gp.tupledict
    discharge: gp.tupledict
    state_of_charge: gp.tupledict
    mode: gp.tupledict | None
    revenue_expression: gp.LinExpr
    throughput_expression: gp.LinExpr


@dataclass(frozen=True, slots=True)
class StorageDispatchResult:
    objective_value: float
    revenue_value: float
    throughput_penalty_value: float
    charge: tuple[float, ...]
    discharge: tuple[float, ...]
    state_of_charge: tuple[float, ...]
    mode: tuple[int, ...]

    @property
    def throughput_value(self) -> float:
        return sum(
            charge_value + discharge_value
            for charge_value, discharge_value in zip(self.charge, self.discharge, strict=True)
        )


@dataclass(frozen=True, slots=True)
class StorageDispatchEvaluation:
    objective_value: float
    revenue_value: float
    throughput_penalty_value: float


class StorageDispatchSPOModel(optGrbModel):
    def __init__(
        self,
        parameters: StorageDispatchParameters,
        horizon: int,
        *,
        solver_params: Mapping[str, float | int | str] | None = None,
    ) -> None:
        self.parameters = parameters
        self.horizon = horizon
        self.solver_params = dict(solver_params or {})
        super().__init__()

    def _getModel(self) -> tuple[gp.Model, dict[int, gp.Var]]:
        _validate_inputs([0.0] * self.horizon, self.parameters)
        model, charge, discharge, _, _ = _build_storage_dispatch_core_model(
            horizon=self.horizon,
            parameters=self.parameters,
            name="storage_dispatch_spo",
            log_to_console=False,
            solver_params=self.solver_params,
        )
        x = {
            hour_idx - 1: charge[hour_idx]
            for hour_idx in range(1, self.horizon + 1)
        }
        x.update(
            {
                self.horizon + hour_idx - 1: discharge[hour_idx]
                for hour_idx in range(1, self.horizon + 1)
            }
        )
        model.setObjective(0.0, GRB.MAXIMIZE)
        return model, x


def prices_to_spo_costs(
    prices: Sequence[float] | np.ndarray,
    throughput_penalty: float,
) -> np.ndarray:
    prices_array = np.asarray(prices, dtype=np.float32)
    if prices_array.ndim == 1:
        return np.concatenate(
            (
                -(prices_array + throughput_penalty),
                prices_array - throughput_penalty,
            )
        )
    if prices_array.ndim == 2:
        return np.concatenate(
            (
                -(prices_array + throughput_penalty),
                prices_array - throughput_penalty,
            ),
            axis=1,
        )
    raise ValueError("prices must be a 1D or 2D array.")


def build_spo_training_targets(
    prices: Sequence[float] | np.ndarray,
    parameters: StorageDispatchParameters,
    *,
    solver_params: Mapping[str, float | int | str] | None = None,
    verbose: bool = False,
    log_every: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    price_matrix = np.asarray(prices, dtype=np.float32)
    if price_matrix.ndim == 1:
        price_matrix = price_matrix[np.newaxis, :]
    elif price_matrix.ndim != 2:
        raise ValueError("prices must be a 1D or 2D array.")

    if log_every is not None and log_every <= 0:
        raise ValueError("log_every must be strictly positive when provided.")

    cost_matrix = prices_to_spo_costs(
        price_matrix,
        throughput_penalty=parameters.throughput_penalty,
    )
    optmodel = StorageDispatchSPOModel(
        parameters=parameters,
        horizon=price_matrix.shape[1],
        solver_params=solver_params,
    )
    solutions = np.zeros_like(cost_matrix, dtype=np.float32)
    objectives = np.zeros((len(cost_matrix), 1), dtype=np.float32)
    resolved_log_every = log_every if log_every is not None else max(1, len(cost_matrix) // 10)
    started_at = time.perf_counter()
    if verbose:
        print(
            "[spo_targets] solving "
            f"{len(cost_matrix)} dispatch problems for ground-truth SPO labels",
            flush=True,
        )
    for row_idx, cost_vector in enumerate(cost_matrix):
        optmodel.setObj(cost_vector)
        solution, objective = optmodel.solve()
        solutions[row_idx] = np.asarray(solution, dtype=np.float32)
        objectives[row_idx, 0] = float(objective)
        if verbose:
            solved = row_idx + 1
            if (
                solved == 1
                or solved == len(cost_matrix)
                or solved % resolved_log_every == 0
            ):
                print(
                    "[spo_targets] solved "
                    f"{solved}/{len(cost_matrix)} "
                    f"(elapsed={time.perf_counter() - started_at:.2f}s)",
                    flush=True,
                )
    return cost_matrix.astype(np.float32, copy=False), solutions, objectives


def build_storage_dispatch_model(
    predicted_prices: Sequence[float],
    parameters: StorageDispatchParameters,
    *,
    name: str = "storage_dispatch",
    log_to_console: bool = False,
    solver_params: Mapping[str, float | int | str] | None = None,
) -> StorageDispatchModel:
    prices = _validate_inputs(predicted_prices=predicted_prices, parameters=parameters)
    model, charge, discharge, state_of_charge, mode = _build_storage_dispatch_core_model(
        horizon=len(prices),
        parameters=parameters,
        name=name,
        log_to_console=log_to_console,
        solver_params=solver_params,
    )

    hours = range(1, len(prices) + 1)
    revenue = gp.quicksum(prices[hour - 1] * (discharge[hour] - charge[hour]) for hour in hours)
    throughput = gp.quicksum(charge[hour] + discharge[hour] for hour in hours)
    model.setObjective(revenue - parameters.throughput_penalty * throughput, GRB.MAXIMIZE)

    return StorageDispatchModel(
        model=model,
        prices=prices,
        parameters=parameters,
        charge=charge,
        discharge=discharge,
        state_of_charge=state_of_charge,
        mode=mode,
        revenue_expression=revenue,
        throughput_expression=throughput,
    )


def solve_storage_dispatch(
    predicted_prices: Sequence[float],
    parameters: StorageDispatchParameters,
    *,
    name: str = "storage_dispatch",
    log_to_console: bool = False,
    solver_params: Mapping[str, float | int | str] | None = None,
) -> StorageDispatchResult:
    storage_model = build_storage_dispatch_model(
        predicted_prices=predicted_prices,
        parameters=parameters,
        name=name,
        log_to_console=log_to_console,
        solver_params=solver_params,
    )
    _optimize_storage_model(storage_model.model)
    return _extract_storage_dispatch_result(storage_model)


def solve_storage_dispatch_lexicographic(
    predicted_prices: Sequence[float],
    parameters: StorageDispatchParameters,
    *,
    name: str = "storage_dispatch",
    log_to_console: bool = False,
    solver_params: Mapping[str, float | int | str] | None = None,
    objective_tolerance: float = 1e-6,
) -> StorageDispatchResult:
    effective_solver_params = dict(solver_params or {})
    effective_solver_params.setdefault("FeasibilityTol", 1e-9)
    storage_model = build_storage_dispatch_model(
        predicted_prices=predicted_prices,
        parameters=parameters,
        name=name,
        log_to_console=log_to_console,
        solver_params=effective_solver_params,
    )
    _optimize_storage_model(storage_model.model)

    primary_objective_value = float(storage_model.model.ObjVal)
    primary_objective_floor = storage_model.model.addConstr(
        storage_model.revenue_expression
        - parameters.throughput_penalty * storage_model.throughput_expression
        >= primary_objective_value - objective_tolerance,
        name="lexicographic_primary_objective_floor",
    )
    storage_model.model.setObjective(storage_model.throughput_expression, GRB.MINIMIZE)
    _optimize_storage_model(storage_model.model)

    # The primary-objective floor is enforced directly in the second-stage model.
    # Trust the solved constrained model here rather than re-checking with a
    # hand-computed floating-point comparison, which can produce false positives.
    return _extract_storage_dispatch_result(storage_model)


def evaluate_storage_dispatch_result(
    realized_prices: Sequence[float],
    dispatch_result: StorageDispatchResult,
    parameters: StorageDispatchParameters,
) -> StorageDispatchEvaluation:
    prices = _validate_inputs(predicted_prices=realized_prices, parameters=parameters)
    if len(prices) != len(dispatch_result.charge):
        raise ValueError(
            "realized_prices must have the same horizon length as the dispatch_result."
        )

    revenue_value = sum(
        price * (discharge_value - charge_value)
        for price, charge_value, discharge_value in zip(
            prices,
            dispatch_result.charge,
            dispatch_result.discharge,
            strict=True,
        )
    )
    throughput_penalty_value = parameters.throughput_penalty * dispatch_result.throughput_value
    return StorageDispatchEvaluation(
        objective_value=revenue_value - throughput_penalty_value,
        revenue_value=revenue_value,
        throughput_penalty_value=throughput_penalty_value,
    )


def _validate_inputs(
    predicted_prices: Sequence[float],
    parameters: StorageDispatchParameters,
) -> tuple[float, ...]:
    prices = tuple(float(price) for price in predicted_prices)
    if not prices:
        raise ValueError("predicted_prices must contain at least one hourly price.")

    if parameters.energy_capacity <= 0:
        raise ValueError("energy_capacity must be strictly positive.")
    if parameters.power_limit < 0:
        raise ValueError("power_limit must be non-negative.")
    if not 0 < parameters.charge_efficiency <= 1:
        raise ValueError("charge_efficiency must be in the interval (0, 1].")
    if not 0 < parameters.discharge_efficiency <= 1:
        raise ValueError("discharge_efficiency must be in the interval (0, 1].")
    if parameters.throughput_penalty < 0:
        raise ValueError("throughput_penalty must be non-negative.")
    if not 0 <= parameters.initial_state_of_charge <= parameters.energy_capacity:
        raise ValueError("initial_state_of_charge must lie between 0 and energy_capacity.")
    if (
        parameters.terminal_state_of_charge is not None
        and not 0 <= parameters.terminal_state_of_charge <= parameters.energy_capacity
    ):
        raise ValueError("terminal_state_of_charge must lie between 0 and energy_capacity.")

    return prices


def _configure_storage_model(
    model: gp.Model,
    *,
    log_to_console: bool,
    solver_params: Mapping[str, float | int | str] | None,
) -> None:
    model.Params.OutputFlag = 1 if log_to_console else 0
    if solver_params is None:
        return
    for param_name, value in solver_params.items():
        model.setParam(param_name, value)


def _build_storage_dispatch_core_model(
    *,
    horizon: int,
    parameters: StorageDispatchParameters,
    name: str,
    log_to_console: bool,
    solver_params: Mapping[str, float | int | str] | None,
) -> tuple[gp.Model, gp.tupledict, gp.tupledict, gp.tupledict, gp.tupledict | None]:
    hours = range(1, horizon + 1)
    soc_points = range(1, horizon + 2)

    model = gp.Model(name)
    _configure_storage_model(
        model,
        log_to_console=log_to_console,
        solver_params=solver_params,
    )

    charge = model.addVars(hours, lb=0.0, ub=parameters.power_limit, name="charge")
    discharge = model.addVars(hours, lb=0.0, ub=parameters.power_limit, name="discharge")
    state_of_charge = model.addVars(
        soc_points,
        lb=0.0,
        ub=parameters.energy_capacity,
        name="state_of_charge",
    )
    mode = None
    if not _should_use_relaxed_dispatch_formulation(parameters):
        mode = model.addVars(hours, vtype=GRB.BINARY, name="mode")

    model.addConstr(
        state_of_charge[1] == parameters.initial_state_of_charge,
        name="initial_state_of_charge",
    )

    if parameters.terminal_state_of_charge is not None:
        model.addConstr(
            state_of_charge[horizon + 1] == parameters.terminal_state_of_charge,
            name="terminal_state_of_charge",
        )

    for hour in hours:
        model.addConstr(
            state_of_charge[hour + 1]
            == state_of_charge[hour]
            + parameters.charge_efficiency * charge[hour]
            - discharge[hour] / parameters.discharge_efficiency,
            name=f"state_transition[{hour}]",
        )
        if mode is None:
            model.addConstr(
                charge[hour] + discharge[hour] <= parameters.power_limit,
                name=f"shared_power_limit[{hour}]",
            )
        else:
            model.addConstr(
                charge[hour] <= parameters.power_limit * mode[hour],
                name=f"charge_limit[{hour}]",
            )
            model.addConstr(
                discharge[hour] <= parameters.power_limit * (1 - mode[hour]),
                name=f"discharge_limit[{hour}]",
            )

    return model, charge, discharge, state_of_charge, mode


def _should_use_relaxed_dispatch_formulation(
    parameters: StorageDispatchParameters,
) -> bool:
    return parameters.throughput_penalty >= RELAXED_THROUGHPUT_PENALTY_THRESHOLD


def _optimize_storage_model(model: gp.Model) -> None:
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            "Storage dispatch model did not solve to optimality. "
            f"Gurobi status code: {model.Status}."
        )


def _extract_storage_dispatch_result(
    storage_model: StorageDispatchModel,
) -> StorageDispatchResult:
    hours = range(1, len(storage_model.prices) + 1)
    soc_points = range(1, len(storage_model.prices) + 2)

    charge = tuple(storage_model.charge[hour].X for hour in hours)
    discharge = tuple(storage_model.discharge[hour].X for hour in hours)
    state_of_charge = tuple(storage_model.state_of_charge[hour].X for hour in soc_points)
    if storage_model.mode is None:
        mode = _infer_dispatch_mode_from_flows(charge, discharge)
    else:
        mode = tuple(int(round(storage_model.mode[hour].X)) for hour in hours)
    revenue_value = float(storage_model.revenue_expression.getValue())
    throughput_penalty_value = float(
        storage_model.parameters.throughput_penalty
        * storage_model.throughput_expression.getValue()
    )

    return StorageDispatchResult(
        objective_value=revenue_value - throughput_penalty_value,
        revenue_value=revenue_value,
        throughput_penalty_value=throughput_penalty_value,
        charge=charge,
        discharge=discharge,
        state_of_charge=state_of_charge,
        mode=mode,
    )


def _infer_dispatch_mode_from_flows(
    charge: Sequence[float],
    discharge: Sequence[float],
    *,
    atol: float = 1e-9,
) -> tuple[int, ...]:
    """Return a deterministic binary mode for downstream reporting.

    The relaxed formulation does not include a discrete mode variable, but the
    rest of the pipeline expects one. Treat charging-dominant or tied hours as
    charge mode and discharge-dominant hours as discharge mode.
    """

    inferred_mode: list[int] = []
    for charge_value, discharge_value in zip(charge, discharge, strict=True):
        if abs(charge_value - discharge_value) <= atol:
            inferred_mode.append(1 if charge_value > atol else 0)
            continue
        inferred_mode.append(1 if charge_value > discharge_value else 0)
    return tuple(inferred_mode)
