from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import pyomo.environ as pyo
from pyomo.opt import SolverResults, SolverStatus, TerminationCondition


DEFAULT_OPTIMIZATION_SOLVER = "highs"

_OPTIMIZATION_SOLVER_ALIASES = {
    "default": DEFAULT_OPTIMIZATION_SOLVER,
    "highs": "highs",
    "highspy": "highs",
    "gurobi": "gurobi",
}


@dataclass(frozen=True, slots=True)
class PyomoSolveResult:
    raw: SolverResults
    solver_name: str

    @property
    def status(self) -> str:
        return str(self.raw.solver.status)

    @property
    def termination_condition(self) -> str:
        return str(self.raw.solver.termination_condition)

    @property
    def optimal(self) -> bool:
        return self.raw.solver.termination_condition == TerminationCondition.optimal

    @property
    def has_solution(self) -> bool:
        return len(self.raw.solution) > 0

    @property
    def solver_runtime_seconds(self) -> float | None:
        return pyomo_solver_runtime_seconds(self)


def normalize_optimization_solver(solver_name: str | None) -> str:
    if solver_name is None:
        return DEFAULT_OPTIMIZATION_SOLVER
    solver_key = str(solver_name).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _OPTIMIZATION_SOLVER_ALIASES.get(solver_key)
    if normalized is None:
        raise ValueError(
            f"Unsupported optimization solver: {solver_name!r}. "
            "Expected one of highs or gurobi."
        )
    return normalized


def solve_pyomo_model(
    model: pyo.ConcreteModel,
    *,
    solver_name: str | None = None,
    solver_params: Mapping[str, float | int | str] | None = None,
    log_to_console: bool = False,
) -> PyomoSolveResult:
    normalized_solver = normalize_optimization_solver(solver_name)
    solver = pyo.SolverFactory(normalized_solver)
    if solver is None or not solver.available(exception_flag=False):
        raise RuntimeError(_missing_solver_message(normalized_solver))

    _configure_solver(
        solver,
        solver_name=normalized_solver,
        solver_params=solver_params,
        log_to_console=log_to_console,
    )
    results = solver.solve(model, tee=log_to_console, load_solutions=False)
    if len(results.solution) > 0:
        model.solutions.load_from(results)
    return PyomoSolveResult(raw=results, solver_name=normalized_solver)


def require_pyomo_solution(
    solve_result: PyomoSolveResult,
    *,
    problem_label: str,
) -> None:
    if solve_result.has_solution:
        return
    raise RuntimeError(
        f"{problem_label} did not produce a feasible solution; "
        f"solver={solve_result.solver_name}, "
        f"status={solve_result.status}, "
        f"termination_condition={solve_result.termination_condition}."
    )


def require_pyomo_optimal(
    solve_result: PyomoSolveResult,
    *,
    problem_label: str,
) -> None:
    require_pyomo_solution(solve_result, problem_label=problem_label)
    if solve_result.optimal:
        return
    raise RuntimeError(
        f"{problem_label} did not solve to optimality; "
        f"solver={solve_result.solver_name}, "
        f"status={solve_result.status}, "
        f"termination_condition={solve_result.termination_condition}."
    )


def pyomo_solver_status(solve_result: PyomoSolveResult) -> str:
    return f"{solve_result.status}:{solve_result.termination_condition}"


def pyomo_mip_gap(solve_result: PyomoSolveResult) -> float | None:
    problem = getattr(solve_result.raw, "problem", None)
    if problem is None:
        return None
    try:
        lower_bound = float(problem.lower_bound)
        upper_bound = float(problem.upper_bound)
    except (TypeError, ValueError, AttributeError):
        return None
    if not all(value == value for value in (lower_bound, upper_bound)):
        return None
    denominator = max(1.0, abs(upper_bound))
    return abs(upper_bound - lower_bound) / denominator


def pyomo_solver_runtime_seconds(solve_result: PyomoSolveResult) -> float | None:
    solver = getattr(solve_result.raw, "solver", None)
    if solver is None:
        return None
    for attr_name in ("time", "wallclock_time", "wallclock"):
        value = _coerce_nonnegative_float(getattr(solver, attr_name, None))
        if value is not None:
            return value
    if hasattr(solver, "get"):
        for key in ("Time", "time", "wallclock_time", "Wallclock time"):
            value = _coerce_nonnegative_float(solver.get(key, None))
            if value is not None:
                return value
    return None


def _coerce_nonnegative_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < 0.0:
        return None
    return numeric


def _configure_solver(
    solver: Any,
    *,
    solver_name: str,
    solver_params: Mapping[str, float | int | str] | None,
    log_to_console: bool,
) -> None:
    if hasattr(solver, "config"):
        config = solver.config
        _maybe_set_config(config, "tee", bool(log_to_console))
        _maybe_set_config(config, "load_solutions", False)
        _maybe_set_config(config, "raise_exception_on_nonoptimal_result", False)

    if solver_params is None:
        return

    options = _translate_solver_params(
        solver_name=solver_name,
        solver_params=solver_params,
    )
    if hasattr(solver, "config"):
        config = solver.config
        if "threads" in options and _maybe_set_config(config, "threads", int(options.pop("threads"))):
            pass
        if "time_limit" in options and _maybe_set_config(config, "time_limit", float(options.pop("time_limit"))):
            pass
        if "rel_gap" in options and _maybe_set_config(config, "rel_gap", float(options.pop("rel_gap"))):
            pass
        if "abs_gap" in options and _maybe_set_config(config, "abs_gap", float(options.pop("abs_gap"))):
            pass
        if "solver_options" in getattr(config, "_data", {}):
            existing = dict(config.solver_options or {})
            existing.update(options)
            config.solver_options = existing
            return

    if hasattr(solver, "options"):
        solver.options.update(options)


def _translate_solver_params(
    *,
    solver_name: str,
    solver_params: Mapping[str, float | int | str],
) -> dict[str, float | int | str]:
    translated: dict[str, float | int | str] = {}
    for param_name, value in solver_params.items():
        key = str(param_name)
        normalized_key = key.strip().lower().replace("-", "_")
        if normalized_key in {"threads", "thread"}:
            if solver_name != "highs":
                translated["Threads"] = int(value)
        elif normalized_key in {"seed", "random_seed"}:
            translated["random_seed" if solver_name == "highs" else "Seed"] = int(value)
        elif normalized_key in {"mipgap", "mip_gap", "rel_gap"}:
            translated["rel_gap" if solver_name == "highs" else "MIPGap"] = float(value)
        elif normalized_key in {"mipgapabs", "mip_gap_abs", "abs_gap"}:
            translated["abs_gap" if solver_name == "highs" else "MIPGapAbs"] = float(value)
        elif normalized_key in {"timelimit", "time_limit", "time_limit_s"}:
            translated["time_limit" if solver_name == "highs" else "TimeLimit"] = float(value)
        elif normalized_key == "feasibilitytol":
            translated[
                "primal_feasibility_tolerance" if solver_name == "highs" else "FeasibilityTol"
            ] = float(value)
        elif normalized_key == "outputflag":
            continue
        else:
            translated[key] = value
    return translated


def _maybe_set_config(config: Any, name: str, value: Any) -> bool:
    try:
        if name not in config:
            return False
        setattr(config, name, value)
        return True
    except (AttributeError, KeyError, TypeError):
        return False


def _missing_solver_message(solver_name: str) -> str:
    if solver_name == "highs":
        return (
            "HiGHS is not available through Pyomo. Install the default solver "
            "dependency with `uv sync` so the highspy package is present."
        )
    if solver_name == "gurobi":
        return (
            "Gurobi is not available through Pyomo. Install the optional Gurobi "
            "dependency with `uv sync --extra gurobi` and ensure a valid Gurobi "
            "license is configured."
        )
    return f"Optimization solver {solver_name!r} is not available through Pyomo."
