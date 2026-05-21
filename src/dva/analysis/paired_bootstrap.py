from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd


MetricDirection = Literal["up", "down"]


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    n_bootstrap: int = 10_000
    confidence_level: float = 0.95
    seed: int = 0


def bootstrap_metric_table(
    metrics: pd.DataFrame,
    *,
    unit_column: str = "unit_id",
    dataset_column: str = "dataset",
    metric_column: str = "metric",
    method_column: str = "method",
    value_column: str = "value",
    reference_method: str | None = None,
    direction_by_metric: Mapping[str, MetricDirection] | None = None,
    config: BootstrapConfig | None = None,
) -> pd.DataFrame:
    """Compute percentile bootstrap CIs for means and paired reference deltas.

    The input must be in long form, with one row per evaluation unit, metric,
    and attribution method. Pairwise deltas are resampled over the common
    finite units for the compared method and the reference method.
    """

    config = config or BootstrapConfig()
    _validate_bootstrap_config(config)
    _validate_required_columns(
        metrics,
        (unit_column, dataset_column, metric_column, method_column, value_column),
    )

    frame = metrics.loc[
        :,
        [unit_column, dataset_column, metric_column, method_column, value_column],
    ].copy()
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna(subset=[unit_column, dataset_column, metric_column, method_column])

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(config.seed)
    directions = dict(direction_by_metric or {})
    group_columns = [dataset_column, metric_column]
    for (dataset, metric), group in frame.groupby(group_columns, sort=True):
        metric_name = str(metric)
        direction = directions.get(metric_name) or infer_metric_direction(metric_name)
        wide = (
            group.pivot_table(
                index=unit_column,
                columns=method_column,
                values=value_column,
                aggfunc="mean",
            )
            .sort_index()
            .sort_index(axis=1)
        )
        if wide.empty:
            continue
        methods = [str(method) for method in wide.columns]
        reference = reference_method if reference_method in methods else None
        for method in methods:
            values = _finite_array(wide[method].to_numpy(dtype=float))
            mean_samples = _bootstrap_mean(values, config=config, rng=rng)
            mean_low, mean_high = _percentile_ci(
                mean_samples,
                confidence_level=config.confidence_level,
            )
            row: dict[str, object] = {
                "dataset": str(dataset),
                "metric": metric_name,
                "metric_direction": direction,
                "method": method,
                "n_units": int(len(values)),
                "mean": _safe_mean(values),
                "mean_ci_low": mean_low,
                "mean_ci_high": mean_high,
                "mean_bootstrap_std": _safe_std(mean_samples),
                "reference_method": reference,
                "n_pair_units": None,
                "reference_minus_method": None,
                "reference_minus_method_ci_low": None,
                "reference_minus_method_ci_high": None,
                "reference_better_delta": None,
                "reference_better_delta_ci_low": None,
                "reference_better_delta_ci_high": None,
                "confidence_level": float(config.confidence_level),
                "n_bootstrap": int(config.n_bootstrap),
                "seed": int(config.seed),
            }
            if reference is not None:
                pair = wide.loc[:, [reference, method]].dropna()
                if not pair.empty:
                    reference_values = pair[reference].to_numpy(dtype=float)
                    method_values = pair[method].to_numpy(dtype=float)
                    raw_deltas = reference_values - method_values
                    raw_samples = _bootstrap_mean(raw_deltas, config=config, rng=rng)
                    raw_low, raw_high = _percentile_ci(
                        raw_samples,
                        confidence_level=config.confidence_level,
                    )
                    better_deltas = _reference_better_deltas(
                        reference_values,
                        method_values,
                        direction,
                    )
                    better_samples = _bootstrap_mean(
                        better_deltas,
                        config=config,
                        rng=rng,
                    )
                    better_low, better_high = _percentile_ci(
                        better_samples,
                        confidence_level=config.confidence_level,
                    )
                    row.update(
                        {
                            "n_pair_units": int(len(pair)),
                            "reference_minus_method": _safe_mean(raw_deltas),
                            "reference_minus_method_ci_low": raw_low,
                            "reference_minus_method_ci_high": raw_high,
                            "reference_better_delta": _safe_mean(better_deltas),
                            "reference_better_delta_ci_low": better_low,
                            "reference_better_delta_ci_high": better_high,
                        }
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def wide_metric_frame(
    path: Path | str,
    *,
    dataset: str,
    metric: str,
    method_columns: Mapping[str, str],
    unit_column: str,
) -> pd.DataFrame:
    """Load a wide metric CSV and return the long form used by the bootstrap."""

    source = pd.read_csv(path)
    _validate_required_columns(source, (unit_column, *method_columns.values()))
    rows = []
    for method, column in method_columns.items():
        frame = source.loc[:, [unit_column, column]].copy()
        frame = frame.rename(columns={unit_column: "unit_id", column: "value"})
        frame["dataset"] = dataset
        frame["metric"] = metric
        frame["method"] = method
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def infer_metric_direction(metric: str) -> MetricDirection | None:
    normalized = metric.lower()
    if "\\downarrow" in normalized or "↓" in normalized:
        return "down"
    if "\\uparrow" in normalized or "↑" in normalized:
        return "up"
    if "infidelity" in normalized or "regret" in normalized or "loss" in normalized:
        return "down"
    if "auc" in normalized or "accuracy" in normalized or "value" in normalized:
        return "up"
    return None


def parse_method_column_specs(specs: Sequence[str]) -> dict[str, str]:
    method_columns: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                "Each --method-column must have the form METHOD=COLUMN; "
                f"got {spec!r}."
            )
        method, column = spec.split("=", 1)
        method = method.strip()
        column = column.strip()
        if not method or not column:
            raise ValueError(
                "Each --method-column must have non-empty METHOD and COLUMN names."
            )
        method_columns[method] = column
    return method_columns


def parse_metric_direction_specs(specs: Sequence[str]) -> dict[str, MetricDirection]:
    directions: dict[str, MetricDirection] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                "Each --metric-direction must have the form METRIC=up or METRIC=down; "
                f"got {spec!r}."
            )
        metric, direction = spec.split("=", 1)
        metric = metric.strip()
        direction = direction.strip().lower()
        if direction not in {"up", "down"}:
            raise ValueError(
                f"Metric direction must be 'up' or 'down'; got {direction!r}."
            )
        directions[metric] = cast(MetricDirection, direction)
    return directions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired percentile bootstrap confidence intervals for "
            "per-instance attribution evaluation metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--input-format", choices=("long", "wide"), default="long")
    parser.add_argument("--reference-method", default="Post-InfoDVA")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--unit-column", default="unit_id")
    parser.add_argument("--dataset-column", default="dataset")
    parser.add_argument("--metric-column", default="metric")
    parser.add_argument("--method-column-name", default="method")
    parser.add_argument("--value-column", default="value")
    parser.add_argument(
        "--metric-direction",
        action="append",
        default=[],
        help="Long input only. Repeat as METRIC=up or METRIC=down.",
    )

    parser.add_argument("--dataset", default=None, help="Wide input only.")
    parser.add_argument("--metric", default=None, help="Wide input only.")
    parser.add_argument(
        "--direction",
        choices=("up", "down"),
        default=None,
        help="Wide input only. Overrides inferred direction for --metric.",
    )
    parser.add_argument(
        "--method-column",
        action="append",
        default=[],
        help="Wide input only. Repeat as METHOD=COLUMN.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = BootstrapConfig(
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    if args.input_format == "wide":
        if args.dataset is None or args.metric is None or not args.method_column:
            raise SystemExit(
                "Wide input requires --dataset, --metric, and at least one "
                "--method-column METHOD=COLUMN."
            )
        method_columns = parse_method_column_specs(args.method_column)
        metrics = wide_metric_frame(
            args.input,
            dataset=args.dataset,
            metric=args.metric,
            method_columns=method_columns,
            unit_column=args.unit_column,
        )
        direction_by_metric = (
            {args.metric: args.direction} if args.direction is not None else {}
        )
        unit_column = "unit_id"
        dataset_column = "dataset"
        metric_column = "metric"
        method_column = "method"
        value_column = "value"
    else:
        metrics = pd.read_csv(args.input)
        direction_by_metric = parse_metric_direction_specs(args.metric_direction)
        unit_column = args.unit_column
        dataset_column = args.dataset_column
        metric_column = args.metric_column
        method_column = args.method_column_name
        value_column = args.value_column

    result = bootstrap_metric_table(
        metrics,
        unit_column=unit_column,
        dataset_column=dataset_column,
        metric_column=metric_column,
        method_column=method_column,
        value_column=value_column,
        reference_method=args.reference_method,
        direction_by_metric=direction_by_metric,
        config=config,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(result.to_string(index=False))


def _validate_bootstrap_config(config: BootstrapConfig) -> None:
    if config.n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    if not 0.0 < config.confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")


def _validate_required_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError("Missing required columns: " + ", ".join(missing))


def _finite_array(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    return np.asarray(finite, dtype=float)


def _bootstrap_mean(
    values: np.ndarray,
    *,
    config: BootstrapConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    values = _finite_array(values)
    if len(values) == 0:
        return np.full(config.n_bootstrap, np.nan, dtype=float)
    samples = np.empty(config.n_bootstrap, dtype=float)
    chunk_size = 1_000
    for start in range(0, config.n_bootstrap, chunk_size):
        stop = min(start + chunk_size, config.n_bootstrap)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = values[indices].mean(axis=1)
    return samples


def _percentile_ci(
    samples: np.ndarray,
    *,
    confidence_level: float,
) -> tuple[float | None, float | None]:
    finite = _finite_array(samples)
    if len(finite) == 0:
        return None, None
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _safe_mean(values: np.ndarray) -> float | None:
    finite = _finite_array(values)
    if len(finite) == 0:
        return None
    return float(np.mean(finite))


def _safe_std(values: np.ndarray) -> float | None:
    finite = _finite_array(values)
    if len(finite) == 0:
        return None
    return float(np.std(finite, ddof=0))


def _reference_better_deltas(
    reference_values: np.ndarray,
    method_values: np.ndarray,
    direction: MetricDirection | None,
) -> np.ndarray:
    if direction == "down":
        return method_values - reference_values
    return reference_values - method_values


if __name__ == "__main__":
    main()
