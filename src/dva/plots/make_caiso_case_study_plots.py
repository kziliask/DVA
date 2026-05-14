from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import calendar

import cmcrameri  # noqa: F401 - registers cmc.* colormaps with Matplotlib.
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm

from dva.model.storage_dispatch import (
    StorageDispatchParameters,
    solve_storage_dispatch_lexicographic,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET_PATH = Path(
    "data/cleaned/caiso_sp15_daily_lmp_weather_2023-01-26_2026-05-07.csv"
)
DEFAULT_OUTDIR = Path("data/plots/caiso_case_study_39_month")
DATE_COLUMN = "date"
FEATURE_COLUMNS = (
    "min_temp_c",
    "max_temp_c",
    "mean_temp_c",
    "mean_humidity",
    "mean_wind_speed",
    "mean_solar_irradiance",
    "max_solar_irradiance",
    "day_of_week",
)
FEATURE_LABELS = {
    "min_temp_c": "Min temp",
    "max_temp_c": "Max temp",
    "mean_temp_c": "Mean temp",
    "mean_humidity": "Mean humidity",
    "mean_wind_speed": "Mean wind speed",
    "mean_solar_irradiance": "Mean solar irradiance",
    "max_solar_irradiance": "Max solar irradiance",
    "day_of_week": "Day of week",
}
LMP_COLUMNS = tuple(f"lmp_opr_hour_{hour:02d}" for hour in range(1, 25))

PRICE_COLOR = "#1f5a9d"
ORACLE_COLOR = "#0b2f6f"
DISPATCH_COLOR = "#d77a00"
SOC_COLOR = "#2f7d32"
GRAY = "#9ca3af"


@dataclass(frozen=True, slots=True)
class PlotOptions:
    moving_average_window: int = 30
    daily_oracle_ymax: float | None = 1000.0
    representative_week_start: str | None = None
    representative_lookback_days: int | None = 90
    soc_smoothing_hours: int = 5
    tail_fraction: float = 0.25
    bootstrap_samples: int = 5_000
    bootstrap_seed: int = 0
    objective_tolerance: float = 1e-6
    log_every: int = 100


def load_caiso_dataset(dataset_path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path, parse_dates=[DATE_COLUMN])
    required_columns = [DATE_COLUMN, *FEATURE_COLUMNS, *LMP_COLUMNS]
    missing_columns = sorted(set(required_columns) - set(dataset.columns))
    if missing_columns:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing_columns))

    frame = (
        dataset.loc[:, required_columns]
        .dropna(subset=[*FEATURE_COLUMNS, *LMP_COLUMNS])
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ValueError("No complete CAISO rows remain after dropping missing values.")
    return frame


def build_oracle_dispatch_frames(
    dataset: pd.DataFrame,
    *,
    parameters: StorageDispatchParameters,
    objective_tolerance: float,
    log_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []
    solver_params = {"Threads": 1, "Seed": 0}

    total_days = len(dataset)
    for row_idx, row in dataset.iterrows():
        date = pd.Timestamp(row[DATE_COLUMN])
        prices = row.loc[list(LMP_COLUMNS)].to_numpy(dtype=float, copy=True)
        result = solve_storage_dispatch_lexicographic(
            prices,
            parameters,
            name=f"caiso_oracle_{date:%Y%m%d}",
            log_to_console=False,
            solver_params=solver_params,
            objective_tolerance=objective_tolerance,
        )

        daily_rows.append(
            {
                "date": date,
                "oracle_value": result.objective_value,
                "revenue_value": result.revenue_value,
                "throughput_penalty_value": result.throughput_penalty_value,
                "throughput": result.throughput_value,
                "mean_price": float(prices.mean()),
                "price_spread": float(prices.max() - prices.min()),
            }
        )
        for hour_idx, price in enumerate(prices):
            charge = float(result.charge[hour_idx])
            discharge = float(result.discharge[hour_idx])
            hourly_rows.append(
                {
                    "date": date,
                    "month": date.to_period("M").strftime("%Y-%m"),
                    "hour": hour_idx,
                    "price": float(price),
                    "charge": charge,
                    "discharge": discharge,
                    "dispatch": charge - discharge,
                    "state_of_charge_start": float(result.state_of_charge[hour_idx]),
                    "state_of_charge_end": float(result.state_of_charge[hour_idx + 1]),
                    "mode": int(result.mode[hour_idx]),
                }
            )

        if log_every > 0 and ((row_idx + 1) == 1 or (row_idx + 1) % log_every == 0):
            print(f"[oracle] solved {row_idx + 1}/{total_days} days", flush=True)
    print(f"[oracle] solved {total_days}/{total_days} days", flush=True)

    daily = pd.DataFrame(daily_rows)
    hourly = pd.DataFrame(hourly_rows)
    return daily, hourly


def choose_representative_week(
    daily: pd.DataFrame,
    requested_start: str | None,
    lookback_days: int | None,
) -> pd.Timestamp:
    daily_sorted = daily.sort_values("date").reset_index(drop=True)
    if requested_start is not None:
        start = pd.Timestamp(requested_start)
        selected_dates = set(pd.date_range(start, periods=7, freq="D"))
        available_dates = set(pd.to_datetime(daily_sorted["date"]))
        if not selected_dates.issubset(available_dates):
            raise ValueError(
                f"Requested representative week starting {start:%Y-%m-%d} is not fully available."
            )
        return start

    if lookback_days is not None and lookback_days <= 0:
        raise ValueError("lookback_days must be strictly positive when provided.")

    candidate_daily = daily_sorted
    if lookback_days is not None:
        latest_date = pd.Timestamp(daily_sorted["date"].max())
        earliest_date = latest_date - pd.Timedelta(days=lookback_days - 1)
        candidate_daily = daily_sorted.loc[
            daily_sorted["date"] >= earliest_date
        ].reset_index(drop=True)
        if len(candidate_daily) < 7:
            raise ValueError(
                f"Need at least seven recent rows to choose a representative week; "
                f"found {len(candidate_daily)} rows in the last {lookback_days} days."
            )

    target_median = float(candidate_daily["oracle_value"].median())
    candidates: list[tuple[float, pd.Timestamp]] = []
    for start_idx in range(0, len(candidate_daily) - 6):
        window = candidate_daily.iloc[start_idx : start_idx + 7]
        dates = pd.to_datetime(window["date"]).reset_index(drop=True)
        if (dates.iloc[-1] - dates.iloc[0]).days != 6:
            continue
        if not dates.diff().dropna().eq(pd.Timedelta(days=1)).all():
            continue
        window_mean = float(window["oracle_value"].mean())
        candidates.append((abs(window_mean - target_median), pd.Timestamp(dates.iloc[0])))

    if not candidates:
        raise ValueError("Could not find a complete seven-day representative week.")
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def bootstrap_tail_differences(
    frame: pd.DataFrame,
    *,
    tail_fraction: float,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    if not 0 < tail_fraction < 0.5:
        raise ValueError("tail_fraction must be in the open interval (0, 0.5).")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be strictly positive.")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for feature_idx, feature in enumerate(FEATURE_COLUMNS):
        lower_threshold = float(frame[feature].quantile(tail_fraction))
        upper_threshold = float(frame[feature].quantile(1.0 - tail_fraction))
        bottom_values = frame.loc[
            frame[feature] <= lower_threshold,
            "oracle_value",
        ].to_numpy(dtype=float)
        top_values = frame.loc[
            frame[feature] >= upper_threshold,
            "oracle_value",
        ].to_numpy(dtype=float)
        if len(bottom_values) == 0 or len(top_values) == 0:
            raise ValueError(f"Empty top or bottom tail for feature {feature}.")

        estimate = float(top_values.mean() - bottom_values.mean())
        boot = np.empty(bootstrap_samples, dtype=float)
        for sample_idx in range(bootstrap_samples):
            bottom_sample = rng.choice(bottom_values, size=len(bottom_values), replace=True)
            top_sample = rng.choice(top_values, size=len(top_values), replace=True)
            boot[sample_idx] = top_sample.mean() - bottom_sample.mean()
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
        rows.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "feature_order": feature_idx,
                "tail_fraction": tail_fraction,
                "lower_threshold": lower_threshold,
                "upper_threshold": upper_threshold,
                "bottom_tail_days": int(len(bottom_values)),
                "top_tail_days": int(len(top_values)),
                "mean_oracle_bottom_tail": float(bottom_values.mean()),
                "mean_oracle_top_tail": float(top_values.mean()),
                "difference_top_minus_bottom": estimate,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def apply_plot_style() -> None:
    register_latin_modern_fonts()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Latin Modern Roman",
                "Latin Modern Roman 10",
                "CMU Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.7,
        }
    )


def register_latin_modern_fonts() -> None:
    font_dirs = (
        Path.home() / "Library" / "Fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
    )
    for font_dir in font_dirs:
        if not font_dir.exists():
            continue
        for font_path in font_dir.glob("lmroman*.otf"):
            font_manager.fontManager.addfont(font_path)


def plot_daily_oracle(
    daily: pd.DataFrame,
    *,
    outdir: Path,
    moving_average_window: int,
    daily_oracle_ymax: float | None,
) -> None:
    if moving_average_window <= 0:
        raise ValueError("moving_average_window must be strictly positive.")
    if daily_oracle_ymax is not None and daily_oracle_ymax <= 0:
        raise ValueError("daily_oracle_ymax must be positive when provided.")

    frame = daily.sort_values("date").copy()
    frame["moving_average"] = (
        frame["oracle_value"].rolling(moving_average_window, min_periods=1).mean()
    )
    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    ax.plot(
        frame["date"],
        frame["oracle_value"],
        color=GRAY,
        linewidth=0.65,
        alpha=0.35,
        label="Daily oracle value",
    )
    ax.plot(
        frame["date"],
        frame["moving_average"],
        color=ORACLE_COLOR,
        linewidth=2.0,
        label=f"{moving_average_window}-day moving average",
    )
    ax.set_title("Daily CAISO Oracle Value")
    ax.set_xlabel("Date")
    ax.set_ylabel("Oracle value")
    if daily_oracle_ymax is not None:
        lower_limit = min(float(frame["oracle_value"].min()), 0.0)
        ax.set_ylim(lower_limit, daily_oracle_ymax)
    ax.legend(loc="upper left", frameon=False)
    ax.margins(x=0.01)
    save_figure(fig, outdir, "daily_oracle_value")


def plot_representative_week(
    daily: pd.DataFrame,
    hourly: pd.DataFrame,
    *,
    outdir: Path,
    representative_week_start: str | None,
    representative_lookback_days: int | None,
    soc_smoothing_hours: int,
    energy_capacity: float,
) -> pd.Timestamp:
    if soc_smoothing_hours <= 0:
        raise ValueError("soc_smoothing_hours must be strictly positive.")

    week_start = choose_representative_week(
        daily,
        representative_week_start,
        representative_lookback_days,
    )
    week_dates = pd.date_range(week_start, periods=7, freq="D")
    week = (
        hourly.loc[hourly["date"].isin(week_dates)]
        .sort_values(["date", "hour"])
        .reset_index(drop=True)
    )
    if len(week) != 168:
        raise ValueError(
            f"Representative week starting {week_start:%Y-%m-%d} has {len(week)} hourly rows."
        )

    hour_starts = np.arange(168)
    hour_edges = np.arange(169)
    prices = week["price"].to_numpy(dtype=float)
    dispatch = week["dispatch"].to_numpy(dtype=float)
    soc = np.concatenate(
        (
            week["state_of_charge_start"].iloc[:1].to_numpy(dtype=float),
            week["state_of_charge_end"].to_numpy(dtype=float),
        )
    )
    soc_smoothed = (
        pd.Series(soc)
        .rolling(soc_smoothing_hours, center=True, min_periods=1)
        .mean()
        .clip(lower=0.0, upper=energy_capacity)
        .to_numpy(dtype=float)
    )

    fig, ax_price = plt.subplots(figsize=(9.6, 4.55))
    ax_dispatch = ax_price.twinx()
    ax_soc = ax_price.twinx()
    ax_soc.spines["right"].set_position(("axes", 1.12))
    ax_soc.spines["right"].set_visible(True)
    fig.subplots_adjust(right=0.78, top=0.78, bottom=0.26)
    ax_price.grid(False)
    ax_dispatch.grid(False)
    ax_soc.grid(False)

    price_line = ax_price.plot(
        hour_starts,
        prices,
        color=PRICE_COLOR,
        linewidth=1.4,
        label="Hourly price",
    )[0]
    dispatch_line = ax_dispatch.step(
        hour_edges,
        np.r_[dispatch, dispatch[-1]],
        where="post",
        color=DISPATCH_COLOR,
        linewidth=1.35,
        label="Charge / discharge",
    )[0]
    soc_line = ax_soc.plot(
        hour_edges,
        soc_smoothed,
        color=SOC_COLOR,
        linewidth=1.8,
        label="Smoothed state of charge",
    )[0]

    for day_boundary in range(24, 168, 24):
        ax_price.axvline(
            day_boundary,
            color="#9ca3af",
            linewidth=0.8,
            linestyle="--",
            alpha=0.55,
        )
    for day_idx, date in enumerate(week_dates):
        ax_price.text(
            day_idx * 24 + 12,
            1.035,
            f"{date.strftime('%a')}\n{date.strftime('%b')} {date.day}",
            transform=ax_price.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )

    ax_price.set_xlim(0, 168)
    ax_price.set_xticks(np.arange(0, 169, 24))
    ax_price.set_xlabel("Hour of representative week")
    ax_price.set_ylabel("Hourly price", color=PRICE_COLOR)
    ax_dispatch.set_ylabel("Optimal dispatch (+ charge / - discharge)", color=DISPATCH_COLOR)
    ax_soc.set_ylabel("State of charge", color=SOC_COLOR)
    ax_price.tick_params(axis="y", colors=PRICE_COLOR)
    ax_dispatch.tick_params(axis="y", colors=DISPATCH_COLOR)
    ax_soc.tick_params(axis="y", colors=SOC_COLOR)
    price_range = max(float(prices.max() - prices.min()), 1.0)
    price_lower = np.floor((float(prices.min()) - 0.65 * price_range) / 10.0) * 10.0
    price_upper = np.ceil((float(prices.max()) + 0.75 * price_range) / 10.0) * 10.0
    ax_price.set_ylim(price_lower, price_upper)
    dispatch_limit = max(float(np.max(np.abs(dispatch))) * 1.85, 1.85)
    ax_dispatch.set_ylim(-dispatch_limit, dispatch_limit)
    ax_soc.set_ylim(-0.7 * energy_capacity, 1.7 * energy_capacity)
    ax_price.set_title(f"Representative Week Dispatch: {week_start:%Y-%m-%d}", pad=42)
    ax_price.legend(
        handles=[price_line, dispatch_line, soc_line],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        frameon=False,
        ncol=3,
    )
    save_figure(fig, outdir, "representative_week_dispatch")
    return week_start


def plot_dispatch_heatmap(hourly: pd.DataFrame, *, outdir: Path) -> None:
    frame = hourly.copy()
    frame["calendar_month"] = pd.to_datetime(frame["date"]).dt.month
    monthly_hourly = (
        frame.groupby(["calendar_month", "hour"], observed=True)["dispatch"]
        .mean()
        .unstack("hour")
        .reindex(index=range(1, 13), columns=range(24))
    )
    month_labels = [calendar.month_abbr[month] for month in monthly_hourly.index]
    values = monthly_hourly.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(values)))
    if vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    image = ax.imshow(
        values,
        aspect="auto",
        cmap=plt.get_cmap("cmc.vik"),
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        interpolation="nearest",
    )
    ax.grid(False)
    ax.set_title("Average Optimal Dispatch by Month and Hour")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Month")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xticklabels([str(hour) for hour in range(0, 24, 2)])
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(month_labels)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Average optimal dispatch (+ charge / - discharge)")
    save_figure(fig, outdir, "monthly_hourly_dispatch_heatmap")


def plot_covariate_tail_differences(
    summary: pd.DataFrame,
    *,
    outdir: Path,
    tail_fraction: float,
) -> None:
    frame = summary.sort_values("feature_order", ascending=False).reset_index(drop=True)
    y = np.arange(len(frame))
    estimates = frame["difference_top_minus_bottom"].to_numpy(dtype=float)
    ci_low = frame["ci_low"].to_numpy(dtype=float)
    ci_high = frame["ci_high"].to_numpy(dtype=float)
    xerr = np.vstack((estimates - ci_low, ci_high - estimates))

    fig, ax = plt.subplots(figsize=(7.1, 3.9))
    ax.axvline(0.0, color="#374151", linewidth=1.0, linestyle=":", zorder=1)
    ax.errorbar(
        estimates,
        y,
        xerr=xerr,
        fmt="o",
        color=ORACLE_COLOR,
        ecolor=ORACLE_COLOR,
        elinewidth=1.3,
        capsize=3,
        markersize=5,
        zorder=2,
    )
    x_span = max(float(ci_high.max() - ci_low.min()), 1.0)
    label_offset = 0.018 * x_span
    for estimate, low, high, y_value in zip(estimates, ci_low, ci_high, y, strict=True):
        if estimate >= 0:
            label_x = high + label_offset
            ha = "left"
        else:
            label_x = low - label_offset
            ha = "right"
        ax.text(
            label_x,
            y_value,
            f"{estimate:+.1f}",
            color=ORACLE_COLOR,
            fontsize=8,
            va="center",
            ha=ha,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(frame["feature_label"])
    ax.set_xlabel("Oracle value difference: top tail minus bottom tail")
    ax.set_ylabel("Covariate")
    ax.set_title(f"Oracle Value Difference by Covariate Tail ({tail_fraction:.0%} tails)")
    ax.margins(x=0.12)
    ax.margins(y=0.12)
    save_figure(fig, outdir, "covariate_oracle_value_tail_difference")


def write_metadata(
    *,
    outdir: Path,
    dataset_path: Path,
    dataset: pd.DataFrame,
    parameters: StorageDispatchParameters,
    options: PlotOptions,
    representative_week_start: pd.Timestamp,
) -> None:
    date_min = pd.Timestamp(dataset[DATE_COLUMN].min())
    date_max = pd.Timestamp(dataset[DATE_COLUMN].max())
    duration_months = (date_max - date_min).days / 365.25 * 12.0
    metadata = {
        "dataset_path": str(dataset_path),
        "date_min": date_min.strftime("%Y-%m-%d"),
        "date_max": date_max.strftime("%Y-%m-%d"),
        "days": int(len(dataset)),
        "duration_months": duration_months,
        "calendar_months": int(dataset[DATE_COLUMN].dt.to_period("M").nunique()),
        "storage_parameters": asdict(parameters),
        "plot_options": asdict(options),
        "representative_week_start": representative_week_start.strftime("%Y-%m-%d"),
    }
    with (outdir / "plot_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the four CAISO case-study oracle dispatch plots.",
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--throughput-penalty", type=float, default=5.0)
    parser.add_argument("--energy-capacity", type=float, default=4.0)
    parser.add_argument("--power-limit", type=float, default=1.0)
    parser.add_argument("--charge-efficiency", type=float, default=0.95)
    parser.add_argument("--discharge-efficiency", type=float, default=0.95)
    parser.add_argument("--initial-soc", type=float, default=2.0)
    parser.add_argument("--terminal-soc", type=float, default=2.0)
    parser.add_argument("--moving-average-window", type=int, default=30)
    parser.add_argument("--daily-oracle-ymax", type=float, default=1000.0)
    parser.add_argument("--representative-week-start", default=None)
    parser.add_argument("--representative-lookback-days", type=int, default=90)
    parser.add_argument("--soc-smoothing-hours", type=int, default=5)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--objective-tolerance", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def main(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    outdir: Path = DEFAULT_OUTDIR,
    throughput_penalty: float = 5.0,
    energy_capacity: float = 4.0,
    power_limit: float = 1.0,
    charge_efficiency: float = 0.95,
    discharge_efficiency: float = 0.95,
    initial_soc: float = 2.0,
    terminal_soc: float = 2.0,
    moving_average_window: int = 30,
    daily_oracle_ymax: float | None = 1000.0,
    representative_week_start: str | None = None,
    representative_lookback_days: int | None = 90,
    soc_smoothing_hours: int = 5,
    tail_fraction: float = 0.25,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 0,
    objective_tolerance: float = 1e-6,
    log_every: int = 100,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    parameters = StorageDispatchParameters(
        energy_capacity=energy_capacity,
        power_limit=power_limit,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
        throughput_penalty=throughput_penalty,
        initial_state_of_charge=initial_soc,
        terminal_state_of_charge=terminal_soc,
    )
    options = PlotOptions(
        moving_average_window=moving_average_window,
        daily_oracle_ymax=daily_oracle_ymax,
        representative_week_start=representative_week_start,
        representative_lookback_days=representative_lookback_days,
        soc_smoothing_hours=soc_smoothing_hours,
        tail_fraction=tail_fraction,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        objective_tolerance=objective_tolerance,
        log_every=log_every,
    )

    dataset = load_caiso_dataset(dataset_path)
    daily, hourly = build_oracle_dispatch_frames(
        dataset,
        parameters=parameters,
        objective_tolerance=objective_tolerance,
        log_every=log_every,
    )
    enriched_daily = dataset.loc[:, [DATE_COLUMN, *FEATURE_COLUMNS]].merge(
        daily,
        on="date",
        validate="one_to_one",
    )
    tail_summary = bootstrap_tail_differences(
        enriched_daily,
        tail_fraction=tail_fraction,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )

    daily.to_csv(outdir / "daily_oracle_values.csv", index=False)
    hourly.to_csv(outdir / "hourly_oracle_dispatch.csv", index=False)
    tail_summary.to_csv(outdir / "covariate_oracle_value_tail_difference.csv", index=False)

    plot_daily_oracle(
        daily,
        outdir=outdir,
        moving_average_window=moving_average_window,
        daily_oracle_ymax=daily_oracle_ymax,
    )
    chosen_week_start = plot_representative_week(
        daily,
        hourly,
        outdir=outdir,
        representative_week_start=representative_week_start,
        representative_lookback_days=representative_lookback_days,
        soc_smoothing_hours=soc_smoothing_hours,
        energy_capacity=energy_capacity,
    )
    plot_dispatch_heatmap(hourly, outdir=outdir)
    plot_covariate_tail_differences(
        tail_summary,
        outdir=outdir,
        tail_fraction=tail_fraction,
    )
    write_metadata(
        outdir=outdir,
        dataset_path=dataset_path,
        dataset=dataset,
        parameters=parameters,
        options=options,
        representative_week_start=chosen_week_start,
    )
    print(f"Wrote CAISO case-study plots to {outdir}", flush=True)


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(**vars(args))
