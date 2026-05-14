from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri  # noqa: F401 - registers cmc.* colormaps with Matplotlib.
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from dva.model.train import load_default_train_explain_split
from dva.plots.make_caiso_hourly_price_profiles import (
    daily_spread_summary,
    hourly_profile_frame,
    predict_hourly_prices,
)
from dva.plots.make_caiso_predicted_spread_quartiles import train_from_metadata


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_MSE_ROOT = Path("results/caiso_faith_shap_torch_mlp_lp_no_params")
DEFAULT_SPO_ROOT = Path("results/caiso_faith_shap_spo_mlp_lp_no_params")
DEFAULT_OUTDIR = Path("data/plots/caiso_solar_four_panel")

SOLAR_FEATURES = ("mean_solar_irradiance", "max_solar_irradiance")
QUARTILE_LABELS = ("Q1\nlow", "Q2", "Q3", "Q4\nhigh")
LMP_COLUMNS = [f"lmp_opr_hour_{hour:02d}" for hour in range(1, 25)]
MIDDAY_COLUMNS = [f"lmp_opr_hour_{hour:02d}" for hour in range(10, 16)]
EVENING_COLUMNS = [f"lmp_opr_hour_{hour:02d}" for hour in range(17, 22)]


def read_result(root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    daily = pd.read_csv(root / "daily_shap.csv", parse_dates=["date"])
    with (root / "run_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return daily, metadata


def combine_solar_shap(frame: pd.DataFrame, prefix: str) -> pd.Series:
    columns = [f"{prefix}_shap_{feature}" for feature in SOLAR_FEATURES]
    return frame.loc[:, columns].sum(axis=1)


def tidy_panel_summary(panel: str, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "panel", panel)
    return out


def build_daily_policy_activity(
    *,
    dispatch_root: Path,
    date_quartiles: pd.DataFrame,
    model_label: str,
) -> pd.DataFrame:
    dispatch = pd.read_csv(dispatch_root / "daily_full_dispatch.csv", parse_dates=["date"])
    daily_activity = (
        dispatch.groupby("date", as_index=False)
        .agg(
            total_charge=("charge", "sum"),
            total_discharge=("discharge", "sum"),
        )
        .merge(date_quartiles, on="date", validate="one_to_one")
    )
    daily_activity["throughput"] = (
        daily_activity["total_charge"] + daily_activity["total_discharge"]
    )
    daily_activity["active"] = daily_activity["throughput"] > 1e-6
    daily_activity["model"] = model_label
    return daily_activity


def build_hourly_profile_outputs(
    *,
    mse_metadata: dict[str, object],
    spo_metadata: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mse_metadata["dataset_path"] != spo_metadata["dataset_path"]:
        raise ValueError("MSE and SPO+ metadata point to different datasets.")
    if mse_metadata["feature_columns"] != spo_metadata["feature_columns"]:
        raise ValueError("MSE and SPO+ metadata use different feature columns.")
    if mse_metadata["target_columns"] != spo_metadata["target_columns"]:
        raise ValueError("MSE and SPO+ metadata use different target columns.")

    feature_columns = tuple(str(name) for name in spo_metadata["feature_columns"])
    target_columns = tuple(str(name) for name in spo_metadata["target_columns"])
    split = load_default_train_explain_split(
        dataset_path=str(spo_metadata["dataset_path"]),
        holdout_days=int(spo_metadata["holdout_days"]),
        feature_columns=feature_columns,
        target_columns=target_columns,
    )

    mse_model = train_from_metadata(mse_metadata, split)
    spo_model = train_from_metadata(spo_metadata, split)
    actual_prices = split.y_explain.to_numpy(dtype=float)
    mse_prices = predict_hourly_prices(mse_model, split.X_explain)
    spo_prices = predict_hourly_prices(spo_model, split.X_explain)

    series = (
        ("Actual", actual_prices),
        ("MSE predicted", mse_prices),
        ("SPO+ predicted", spo_prices),
    )
    profile = pd.concat(
        [hourly_profile_frame(label=label, prices=prices) for label, prices in series],
        ignore_index=True,
    )
    spread_summary = pd.DataFrame(
        [
            {
                "series": label,
                "mean_daily_spread": daily_spread_summary(prices)[0],
                "std_daily_spread": daily_spread_summary(prices)[1],
            }
            for label, prices in series
        ]
    )
    return profile, spread_summary


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def fit_line(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_finite = x_values[finite]
    y_finite = y_values[finite]
    xs = np.linspace(float(x_finite.min()), float(x_finite.max()), 100)
    if len(x_finite) < 2:
        return xs, np.full_like(xs, np.nan)
    slope, intercept = np.polyfit(x_finite, y_finite, deg=1)
    return xs, slope * xs + intercept


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the CAISO solar four-panel figure.",
    )
    parser.add_argument("--mse-root", type=Path, default=DEFAULT_MSE_ROOT)
    parser.add_argument("--spo-root", type=Path, default=DEFAULT_SPO_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser


def main(
    *,
    mse_root: Path = DEFAULT_MSE_ROOT,
    spo_root: Path = DEFAULT_SPO_ROOT,
    outdir: Path = DEFAULT_OUTDIR,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / "caiso_solar_four_panel.png"
    out_pdf = outdir / "caiso_solar_four_panel.pdf"
    out_summary = outdir / "caiso_solar_four_panel_summary.csv"

    mse_daily, mse_meta = read_result(mse_root)
    spo_daily, spo_meta = read_result(spo_root)
    dataset_path = Path(str(mse_meta["dataset_path"]))
    if Path(str(spo_meta["dataset_path"])) != dataset_path:
        raise ValueError("MSE and SPO+ runs use different dataset paths.")

    dataset = pd.read_csv(dataset_path, parse_dates=["date"])
    base = (
        mse_daily[
            [
                "date",
                "oracle_obj",
                "decision_full_value",
                "predictive_decision_insertion_auc",
                "decision_decision_insertion_auc",
                "predictive_decision_infidelity",
                "decision_decision_infidelity",
                *[f"predictive_shap_{feature}" for feature in SOLAR_FEATURES],
                *[f"decision_shap_{feature}" for feature in SOLAR_FEATURES],
            ]
        ]
        .rename(columns={"decision_full_value": "mse_decision_value"})
        .merge(
            spo_daily[["date", "decision_full_value"]].rename(
                columns={"decision_full_value": "spo_decision_value"}
            ),
            on="date",
            validate="one_to_one",
        )
        .merge(dataset[["date", "mean_solar_irradiance", "max_solar_irradiance", *LMP_COLUMNS]], on="date", validate="one_to_one")
    )

    base["solar_quartile"] = pd.qcut(
        base["mean_solar_irradiance"],
        4,
        labels=QUARTILE_LABELS,
    )
    base["actual_average_price"] = base[LMP_COLUMNS].mean(axis=1)
    base["actual_daily_spread"] = base[LMP_COLUMNS].max(axis=1) - base[LMP_COLUMNS].min(axis=1)
    base["evening_minus_midday_spread"] = (
        base[EVENING_COLUMNS].mean(axis=1) - base[MIDDAY_COLUMNS].mean(axis=1)
    )
    base["combined_prediction_solar_shap"] = combine_solar_shap(base, "predictive")
    base["combined_decision_solar_shap"] = combine_solar_shap(base, "decision")
    date_quartiles = base[["date", "solar_quartile"]].copy()

    quartile_metrics = (
        base.groupby("solar_quartile", observed=True)
        .agg(
            actual_average_price=("actual_average_price", "mean"),
            actual_daily_spread=("actual_daily_spread", "mean"),
            evening_minus_midday_spread=("evening_minus_midday_spread", "mean"),
            oracle_value=("oracle_obj", "mean"),
            mse_decision_value=("mse_decision_value", "mean"),
            spo_decision_value=("spo_decision_value", "mean"),
            combined_prediction_solar_shap=("combined_prediction_solar_shap", "mean"),
            combined_decision_solar_shap=("combined_decision_solar_shap", "mean"),
        )
        .reset_index()
    )
    policy_daily = pd.concat(
        [
            build_daily_policy_activity(
                dispatch_root=mse_root,
                date_quartiles=date_quartiles,
                model_label="MSE-only",
            ),
            build_daily_policy_activity(
                dispatch_root=spo_root,
                date_quartiles=date_quartiles,
                model_label="SPO+",
            ),
        ],
        ignore_index=True,
    )
    policy_metrics = (
        policy_daily.groupby(["model", "solar_quartile"], observed=True)
        .agg(
            days=("active", "count"),
            active_days=("active", "sum"),
            active_rate=("active", "mean"),
            mean_throughput=("throughput", "mean"),
        )
        .reset_index()
    )
    hourly_profile, hourly_spread_summary = build_hourly_profile_outputs(
        mse_metadata=mse_meta,
        spo_metadata=spo_meta,
    )

    metrics = []
    for model_label, daily in (("MSE-only", mse_daily), ("SPO+", spo_daily)):
        metrics.extend(
            [
                {
                    "model": model_label,
                    "attribution": "Prediction SHAP",
                    "decision_insertion_auc": daily["predictive_decision_insertion_auc"].mean(skipna=True),
                    "decision_infidelity": daily["predictive_decision_infidelity"].mean(),
                },
                {
                    "model": model_label,
                    "attribution": "Decision SHAP",
                    "decision_insertion_auc": daily["decision_decision_insertion_auc"].mean(skipna=True),
                    "decision_infidelity": daily["decision_decision_infidelity"].mean(),
                },
            ]
        )
    metric_frame = pd.DataFrame(metrics)

    summary = pd.concat(
        [
            tidy_panel_summary("quartiles", quartile_metrics),
            tidy_panel_summary("policy_activity", policy_metrics),
            tidy_panel_summary("hourly_profile", hourly_profile),
            tidy_panel_summary("hourly_spread_summary", hourly_spread_summary),
            tidy_panel_summary("faithfulness", metric_frame),
        ],
        ignore_index=True,
        sort=False,
    )
    summary.to_csv(out_summary, index=False)

    cmap = plt.get_cmap("cmc.batlow")
    series_colors = {
        "average_price": cmap(0.12),
        "daily_spread": cmap(0.38),
        "evening_midday": cmap(0.62),
        "oracle": cmap(0.86),
        "prediction": cmap(0.18),
        "decision": "#9a4f11",
        "mse": cmap(0.30),
        "spo": cmap(0.68),
    }
    hourly_colors = {
        "Actual": series_colors["daily_spread"],
        "MSE predicted": series_colors["mse"],
        "SPO+ predicted": series_colors["spo"],
    }
    quartile_colors = [cmap(value) for value in np.linspace(0.16, 0.88, len(QUARTILE_LABELS))]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(12.5, 8.6), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.28, hspace=0.36)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    x = np.arange(len(quartile_metrics))
    ax_a.plot(
        x,
        quartile_metrics["actual_average_price"],
        marker="o",
        linewidth=2,
        color=series_colors["average_price"],
        label="Average price",
    )
    ax_a.plot(
        x,
        quartile_metrics["actual_daily_spread"],
        marker="o",
        linewidth=2,
        color=series_colors["daily_spread"],
        label="Daily spread",
    )
    ax_a.plot(
        x,
        quartile_metrics["evening_minus_midday_spread"],
        marker="o",
        linewidth=2,
        color=series_colors["evening_midday"],
        label="Evening-midday spread",
    )
    ax_a.plot(
        x,
        quartile_metrics["oracle_value"],
        marker="o",
        linewidth=2.2,
        color=series_colors["oracle"],
        label="Oracle value",
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(QUARTILE_LABELS)
    ax_a.set_xlabel("Mean solar irradiance quartile")
    ax_a.set_ylabel("Daily mean")
    ax_a.grid(axis="y", alpha=0.22)
    ax_a.legend(loc="upper left", frameon=False, ncol=2)
    add_panel_label(ax_a, "(a)")

    quartile_codes = base["solar_quartile"].cat.codes.to_numpy()
    point_colors = [quartile_colors[index] for index in quartile_codes]
    ax_b.scatter(
        base["mean_solar_irradiance"],
        base["combined_prediction_solar_shap"],
        s=30,
        color=series_colors["prediction"],
        alpha=0.82,
        edgecolors="white",
        linewidths=0.35,
        label="Prediction SHAP",
    )
    xs, ys = fit_line(base["mean_solar_irradiance"], base["combined_prediction_solar_shap"])
    ax_b.plot(xs, ys, color=series_colors["prediction"], linewidth=2.2)
    ax_b.axhline(0, color="#222222", linewidth=0.7, alpha=0.6)
    ax_b.set_xlabel("Mean solar irradiance")
    ax_b.set_ylabel("Combined Prediction SHAP", color=series_colors["prediction"])
    ax_b.tick_params(axis="y", labelcolor=series_colors["prediction"])
    ax_b.grid(axis="both", alpha=0.18)

    ax_b2 = ax_b.twinx()
    ax_b2.scatter(
        base["mean_solar_irradiance"],
        base["combined_decision_solar_shap"],
        s=30,
        color=series_colors["decision"],
        alpha=0.80,
        marker="s",
        edgecolors="white",
        linewidths=0.35,
        label="Decision SHAP",
    )
    xs2, ys2 = fit_line(base["mean_solar_irradiance"], base["combined_decision_solar_shap"])
    ax_b2.plot(xs2, ys2, color=series_colors["decision"], linewidth=2.2)
    ax_b2.axhline(0, color="#222222", linewidth=0.7, alpha=0.35, linestyle="--")
    ax_b2.set_ylabel("Combined Decision SHAP", color=series_colors["decision"])
    ax_b2.tick_params(axis="y", labelcolor=series_colors["decision"])
    ax_b2.spines["right"].set_visible(True)
    ax_b2.spines["right"].set_color(series_colors["decision"])
    ax_b.spines["left"].set_color(series_colors["prediction"])
    ax_b.legend(
        handles=[
            Line2D([0], [0], marker="o", color=series_colors["prediction"], label="Prediction SHAP", markersize=6, linewidth=2),
            Line2D([0], [0], marker="s", color=series_colors["decision"], label="Decision SHAP", markersize=6, linewidth=2),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )
    add_panel_label(ax_b, "(b)")

    for label in ("Actual", "MSE predicted", "SPO+ predicted"):
        profile = hourly_profile[hourly_profile["series"] == label]
        hours = profile["hour"].to_numpy(dtype=float)
        mean_price = profile["mean_price"].to_numpy(dtype=float)
        lower = profile["lower_1std"].to_numpy(dtype=float)
        upper = profile["upper_1std"].to_numpy(dtype=float)
        color = hourly_colors[label]
        ax_c.fill_between(hours, lower, upper, color=color, alpha=0.12, linewidth=0)
        ax_c.plot(hours, mean_price, color=color, linewidth=2.0, label=label)
    spread_text = "Mean spread: " + ", ".join(
        f"{row['series'].replace(' predicted', '')} {row['mean_daily_spread']:.1f}"
        for row in hourly_spread_summary.to_dict(orient="records")
    )
    ax_c.text(
        0.03,
        0.05,
        spread_text,
        transform=ax_c.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
    )
    ax_c.set_xlim(1, 24)
    ax_c.set_xticks([1, 6, 12, 18, 24])
    ax_c.set_xlabel("Hour of day")
    ax_c.set_ylabel("Hourly price")
    ax_c.grid(axis="y", alpha=0.22)
    ax_c.legend(loc="upper left", frameon=False, ncol=1)
    add_panel_label(ax_c, "(c)")

    d_width = 0.25
    for offset, column, label, color in (
        (-d_width, "oracle_value", "Oracle", series_colors["oracle"]),
        (0.0, "mse_decision_value", "MSE-only", series_colors["mse"]),
        (d_width, "spo_decision_value", "SPO+", series_colors["spo"]),
    ):
        values = quartile_metrics[column].to_numpy(dtype=float)
        ax_d.bar(x + offset, values, width=d_width, color=color, label=label)
        for xpos, value in zip(x + offset, values, strict=True):
            if value >= 0:
                ax_d.text(xpos, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
            else:
                ax_d.text(
                    xpos,
                    value / 2.0,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                )
    ax_d.axhline(0, color="#222222", linewidth=0.8)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(QUARTILE_LABELS)
    ax_d.set_xlabel("Mean solar irradiance quartile")
    ax_d.set_ylabel("Realized decision value")
    ax_d.set_ylim(-8, 59)
    ax_d.grid(axis="y", alpha=0.22)
    ax_d.legend(loc="upper left", frameon=False, ncol=3)
    add_panel_label(ax_d, "(d)")

    fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.08)
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(mse_root=args.mse_root, spo_root=args.spo_root, outdir=args.outdir)
