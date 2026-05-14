from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri  # noqa: F401 - registers cmc.* colormaps with Matplotlib.
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


def save_both(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_solar_shap_quartile_frame(
    *,
    mse_daily: pd.DataFrame,
    spo_daily: pd.DataFrame,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    date_frame = (
        mse_daily[["date"]]
        .merge(spo_daily[["date"]], on="date", validate="one_to_one")
        .merge(
            dataset[["date", "mean_solar_irradiance", "max_solar_irradiance"]],
            on="date",
            validate="one_to_one",
        )
    )
    date_frame["solar_quartile"] = pd.qcut(
        date_frame["mean_solar_irradiance"],
        4,
        labels=QUARTILE_LABELS,
    )

    rows = []
    shap_columns = [
        *[f"predictive_shap_{feature}" for feature in SOLAR_FEATURES],
        *[f"decision_shap_{feature}" for feature in SOLAR_FEATURES],
    ]
    for model_label, daily in (("MSE-only", mse_daily), ("SPO+", spo_daily)):
        frame = daily[["date", *shap_columns]].merge(
            date_frame,
            on="date",
            validate="one_to_one",
        )
        frame["combined_prediction_solar_shap"] = combine_solar_shap(
            frame,
            "predictive",
        )
        frame["combined_decision_solar_shap"] = combine_solar_shap(
            frame,
            "decision",
        )
        for attribution, column in (
            ("Prediction SHAP", "combined_prediction_solar_shap"),
            ("Decision SHAP", "combined_decision_solar_shap"),
        ):
            attribution_frame = frame[
                [
                    "date",
                    "mean_solar_irradiance",
                    "max_solar_irradiance",
                    "solar_quartile",
                    column,
                ]
            ].rename(columns={column: "combined_solar_shap"})
            attribution_frame = attribution_frame.assign(
                model=model_label,
                attribution=attribution,
            )
            rows.extend(attribution_frame.to_dict(orient="records"))

    frame = pd.DataFrame(rows)
    scale = frame.groupby(["model", "attribution"], observed=True)[
        "combined_solar_shap"
    ].transform(lambda values: max(float(values.abs().max()), 1.0))
    frame["normalized_combined_solar_shap"] = frame["combined_solar_shap"] / scale
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create standalone CAISO solar plots.",
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
    mse_daily, mse_meta = read_result(mse_root)
    spo_daily, spo_meta = read_result(spo_root)
    dataset_path = Path(str(mse_meta["dataset_path"]))
    if Path(str(spo_meta["dataset_path"])) != dataset_path:
        raise ValueError("MSE and SPO+ runs use different dataset paths.")
    dataset = pd.read_csv(dataset_path, parse_dates=["date"])

    merged = (
        mse_daily[["date", "oracle_obj", "decision_full_value"]]
        .rename(columns={"decision_full_value": "mse_decision_value"})
        .merge(
            spo_daily[
                [
                    "date",
                    "decision_full_value",
                    *[f"predictive_shap_{feature}" for feature in SOLAR_FEATURES],
                    *[f"decision_shap_{feature}" for feature in SOLAR_FEATURES],
                ]
            ].rename(columns={"decision_full_value": "spo_decision_value"}),
            on="date",
            validate="one_to_one",
        )
        .merge(
            dataset[["date", "mean_solar_irradiance", "max_solar_irradiance", *LMP_COLUMNS]],
            on="date",
            validate="one_to_one",
        )
    )
    merged["solar_quartile"] = pd.qcut(
        merged["mean_solar_irradiance"],
        4,
        labels=QUARTILE_LABELS,
    )
    merged["actual_average_price"] = merged[LMP_COLUMNS].mean(axis=1)
    merged["actual_daily_spread"] = merged[LMP_COLUMNS].max(axis=1) - merged[LMP_COLUMNS].min(axis=1)
    merged["evening_minus_midday_spread"] = (
        merged[EVENING_COLUMNS].mean(axis=1) - merged[MIDDAY_COLUMNS].mean(axis=1)
    )
    merged["combined_prediction_solar_shap"] = combine_solar_shap(merged, "predictive")
    merged["combined_decision_solar_shap"] = combine_solar_shap(merged, "decision")

    quartile_metrics = (
        merged.groupby("solar_quartile", observed=True)
        .agg(
            actual_average_price=("actual_average_price", "mean"),
            actual_daily_spread=("actual_daily_spread", "mean"),
            evening_minus_midday_spread=("evening_minus_midday_spread", "mean"),
            oracle_value=("oracle_obj", "mean"),
            mse_decision_value=("mse_decision_value", "mean"),
            spo_decision_value=("spo_decision_value", "mean"),
        )
        .reset_index()
    )

    cmap = plt.get_cmap("cmc.batlow")
    colors = {
        "average_price": cmap(0.10),
        "daily_spread": cmap(0.34),
        "evening_midday": cmap(0.57),
        "oracle": cmap(0.86),
        "mse": cmap(0.27),
        "spo": cmap(0.68),
        "prediction": cmap(0.16),
        "decision": "#9a4f11",
    }
    plt.rcParams.update(
        {
            "font.size": 9,
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

    x = np.arange(len(quartile_metrics))
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for column, label, color, marker, linewidth in (
        ("actual_average_price", "Average price", colors["average_price"], "o", 2.0),
        ("actual_daily_spread", "Daily spread", colors["daily_spread"], "o", 2.0),
        ("evening_minus_midday_spread", "Evening-midday spread", colors["evening_midday"], "o", 2.0),
        ("oracle_value", "Oracle value", colors["oracle"], "o", 2.2),
        ("mse_decision_value", "MSE decision value", colors["mse"], "s", 1.9),
        ("spo_decision_value", "SPO+ decision value", colors["spo"], "s", 1.9),
    ):
        ax.plot(
            x,
            quartile_metrics[column],
            marker=marker,
            linewidth=linewidth,
            color=color,
            label=label,
        )
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(QUARTILE_LABELS)
    ax.set_xlabel("Mean solar irradiance quartile")
    ax.set_ylabel("Daily mean / realized value")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    save_both(fig, outdir, "solar_quartiles_with_mse_spo_performance")

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.scatter(
        merged["mean_solar_irradiance"],
        merged["combined_prediction_solar_shap"],
        s=30,
        color=colors["prediction"],
        alpha=0.82,
        edgecolors="white",
        linewidths=0.35,
    )
    xs, ys = fit_line(merged["mean_solar_irradiance"], merged["combined_prediction_solar_shap"])
    ax.plot(xs, ys, color=colors["prediction"], linewidth=2.2)
    ax.axhline(0, color="#222222", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Mean solar irradiance")
    ax.set_ylabel("Combined Prediction SHAP", color=colors["prediction"])
    ax.tick_params(axis="y", labelcolor=colors["prediction"])
    ax.spines["left"].set_color(colors["prediction"])
    ax.grid(axis="both", alpha=0.18)

    ax2 = ax.twinx()
    ax2.scatter(
        merged["mean_solar_irradiance"],
        merged["combined_decision_solar_shap"],
        s=30,
        color=colors["decision"],
        alpha=0.80,
        marker="s",
        edgecolors="white",
        linewidths=0.35,
    )
    xs2, ys2 = fit_line(merged["mean_solar_irradiance"], merged["combined_decision_solar_shap"])
    ax2.plot(xs2, ys2, color=colors["decision"], linewidth=2.2)
    ax2.axhline(0, color="#222222", linewidth=0.7, alpha=0.35, linestyle="--")
    ax2.set_ylabel("Combined Decision SHAP", color=colors["decision"])
    ax2.tick_params(axis="y", labelcolor=colors["decision"])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(colors["decision"])
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=colors["prediction"], label="Prediction SHAP", markersize=6, linewidth=2),
            Line2D([0], [0], marker="s", color=colors["decision"], label="Decision SHAP", markersize=6, linewidth=2),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )
    save_both(fig, outdir, "spo_solar_shap_vs_irradiance")

    shap_quartile_frame = build_solar_shap_quartile_frame(
        mse_daily=mse_daily,
        spo_daily=spo_daily,
        dataset=dataset,
    )
    attr_order = ("Prediction SHAP", "Decision SHAP")
    model_order = ("MSE-only", "SPO+")
    attr_colors = {
        "Prediction SHAP": colors["prediction"],
        "Decision SHAP": colors["decision"],
    }
    box_width = 0.28
    x_offsets = {
        "Prediction SHAP": -box_width / 1.6,
        "Decision SHAP": box_width / 1.6,
    }
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 4.6),
        sharey=True,
        constrained_layout=False,
    )
    x_positions = np.arange(len(QUARTILE_LABELS), dtype=float)
    for ax, model_label in zip(axes, model_order, strict=True):
        model_frame = shap_quartile_frame[shap_quartile_frame["model"] == model_label]
        for attribution in attr_order:
            values_by_quartile = [
                model_frame.loc[
                    (model_frame["solar_quartile"] == quartile)
                    & (model_frame["attribution"] == attribution),
                    "normalized_combined_solar_shap",
                ].to_numpy(dtype=float)
                for quartile in QUARTILE_LABELS
            ]
            box = ax.boxplot(
                values_by_quartile,
                positions=x_positions + x_offsets[attribution],
                widths=box_width,
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "#222222", "linewidth": 1.2},
                boxprops={
                    "facecolor": attr_colors[attribution],
                    "edgecolor": "#222222",
                    "linewidth": 0.8,
                    "alpha": 0.82,
                },
                whiskerprops={"color": "#222222", "linewidth": 0.8},
                capprops={"color": "#222222", "linewidth": 0.8},
                flierprops={
                    "marker": "o",
                    "markersize": 3.2,
                    "markerfacecolor": attr_colors[attribution],
                    "markeredgecolor": "white",
                    "markeredgewidth": 0.35,
                    "alpha": 0.7,
                },
            )
            for patch in box["boxes"]:
                patch.set_facecolor(attr_colors[attribution])
        ax.axhline(0, color="#222222", linewidth=0.8, alpha=0.7)
        ax.set_title(model_label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(QUARTILE_LABELS)
        ax.set_xlabel("Mean solar irradiance quartile")
        ax.set_ylim(-1.08, 1.08)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Signed normalized combined solar SHAP")
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=attr_colors[attribution],
                marker="s",
                linewidth=0,
                markersize=8,
                label=attribution,
            )
            for attribution in attr_order
        ],
        loc="lower center",
        bbox_to_anchor=(1.05, 1.02),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14, wspace=0.16)
    save_both(fig, outdir, "solar_shap_by_irradiance_quartile_normalized_boxplots")

    quartile_metrics.to_csv(outdir / "solar_quartiles_with_mse_spo_performance.csv", index=False)
    merged[
        [
            "date",
            "mean_solar_irradiance",
            "max_solar_irradiance",
            "combined_prediction_solar_shap",
            "combined_decision_solar_shap",
        ]
    ].to_csv(outdir / "spo_solar_shap_vs_irradiance.csv", index=False)
    shap_quartile_frame.to_csv(
        outdir / "solar_shap_by_irradiance_quartile_normalized_boxplots.csv",
        index=False,
    )
    (
        shap_quartile_frame.groupby(
            ["model", "solar_quartile", "attribution"],
            observed=True,
        )
        .agg(
            count=("combined_solar_shap", "count"),
            raw_mean=("combined_solar_shap", "mean"),
            raw_median=("combined_solar_shap", "median"),
            raw_q25=("combined_solar_shap", lambda values: values.quantile(0.25)),
            raw_q75=("combined_solar_shap", lambda values: values.quantile(0.75)),
            normalized_mean=("normalized_combined_solar_shap", "mean"),
            normalized_median=("normalized_combined_solar_shap", "median"),
            normalized_q25=(
                "normalized_combined_solar_shap",
                lambda values: values.quantile(0.25),
            ),
            normalized_q75=(
                "normalized_combined_solar_shap",
                lambda values: values.quantile(0.75),
            ),
        )
        .reset_index()
        .to_csv(
            outdir / "solar_shap_by_irradiance_quartile_normalized_boxplot_summary.csv",
            index=False,
        )
    )

    print(f"Wrote {outdir / 'solar_quartiles_with_mse_spo_performance.png'}")
    print(f"Wrote {outdir / 'spo_solar_shap_vs_irradiance.png'}")
    print(f"Wrote {outdir / 'solar_shap_by_irradiance_quartile_normalized_boxplots.png'}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(mse_root=args.mse_root, spo_root=args.spo_root, outdir=args.outdir)
