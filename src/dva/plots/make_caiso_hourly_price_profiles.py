from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cmcrameri  # noqa: F401 - registers cmc.* colormaps with Matplotlib.
import matplotlib
import numpy as np
import pandas as pd

from dva.model.train import load_default_train_explain_split
from dva.plots.make_caiso_predicted_spread_quartiles import (
    read_metadata,
    train_from_metadata,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_MSE_ROOT = Path("results/caiso_normal_shap_torch_mlp_lp_no_params_matched_tp5")
DEFAULT_SPO_ROOT = Path("results/caiso_normal_shap_spo_mlp_lp_no_params_matched_tp5")
DEFAULT_OUTDIR = Path("data/plots/caiso_hourly_price_profiles_normal_shap_matched_tp5")


def predict_hourly_prices(model: Any, X: pd.DataFrame) -> np.ndarray:
    predictions = np.asarray(model.predict(X), dtype=float)
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]
    if predictions.shape[1] != 24:
        raise ValueError(f"Expected 24 hourly outputs, got shape {predictions.shape}.")
    return predictions


def hourly_profile_frame(
    *,
    label: str,
    prices: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for hour_idx in range(prices.shape[1]):
        hourly_values = prices[:, hour_idx]
        mean_price = float(np.mean(hourly_values))
        std_price = float(np.std(hourly_values, ddof=1))
        rows.append(
            {
                "series": label,
                "hour": hour_idx + 1,
                "mean_price": mean_price,
                "std_price": std_price,
                "lower_1std": mean_price - std_price,
                "upper_1std": mean_price + std_price,
            }
        )
    return pd.DataFrame(rows)


def daily_hourly_frame(
    *,
    dates: pd.Series,
    label: str,
    prices: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for row_idx, date in enumerate(pd.to_datetime(dates)):
        for hour_idx, value in enumerate(prices[row_idx], start=1):
            rows.append(
                {
                    "date": date,
                    "series": label,
                    "hour": hour_idx,
                    "price": float(value),
                }
            )
    return pd.DataFrame(rows)


def daily_spread_summary(prices: np.ndarray) -> tuple[float, float]:
    spreads = prices.max(axis=1) - prices.min(axis=1)
    return float(np.mean(spreads)), float(np.std(spreads, ddof=1))


def save_both(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean hourly price profiles with +/- one standard deviation "
            "for actual, MSE-predicted, and SPO+-predicted CAISO prices."
        ),
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

    mse_metadata = read_metadata(mse_root)
    spo_metadata = read_metadata(spo_root)
    if mse_metadata["dataset_path"] != spo_metadata["dataset_path"]:
        raise ValueError("MSE and SPO+ metadata point to different datasets.")
    if mse_metadata["feature_columns"] != spo_metadata["feature_columns"]:
        raise ValueError("MSE and SPO+ metadata use different feature columns.")
    if mse_metadata["target_columns"] != spo_metadata["target_columns"]:
        raise ValueError("MSE and SPO+ metadata use different target columns.")

    feature_columns = tuple(str(name) for name in spo_metadata["feature_columns"])
    target_columns = tuple(str(name) for name in spo_metadata["target_columns"])
    split = load_default_train_explain_split(
        dataset_path=spo_metadata["dataset_path"],
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
    daily_hourly = pd.concat(
        [
            daily_hourly_frame(
                dates=split.explain_dates,
                label=label,
                prices=prices,
            )
            for label, prices in series
        ],
        ignore_index=True,
    )
    spread_rows = []
    for label, prices in series:
        mean_spread, std_spread = daily_spread_summary(prices)
        spread_rows.append(
            {
                "series": label,
                "mean_daily_spread": mean_spread,
                "std_daily_spread": std_spread,
            }
        )
    spread_summary = pd.DataFrame(spread_rows)

    cmap = plt.get_cmap("cmc.batlow")
    colors = {
        "Actual": cmap(0.36),
        "MSE predicted": cmap(0.15),
        "SPO+ predicted": cmap(0.70),
    }
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

    y_min = float(profile["lower_1std"].min())
    y_max = float(profile["upper_1std"].max())
    y_padding = 0.04 * (y_max - y_min)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for label, _ in series:
        frame = profile[profile["series"] == label]
        hours = frame["hour"].to_numpy(dtype=float)
        mean_price = frame["mean_price"].to_numpy(dtype=float)
        lower = frame["lower_1std"].to_numpy(dtype=float)
        upper = frame["upper_1std"].to_numpy(dtype=float)
        color = colors[label]
        ax.fill_between(hours, lower, upper, color=color, alpha=0.22, linewidth=0)
        ax.plot(hours, mean_price, color=color, linewidth=2.4, label=label)
    spread_text = "Mean spread: " + ", ".join(
        f"{row['series'].replace(' predicted', '')} {row['mean_daily_spread']:.1f}"
        for row in spread_summary.to_dict(orient="records")
    )
    ax.text(
        0.03,
        0.05,
        spread_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
    )
    ax.set_xlim(1, 24)
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Hourly price")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left", frameon=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.14)
    save_both(fig, outdir, "hourly_price_profiles_mean_std")

    profile.to_csv(outdir / "hourly_price_profiles_mean_std.csv", index=False)
    daily_hourly.to_csv(outdir / "hourly_price_profiles_by_day.csv", index=False)
    spread_summary.to_csv(outdir / "hourly_price_profiles_spread_summary.csv", index=False)

    print(f"Wrote {outdir / 'hourly_price_profiles_mean_std.png'}")
    print(f"Wrote {outdir / 'hourly_price_profiles_mean_std.pdf'}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(mse_root=args.mse_root, spo_root=args.spo_root, outdir=args.outdir)
