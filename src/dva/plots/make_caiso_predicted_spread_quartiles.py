from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cmcrameri  # noqa: F401 - registers cmc.* colormaps with Matplotlib.
import matplotlib
import numpy as np
import pandas as pd

from dva.model.storage_dispatch import StorageDispatchParameters
from dva.model.train import load_default_train_explain_split, train_model


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_MSE_ROOT = Path("results/caiso_normal_shap_torch_mlp_lp_no_params_matched_tp5")
DEFAULT_SPO_ROOT = Path("results/caiso_normal_shap_spo_mlp_lp_no_params_matched_tp5")
DEFAULT_OUTDIR = Path("data/plots/caiso_predicted_spread_solar_quartiles_normal_shap_matched_tp5")
QUARTILE_LABELS = ("Q1\nlow", "Q2", "Q3", "Q4\nhigh")
SOLAR_FEATURES = ("mean_solar_irradiance", "max_solar_irradiance")
FEATURE_LABELS = {
    "min_temp_c": "Minimum temperature",
    "max_temp_c": "Maximum temperature",
    "mean_temp_c": "Mean temperature",
    "mean_humidity": "Mean humidity",
    "mean_wind_speed": "Mean wind speed",
    "mean_solar_irradiance": "Mean solar irradiance",
    "max_solar_irradiance": "Maximum solar irradiance",
    "day_of_week": "Day of week",
}


def read_metadata(root: Path) -> dict[str, Any]:
    with (root / "run_metadata.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def storage_parameters_from_metadata(metadata: dict[str, Any]) -> StorageDispatchParameters:
    storage = metadata["storage_parameters"]
    return StorageDispatchParameters(
        energy_capacity=float(storage["energy_capacity"]),
        power_limit=float(storage["power_limit"]),
        charge_efficiency=float(storage["charge_efficiency"]),
        discharge_efficiency=float(storage["discharge_efficiency"]),
        throughput_penalty=float(storage["throughput_penalty"]),
        initial_state_of_charge=float(storage["initial_state_of_charge"]),
        terminal_state_of_charge=(
            None
            if storage["terminal_state_of_charge"] is None
            else float(storage["terminal_state_of_charge"])
        ),
    )


def train_from_metadata(metadata: dict[str, Any], split: Any) -> Any:
    artifacts = train_model(
        split.X_train,
        split.y_train,
        model_name=str(metadata["model_name"]),
        feature_columns=tuple(str(name) for name in metadata["feature_columns"]),
        target_columns=tuple(str(name) for name in metadata["target_columns"]),
        random_state=int(metadata["random_state"]),
        n_jobs=int(metadata["n_jobs"]) if metadata["n_jobs"] is not None else None,
        mlp_hidden_layer_sizes=tuple(
            int(value) for value in metadata["mlp_hidden_layer_sizes"]
        ),
        mlp_max_iter=int(metadata["mlp_max_iter"]),
        learning_rate=metadata["learning_rate"],
        mse_learning_rate=metadata["mse_learning_rate"],
        spo_learning_rate=metadata["spo_learning_rate"],
        storage_parameters=storage_parameters_from_metadata(metadata),
        training_verbose=False,
        training_log_every=metadata["training_log_every"],
        spo_processes=metadata["spo_processes"],
        spo_warm_start_with_mse=bool(metadata["spo_warm_start_with_mse"]),
    )
    return artifacts.model


def predicted_spread(model: Any, X: pd.DataFrame) -> np.ndarray:
    predictions = np.asarray(model.predict(X), dtype=float)
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]
    return predictions.max(axis=1) - predictions.min(axis=1)


def pearson_correlation(x: pd.Series, y: pd.Series) -> float:
    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if finite.sum() < 2:
        return np.nan
    return float(np.corrcoef(x_values[finite], y_values[finite])[0, 1])


def assign_quartiles(frame: pd.DataFrame, feature_name: str) -> pd.Series:
    # Rank first so tied feature values still produce four equally sized bins.
    ranked_feature = frame[feature_name].rank(method="first")
    return pd.qcut(ranked_feature, 4, labels=QUARTILE_LABELS)


def select_solar_feature(frame: pd.DataFrame) -> str:
    scores = {}
    for feature_name in SOLAR_FEATURES:
        quartiles = assign_quartiles(frame, feature_name)
        actual_by_quartile = frame.groupby(quartiles, observed=True)["actual_spread"].mean()
        scores[feature_name] = float(
            actual_by_quartile.loc[QUARTILE_LABELS[-1]]
            - actual_by_quartile.loc[QUARTILE_LABELS[0]]
        )
    return max(scores, key=scores.get)


def feature_label(feature_name: str) -> str:
    return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " "))


def save_both(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare actual, MSE-predicted, and SPO+-predicted daily spreads "
            "by quartiles of mean or max solar irradiance."
        ),
    )
    parser.add_argument("--mse-root", type=Path, default=DEFAULT_MSE_ROOT)
    parser.add_argument("--spo-root", type=Path, default=DEFAULT_SPO_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--solar-feature",
        choices=SOLAR_FEATURES,
        default=None,
        help=(
            "Solar feature for the x-axis. Defaults to whichever solar feature "
            "has the larger Q4-Q1 actual spread increase."
        ),
    )
    return parser


def main(
    *,
    mse_root: Path = DEFAULT_MSE_ROOT,
    spo_root: Path = DEFAULT_SPO_ROOT,
    outdir: Path = DEFAULT_OUTDIR,
    solar_feature: str | None = None,
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

    frame = split.X_explain.reset_index(drop=True).copy()
    frame.insert(0, "date", pd.to_datetime(split.explain_dates).reset_index(drop=True))
    actual_prices = split.y_explain.to_numpy(dtype=float)
    frame["actual_spread"] = actual_prices.max(axis=1) - actual_prices.min(axis=1)
    frame["mse_predicted_spread"] = predicted_spread(mse_model, split.X_explain)
    frame["spo_predicted_spread"] = predicted_spread(spo_model, split.X_explain)

    solar_feature = solar_feature or select_solar_feature(frame)
    correlation_rows = []
    for feature_name in SOLAR_FEATURES:
        correlation_rows.append(
            {
                "feature": feature_name,
                "correlation_with_spo_predicted_spread": pearson_correlation(
                    frame[feature_name],
                    frame["spo_predicted_spread"],
                ),
                "correlation_with_actual_spread": pearson_correlation(
                    frame[feature_name],
                    frame["actual_spread"],
                ),
                "correlation_with_mse_predicted_spread": pearson_correlation(
                    frame[feature_name],
                    frame["mse_predicted_spread"],
                ),
            }
        )
    correlation_frame = pd.DataFrame(correlation_rows)

    frame["solar_feature"] = frame[solar_feature]
    frame["solar_feature_quartile"] = assign_quartiles(frame, solar_feature)

    quartile_summary = (
        frame.groupby("solar_feature_quartile", observed=True)
        .agg(
            solar_feature_min=("solar_feature", "min"),
            solar_feature_max=("solar_feature", "max"),
            actual_spread=("actual_spread", "mean"),
            mse_predicted_spread=("mse_predicted_spread", "mean"),
            spo_predicted_spread=("spo_predicted_spread", "mean"),
        )
        .reset_index()
    )
    quartile_summary.insert(0, "solar_feature", solar_feature)

    cmap = plt.get_cmap("cmc.batlow")
    colors = {
        "actual": cmap(0.36),
        "mse": cmap(0.15),
        "spo": cmap(0.70),
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

    x = np.arange(len(quartile_summary))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for column, label, color, marker in (
        ("actual_spread", "Actual spread", colors["actual"], "o"),
        ("mse_predicted_spread", "MSE predicted spread", colors["mse"], "s"),
        ("spo_predicted_spread", "SPO+ predicted spread", colors["spo"], "s"),
    ):
        ax.plot(
            x,
            quartile_summary[column],
            marker=marker,
            linewidth=2.2,
            color=color,
            label=label,
        )
    selected_correlations = correlation_frame.set_index("feature").loc[solar_feature]
    actual_correlation = float(
        selected_correlations["correlation_with_actual_spread"]
    )
    spo_correlation = float(
        selected_correlations["correlation_with_spo_predicted_spread"]
    )
    ax.set_xticks(x)
    ax.set_xticklabels(QUARTILE_LABELS)
    ax.set_xlabel(f"{feature_label(solar_feature)} quartile")
    ax.set_ylabel("Daily price spread")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
    )
    ax.text(
        0.02,
        0.04,
        f"corr(solar, actual spread) = {actual_correlation:.3f}\n"
        f"corr(solar, SPO+ spread) = {spo_correlation:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.86)
    save_both(fig, outdir, "predicted_spread_by_solar_quartile")

    frame.to_csv(outdir / "predicted_spread_by_day.csv", index=False)
    correlation_frame.to_csv(
        outdir / "solar_spread_feature_correlations.csv",
        index=False,
    )
    quartile_summary.to_csv(
        outdir / "predicted_spread_by_solar_quartile.csv",
        index=False,
    )

    print(f"Selected solar feature: {solar_feature}")
    print(f"Actual spread correlation: {actual_correlation:.6f}")
    print(f"SPO+ predicted spread correlation: {spo_correlation:.6f}")
    print(f"Wrote {outdir / 'predicted_spread_by_solar_quartile.png'}")
    print(f"Wrote {outdir / 'predicted_spread_by_solar_quartile.pdf'}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(
        mse_root=args.mse_root,
        spo_root=args.spo_root,
        outdir=args.outdir,
        solar_feature=args.solar_feature,
    )
