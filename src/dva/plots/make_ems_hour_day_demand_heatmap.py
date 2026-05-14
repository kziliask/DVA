from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cmcrameri  # noqa: F401 - registers cmc.* colormaps with Matplotlib.
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET_PATH = Path(
    "data/ems_data/processed/ems_zip_hour_features_2025_manhattan.csv"
)
DEFAULT_OUTDIR = Path("data/plots/ems_hour_day_demand_heatmap")
DEFAULT_CMAP = "cmc.lajolla"
DAY_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
HOURS = tuple(range(24))


@dataclass(frozen=True, slots=True)
class HeatmapOutputs:
    png: Path
    pdf: Path
    csv: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an EMS demand heatmap by hour of day and day of week from "
            "the processed Manhattan ZIP-hour feature table."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Processed EMS ZIP-hour feature CSV.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--cmap",
        default=DEFAULT_CMAP,
        help="Registered Matplotlib colormap name; cmcrameri maps use cmc.* names.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = write_ems_hour_day_demand_heatmap(
        dataset_path=args.dataset,
        outdir=args.outdir,
        cmap_name=args.cmap,
    )
    print(f"Wrote EMS demand heatmap PNG to {outputs.png}")
    print(f"Wrote EMS demand heatmap PDF to {outputs.pdf}")
    print(f"Wrote EMS demand heatmap table to {outputs.csv}")


def write_ems_hour_day_demand_heatmap(
    *,
    dataset_path: Path,
    outdir: Path,
    cmap_name: str = DEFAULT_CMAP,
) -> HeatmapOutputs:
    outdir.mkdir(parents=True, exist_ok=True)
    demand = load_hourly_demand(dataset_path)
    summary = summarize_hour_day_demand(demand)

    summary_path = outdir / "ems_hour_day_demand_heatmap.csv"
    summary.to_csv(summary_path, index=False)

    png_path = outdir / "ems_hour_day_demand_heatmap.png"
    pdf_path = outdir / "ems_hour_day_demand_heatmap.pdf"
    plot_hour_day_demand_heatmap(
        summary=summary,
        output_paths=(png_path, pdf_path),
        cmap_name=cmap_name,
    )
    return HeatmapOutputs(png=png_path, pdf=pdf_path, csv=summary_path)


def load_hourly_demand(dataset_path: Path) -> pd.DataFrame:
    required_columns = {
        "timestamp_hour",
        "hour",
        "day_of_week",
        "ems_incident_count",
    }
    frame = pd.read_csv(
        dataset_path,
        usecols=sorted(required_columns),
        parse_dates=["timestamp_hour"],
    )
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise KeyError(
            f"{dataset_path} is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    hourly = (
        frame.groupby("timestamp_hour", as_index=False)
        .agg(
            hour=("hour", "first"),
            day_of_week=("day_of_week", "first"),
            ems_incidents=("ems_incident_count", "sum"),
        )
        .sort_values("timestamp_hour")
    )
    hourly["hour"] = hourly["hour"].astype(int)
    hourly["day_of_week"] = hourly["day_of_week"].astype(int)
    return hourly


def summarize_hour_day_demand(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"hour", "day_of_week", "ems_incidents"}
    missing_columns = required_columns - set(hourly_demand.columns)
    if missing_columns:
        raise KeyError(
            "Hourly demand frame is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    summary = (
        hourly_demand.groupby(["day_of_week", "hour"], as_index=False)
        .agg(
            mean_hourly_demand=("ems_incidents", "mean"),
            median_hourly_demand=("ems_incidents", "median"),
            total_demand=("ems_incidents", "sum"),
            n_hours=("ems_incidents", "size"),
        )
        .sort_values(["day_of_week", "hour"])
    )
    complete_grid = pd.MultiIndex.from_product(
        [range(7), HOURS],
        names=["day_of_week", "hour"],
    ).to_frame(index=False)
    summary = complete_grid.merge(summary, on=["day_of_week", "hour"], how="left")
    if summary["mean_hourly_demand"].isna().any():
        missing = summary.loc[summary["mean_hourly_demand"].isna(), ["day_of_week", "hour"]]
        raise ValueError(
            "EMS demand data do not cover every day/hour cell. Missing cells: "
            + ", ".join(
                f"{DAY_LABELS[int(row.day_of_week)]} {int(row.hour):02d}:00"
                for row in missing.itertuples(index=False)
            )
        )
    summary["day_label"] = summary["day_of_week"].map(
        {idx: label for idx, label in enumerate(DAY_LABELS)}
    )
    total = float(summary["total_demand"].sum())
    summary["share_of_total_demand"] = summary["total_demand"] / total
    return summary


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
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
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


def plot_hour_day_demand_heatmap(
    *,
    summary: pd.DataFrame,
    output_paths: tuple[Path, Path],
    cmap_name: str,
) -> None:
    heatmap = (
        summary.pivot(index="day_of_week", columns="hour", values="mean_hourly_demand")
        .reindex(index=range(7), columns=HOURS)
        .to_numpy(dtype=float)
    )
    if not np.isfinite(heatmap).all():
        raise ValueError("Heatmap contains non-finite values.")

    apply_plot_style()

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    image = ax.imshow(
        heatmap,
        aspect="auto",
        cmap=plt.get_cmap(cmap_name),
        interpolation="nearest",
    )

    ax.set_title("EMS demand by hour of day and day of week")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week")
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 23])
    ax.set_xticklabels(["00", "03", "06", "09", "12", "15", "18", "21", "23"])
    ax.set_yticks(range(7))
    ax.set_yticklabels(DAY_LABELS)
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(image, ax=ax, pad=0.018, fraction=0.046)
    colorbar.set_label("Mean EMS incidents per hour")

    fig.subplots_adjust(left=0.08, right=0.96, top=0.89, bottom=0.15)
    for output_path in output_paths:
        save_kwargs = {"bbox_inches": "tight"}
        if output_path.suffix.lower() == ".png":
            save_kwargs["dpi"] = 320
        fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
