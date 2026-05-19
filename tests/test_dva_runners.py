from __future__ import annotations

import sys
from pathlib import Path

from dva.case_studies.caiso.runners import gdsi as caiso_gdsi
from dva.case_studies.caiso.runners import joint_dvi as caiso_joint_dvi
from dva.case_studies.ems.runners import design_joint_dvi as ems_design_joint_dvi
from dva.case_studies.ems.runners import infodva as ems_infodva


def test_caiso_gdsi_runner_forwards_l25_split_and_ante_infodva() -> None:
    args = caiso_gdsi.build_parser().parse_args(
        [
            "--model-id",
            "xgb_001",
            "--model-id",
            "xgb_025",
            "--outdir",
            "results/custom",
        ]
    )

    forwarded = caiso_gdsi.build_forward_args(args, ["--max-days", "2"])

    assert forwarded.count("--model-id") == 2
    assert forwarded[forwarded.index("--train-months") + 1] == "24"
    assert forwarded[forwarded.index("--validation-months") + 1] == "12"
    assert "--test-rest" in forwarded
    assert "--compute-ante-infodva" in forwarded
    assert forwarded[-2:] == ["--max-days", "2"]


def test_caiso_joint_dvi_dry_run_uses_model_id(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dva-caiso-joint-dvi",
            "--model-id",
            "xgb_025",
            "--baseline",
            "optimistic",
            "--value-mode",
            "ante",
            "--dry-run",
        ],
    )

    caiso_joint_dvi.main()

    output = capsys.readouterr().out.strip()
    assert output.startswith("uv run dva-caiso-joint-dvi")
    assert "--model-id xgb_025" in output
    assert "--baseline optimistic" in output
    assert "--value-mode ante" in output
    assert "results/caiso/joint_dvi/xgb_025/optimistic_ante" in output


def test_ems_infodva_runner_forwards_exact_3x3_and_ante_infodva() -> None:
    args = ems_infodva.build_parser().parse_args(["--model-id", "xgb_001"])

    forwarded = ems_infodva.build_forward_args(args, [])

    assert forwarded[forwarded.index("--model-id") + 1] == "xgb_001"
    assert forwarded[forwarded.index("--solver") + 1] == "exact"
    assert forwarded[
        forwarded.index("--coverage-radius-km") + 1 : forwarded.index("--facility-budget")
    ] == [
        "1.0",
        "2.0",
        "3.0",
    ]
    assert forwarded[
        forwarded.index("--facility-budget") + 1 : forwarded.index("--holdout-hours")
    ] == [
        "3",
        "5",
        "8",
    ]
    assert "--compute-ante-infodva" in forwarded
    assert "--no-cvar-decision-shap" in forwarded
    assert "--no-plots" in forwarded


def test_ems_design_joint_dvi_dry_run_command(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dva-ems-design-joint-dvi",
            "--analysis-kind",
            "joint_dvi",
            "--solver",
            "naive",
            "--value-mode",
            "ante",
            "--outdir",
            str(Path("results/ems/joint_dvi/naive_ante")),
            "--dry-run",
        ],
    )

    ems_design_joint_dvi.main()

    output = capsys.readouterr().out.strip()
    assert output.startswith("uv run dva-ems-design-joint-dvi")
    assert "--analysis-kind joint_dvi" in output
    assert "--solver naive" in output
    assert "--value-mode ante" in output
    assert "results/ems/joint_dvi/naive_ante" in output


def test_ems_joint_dvi_value_mode_columns_are_explicit() -> None:
    assert ems_design_joint_dvi._joint_value_column("ante") == "ante_decision_value"
    assert ems_design_joint_dvi._joint_value_column("post") == "decision_value"
