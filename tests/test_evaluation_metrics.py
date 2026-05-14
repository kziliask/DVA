from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dva.analysis.caiso_sweep import (
    compare_caiso_shap_sweeps,
    write_caiso_sweep_comparison_outputs,
)
from dva.analysis.evaluation_metrics import (
    build_metric_summary,
    compute_decision_activation_metrics,
    compute_decision_deletion_auc,
    compute_decision_insertion_auc,
    compute_exact_decision_infidelity,
    compute_kendall_tau_correlation,
    compute_normalized_importance_l1,
    compute_rank_kendall_tau_from_rankings,
    compute_top_k_jaccard,
    compute_truncated_rbo,
    identify_invariant_policy_days,
)


class DecisionMetricUtilityTests(unittest.TestCase):
    def test_decision_activation_metrics_match_known_two_feature_game(self) -> None:
        metrics = compute_decision_activation_metrics(
            coalition_values=[0.0, 1.0, 0.0, 3.0],
            decisions=("same", "feature_0", "same", "both"),
            feature_count=2,
            decision_changed=lambda left, right: left != right,
        )

        np.testing.assert_allclose(metrics.activation_rate, [1.0, 0.5])
        np.testing.assert_allclose(metrics.activated_value_sum, [2.0, 1.0])
        np.testing.assert_allclose(metrics.activated_value, [2.0, 2.0])

    def test_decision_activation_value_is_zero_without_activation(self) -> None:
        metrics = compute_decision_activation_metrics(
            coalition_values=[0.0, 1.0, 2.0, 3.0],
            decisions=("same", "same", "same", "same"),
            feature_count=2,
            decision_changed=lambda left, right: left != right,
        )

        np.testing.assert_allclose(metrics.activation_rate, [0.0, 0.0])
        np.testing.assert_allclose(metrics.activated_value_sum, [0.0, 0.0])
        np.testing.assert_allclose(metrics.activated_value, [0.0, 0.0])

    def test_decision_insertion_auc_matches_known_additive_path(self) -> None:
        feature_names = ("f0", "f1", "f2")
        coalition_values = [0.0, 3.0, 2.0, 5.0, 1.0, 4.0, 3.0, 6.0]
        attributions = [3.0, 2.0, 1.0]

        insertion_auc = compute_decision_insertion_auc(
            attributions,
            coalition_values,
            feature_names,
        )

        assert insertion_auc is not None
        self.assertAlmostEqual(insertion_auc, 7.0 / 12.0, places=8)

    def test_decision_deletion_auc_matches_known_additive_path(self) -> None:
        feature_names = ("f0", "f1", "f2")
        coalition_values = [0.0, 3.0, 2.0, 5.0, 1.0, 4.0, 3.0, 6.0]
        attributions = [3.0, 2.0, 1.0]

        deletion_auc = compute_decision_deletion_auc(
            attributions,
            coalition_values,
            feature_names,
        )

        assert deletion_auc is not None
        self.assertAlmostEqual(deletion_auc, 5.0 / 12.0, places=8)

    def test_decision_insertion_auc_undefined_for_nonpositive_full_gain_and_reports_coverage(
        self,
    ) -> None:
        feature_names = ("f0", "f1")
        coalition_values = [0.0, -1.0, -2.0, -3.0]

        insertion_auc = compute_decision_insertion_auc(
            [1.0, 0.5],
            coalition_values,
            feature_names,
        )
        summary = build_metric_summary(
            {
                "2025-01-01": insertion_auc,
                "2025-01-02": 0.5,
            }
        )

        self.assertIsNone(insertion_auc)
        self.assertIsNone(
            compute_decision_deletion_auc(
                [1.0, 0.5],
                coalition_values,
                feature_names,
            )
        )
        self.assertAlmostEqual(summary["coverage"], 0.5)
        self.assertEqual(summary["valid_days"], 1)
        self.assertEqual(summary["total_days"], 2)

    def test_decision_insertion_auc_remains_defined_for_small_positive_full_gain(
        self,
    ) -> None:
        feature_names = ("f0", "f1")
        coalition_values = [0.0, 0.06, 0.04, 0.08]

        insertion_auc = compute_decision_insertion_auc(
            [0.06, 0.02],
            coalition_values,
            feature_names,
        )

        assert insertion_auc is not None
        self.assertAlmostEqual(insertion_auc, (0.0 + 0.06 / 0.08 + 1.0) / 3.0)

    def test_decision_insertion_auc_returns_zero_when_all_strict_prefixes_are_nonpositive(
        self,
    ) -> None:
        feature_names = ("f0", "f1")
        coalition_values = [0.0, -2.0, 1.0, 1.0]

        insertion_auc = compute_decision_insertion_auc(
            [2.0, 0.5],
            coalition_values,
            feature_names,
        )

        assert insertion_auc is not None
        self.assertAlmostEqual(insertion_auc, 0.0)

    def test_decision_deletion_auc_returns_zero_when_all_strict_suffixes_are_nonpositive(
        self,
    ) -> None:
        feature_names = ("f0", "f1", "f2")
        coalition_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0]

        deletion_auc = compute_decision_deletion_auc(
            [3.0, 2.0, 1.0],
            coalition_values,
            feature_names,
        )

        assert deletion_auc is not None
        self.assertAlmostEqual(deletion_auc, 0.0)

    def test_exact_decision_infidelity_is_zero_for_additive_game(self) -> None:
        feature_names = ("f0", "f1", "f2")
        coalition_values = [0.0, 3.0, 2.0, 5.0, 1.0, 4.0, 3.0, 6.0]
        attributions = [3.0, 2.0, 1.0]

        infidelity = compute_exact_decision_infidelity(
            attributions,
            coalition_values,
            feature_names,
        )

        self.assertAlmostEqual(infidelity, 0.0, places=10)

    def test_rank_metrics_cover_identical_partial_and_disjoint_top_k_cases(self) -> None:
        left_ranking = ("a", "b", "c", "d", "e", "f")
        identical_ranking = ("a", "b", "c", "d", "e", "f")
        partial_ranking = ("a", "c", "b", "d", "e", "f")
        disjoint_top_k_ranking = ("c", "d", "a", "b", "e", "f")

        self.assertEqual(
            compute_top_k_jaccard(left_ranking, identical_ranking, k=2),
            1.0,
        )
        partial_jaccard = compute_top_k_jaccard(left_ranking, partial_ranking, k=2)
        assert partial_jaccard is not None
        self.assertAlmostEqual(
            partial_jaccard,
            1.0 / 3.0,
        )
        self.assertEqual(
            compute_top_k_jaccard(left_ranking, disjoint_top_k_ranking, k=2),
            0.0,
        )
        identical_rbo = compute_truncated_rbo(left_ranking, identical_ranking, depth=6)
        partial_rbo = compute_truncated_rbo(left_ranking, partial_ranking, depth=6)
        disjoint_rbo = compute_truncated_rbo(
            left_ranking,
            disjoint_top_k_ranking,
            depth=6,
        )
        assert identical_rbo is not None
        assert partial_rbo is not None
        assert disjoint_rbo is not None
        self.assertAlmostEqual(
            identical_rbo,
            1.0,
        )
        self.assertLess(partial_rbo, 1.0)
        self.assertLess(disjoint_rbo, partial_rbo)

    def test_kendall_tau_rank_metrics_cover_identical_partial_and_reversed_cases(self) -> None:
        left_ranking = ("a", "b", "c", "d")
        identical_ranking = ("a", "b", "c", "d")
        partial_ranking = ("a", "c", "b", "d")
        reversed_ranking = ("d", "c", "b", "a")

        self.assertEqual(
            compute_rank_kendall_tau_from_rankings(left_ranking, identical_ranking),
            1.0,
        )
        partial_tau = compute_rank_kendall_tau_from_rankings(
            left_ranking,
            partial_ranking,
        )
        assert partial_tau is not None
        self.assertAlmostEqual(
            partial_tau,
            2.0 / 3.0,
        )
        self.assertEqual(
            compute_rank_kendall_tau_from_rankings(left_ranking, reversed_ranking),
            -1.0,
        )
        self.assertIsNone(
            compute_kendall_tau_correlation(
                [1.0, 1.0, 1.0],
                [1.0, 2.0, 3.0],
            )
        )

    def test_kendall_tau_correlation_uses_tau_b_tie_correction(self) -> None:
        tau = compute_kendall_tau_correlation(
            [1.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 2.0, 3.0],
        )

        assert tau is not None
        self.assertAlmostEqual(tau, 0.8)

    def test_normalized_importance_l1_detects_magnitude_change_even_when_ranking_matches(
        self,
    ) -> None:
        feature_names = ("f0", "f1", "f2")
        left_scores = {"f0": 9.0, "f1": 3.0, "f2": 1.0}
        right_scores = {"f0": 6.0, "f1": 2.0, "f2": 1.0}

        l1_distance = compute_normalized_importance_l1(
            left_scores,
            right_scores,
            feature_names,
        )

        assert l1_distance is not None
        self.assertGreater(l1_distance, 0.0)

    def test_identify_invariant_policy_days_respects_mode_and_flow_tolerance(self) -> None:
        left = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "hour": 1,
                    "charge": 1.0,
                    "discharge": 0.0,
                    "mode": 1,
                },
                {
                    "date": "2025-01-02",
                    "hour": 1,
                    "charge": 0.0,
                    "discharge": 1.0,
                    "mode": 0,
                },
            ]
        )
        right = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "hour": 1,
                    "charge": 1.0 + 1e-7,
                    "discharge": 0.0,
                    "mode": 1,
                },
                {
                    "date": "2025-01-02",
                    "hour": 1,
                    "charge": 0.0,
                    "discharge": 0.999,
                    "mode": 0,
                },
            ]
        )

        invariant_dates = identify_invariant_policy_days(left, right)

        self.assertEqual(invariant_dates, ["2025-01-01"])


class SweepComparisonTests(unittest.TestCase):
    def test_synthetic_manifest_produces_pairwise_summary_and_trend_outputs(self) -> None:
        feature_names = ("f0", "f1", "f2")
        dates = ["2025-01-01", "2025-01-02", "2025-01-03"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            setting_dirs = {
                "s0": tmp_path / "s0",
                "s1": tmp_path / "s1",
                "s2": tmp_path / "s2",
            }
            self._write_setting_outputs(
                setting_dirs["s0"],
                feature_names=feature_names,
                dates=dates,
                predictive_rows=[(3.0, 2.0, 1.0)] * 3,
                decision_rows=[(2.0, 1.0, 0.5)] * 3,
                evaluation_metrics=self._evaluation_metrics_payload(
                    predictive_auc=0.2,
                    predictive_infidelity=3.0,
                    decision_auc=0.4,
                    decision_infidelity=2.5,
                    dates=dates,
                ),
                policy_rows=[
                    ("2025-01-01", 1.0, 0.0, 1),
                    ("2025-01-02", 0.0, 1.0, 0),
                    ("2025-01-03", 1.0, 0.0, 1),
                ],
            )
            self._write_setting_outputs(
                setting_dirs["s1"],
                feature_names=feature_names,
                dates=dates,
                predictive_rows=[(2.0, 3.0, 1.0)] * 3,
                decision_rows=[(1.5, 2.5, 0.5)] * 3,
                evaluation_metrics=self._evaluation_metrics_payload(
                    predictive_auc=0.5,
                    predictive_infidelity=2.0,
                    decision_auc=0.6,
                    decision_infidelity=2.0,
                    dates=dates,
                ),
                policy_rows=[
                    ("2025-01-01", 1.0, 0.0, 1),
                    ("2025-01-02", 0.0, 1.0, 0),
                    ("2025-01-03", 0.0, 0.0, 1),
                ],
            )
            self._write_setting_outputs(
                setting_dirs["s2"],
                feature_names=feature_names,
                dates=dates,
                predictive_rows=[(1.0, 4.0, 2.0)] * 3,
                decision_rows=[(1.0, 3.0, 2.0)] * 3,
                evaluation_metrics=self._evaluation_metrics_payload(
                    predictive_auc=0.8,
                    predictive_infidelity=1.0,
                    decision_auc=0.7,
                    decision_infidelity=1.5,
                    dates=dates,
                ),
                policy_rows=[
                    ("2025-01-01", 1.0, 0.0, 1),
                    ("2025-01-02", 0.5, 0.5, 1),
                    ("2025-01-03", 0.0, 0.0, 0),
                ],
            )

            manifest_path = tmp_path / "manifest.csv"
            manifest = pd.DataFrame(
                [
                    {
                        "setting_id": "s0",
                        "results_dir": str(setting_dirs["s0"]),
                        "sweep_id": "capacity",
                        "parameter_name": "energy_capacity",
                        "parameter_value": 1.0,
                        "step_index": 0,
                        "step_label": "1 MWh",
                    },
                    {
                        "setting_id": "s1",
                        "results_dir": str(setting_dirs["s1"]),
                        "sweep_id": "capacity",
                        "parameter_name": "energy_capacity",
                        "parameter_value": 2.0,
                        "step_index": 1,
                        "step_label": "2 MWh",
                    },
                    {
                        "setting_id": "s2",
                        "results_dir": str(setting_dirs["s2"]),
                        "sweep_id": "capacity",
                        "parameter_name": "energy_capacity",
                        "parameter_value": 4.0,
                        "step_index": 2,
                        "step_label": "4 MWh",
                    },
                ]
            )
            manifest.to_csv(manifest_path, index=False)

            outputs = compare_caiso_shap_sweeps(
                manifest_path,
                min_invariant_days=3,
            )

            self.assertEqual(len(outputs.pairwise_rank_metrics), 4)
            self.assertEqual(len(outputs.setting_metric_summary), 6)
            predictive_pair = outputs.pairwise_rank_metrics[
                outputs.pairwise_rank_metrics["explainer_family"] == "predictive"
            ].reset_index(drop=True)
            self.assertTrue(predictive_pair["normalized_importance_l1"].gt(0).all())
            self.assertTrue(predictive_pair["rank_kendall_tau"].between(-1.0, 1.0).all())
            self.assertTrue(predictive_pair["decision_invariant_rbo_10"].isna().all())
            self.assertTrue(
                predictive_pair["decision_invariant_normalized_importance_l1"].isna().all()
            )
            self.assertEqual(
                outputs.sweep_trend_metrics["capacity"]["predictive"][
                    "decision_insertion_auc_spearman"
                ]["spearman"],
                1.0,
            )
            self.assertEqual(
                outputs.sweep_trend_metrics["capacity"]["predictive"][
                    "decision_infidelity_spearman"
                ]["spearman"],
                -1.0,
            )
            self.assertEqual(
                outputs.sweep_trend_metrics["capacity"]["predictive"][
                    "decision_insertion_auc_kendall_tau"
                ]["kendall_tau"],
                1.0,
            )
            self.assertEqual(
                outputs.sweep_trend_metrics["capacity"]["predictive"][
                    "decision_deletion_auc_spearman"
                ]["spearman"],
                -1.0,
            )
            self.assertEqual(
                outputs.sweep_trend_metrics["capacity"]["predictive"][
                    "decision_deletion_auc_kendall_tau"
                ]["kendall_tau"],
                -1.0,
            )
            self.assertEqual(
                outputs.sweep_trend_metrics["capacity"]["predictive"][
                    "decision_infidelity_kendall_tau"
                ]["kendall_tau"],
                -1.0,
            )

            outdir = tmp_path / "comparison_outputs"
            write_caiso_sweep_comparison_outputs(outputs, outdir)
            self.assertTrue((outdir / "pairwise_rank_metrics.csv").exists())
            self.assertTrue((outdir / "pairwise_rank_metrics.json").exists())
            self.assertTrue((outdir / "setting_metric_summary.csv").exists())
            self.assertTrue((outdir / "sweep_trend_metrics.json").exists())

    def _write_setting_outputs(
        self,
        results_dir: Path,
        *,
        feature_names: tuple[str, ...],
        dates: list[str],
        predictive_rows: list[tuple[float, ...]],
        decision_rows: list[tuple[float, ...]],
        evaluation_metrics: dict[str, object],
        policy_rows: list[tuple[str, float, float, int]],
    ) -> None:
        results_dir.mkdir(parents=True, exist_ok=True)
        daily_rows: list[dict[str, str | float]] = []
        for date, predictive_values, decision_values in zip(
            dates,
            predictive_rows,
            decision_rows,
            strict=True,
        ):
            row: dict[str, str | float] = {"date": date}
            for feature_name, value in zip(feature_names, predictive_values, strict=True):
                row[f"predictive_shap_{feature_name}"] = value
            for feature_name, value in zip(feature_names, decision_values, strict=True):
                row[f"decision_shap_{feature_name}"] = value
            daily_rows.append(row)
        pd.DataFrame(daily_rows).to_csv(results_dir / "daily_shap.csv", index=False)

        summary_rows = []
        for rank, feature_name in enumerate(feature_names, start=1):
            summary_rows.append(
                {
                    "feature": feature_name,
                    "predictive_mean_abs_shap": rank,
                    "predictive_rank": rank,
                    "decision_mean_abs_shap": rank,
                    "decision_rank": rank,
                }
            )
        pd.DataFrame(summary_rows).to_csv(results_dir / "summary_shap.csv", index=False)

        dispatch_rows = [
            {
                "date": date,
                "hour": 1,
                "charge": charge,
                "discharge": discharge,
                "mode": mode,
                "state_of_charge_start": 0.0,
                "state_of_charge_end": 0.0,
            }
            for date, charge, discharge, mode in policy_rows
        ]
        pd.DataFrame(dispatch_rows).to_csv(
            results_dir / "daily_full_dispatch.csv",
            index=False,
        )

        with (results_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(evaluation_metrics, handle, indent=2, sort_keys=True)
        with (results_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump({"feature_columns": list(feature_names)}, handle, indent=2, sort_keys=True)

    def _evaluation_metrics_payload(
        self,
        *,
        predictive_auc: float,
        predictive_infidelity: float,
        decision_auc: float,
        decision_infidelity: float,
        dates: list[str],
    ) -> dict[str, object]:
        return {
            "predictive": {
                "decision_deletion_auc": self._metric_summary(
                    1.0 - predictive_auc,
                    coverage=1.0,
                    dates=dates,
                ),
                "decision_insertion_auc": self._metric_summary(
                    predictive_auc,
                    coverage=1.0,
                    dates=dates,
                ),
                "decision_infidelity": self._metric_summary(
                    predictive_infidelity,
                    coverage=1.0,
                    dates=dates,
                ),
            },
            "decision": {
                "decision_deletion_auc": self._metric_summary(
                    1.0 - decision_auc,
                    coverage=1.0,
                    dates=dates,
                ),
                "decision_insertion_auc": self._metric_summary(
                    decision_auc,
                    coverage=1.0,
                    dates=dates,
                ),
                "decision_infidelity": self._metric_summary(
                    decision_infidelity,
                    coverage=1.0,
                    dates=dates,
                ),
            },
        }

    def _metric_summary(
        self,
        value: float,
        *,
        coverage: float,
        dates: list[str],
    ) -> dict[str, object]:
        return {
            "mean": value,
            "median": value,
            "std": 0.0,
            "valid_days": int(round(len(dates) * coverage)),
            "total_days": len(dates),
            "coverage": coverage,
            "values_by_date": {date: value for date in dates},
        }


if __name__ == "__main__":
    unittest.main()
