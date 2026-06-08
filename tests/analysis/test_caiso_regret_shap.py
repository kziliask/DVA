from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dva.analysis.caiso_regret_shap import (
    CaisoRegretShapCaseStudyConfig,
    resolve_regret_model_name,
    run_caiso_regret_shap_case_study,
    write_caiso_regret_shap_case_study_outputs,
)
from dva.plots.compare_pred_dec import create_comparison_plots


class CaisoRegretShapCaseStudyTests(unittest.TestCase):
    def test_one_day_regret_case_study_produces_standard_shap_outputs(self) -> None:
        outputs = run_caiso_regret_shap_case_study(
            CaisoRegretShapCaseStudyConfig(
                model_name="xgb",
                holdout_days=360,
                max_train_days=3,
                max_days=1,
            )
        )

        self.assertEqual(len(outputs.daily_shap), 1)
        self.assertEqual(len(outputs.regret_feature_shap), 8)
        self.assertEqual(len(outputs.daily_full_dispatch), 24)
        self.assertEqual(len(outputs.summary_shap), 8)
        self.assertEqual(outputs.run_metadata["experiment_type"], "caiso_regret_predictor_shap")
        self.assertEqual(outputs.run_metadata["base_model_name"], "xgb")
        self.assertEqual(outputs.run_metadata["regret_model_name"], "xgb")
        self.assertEqual(outputs.run_metadata["regret_target_column"], "actual_daily_regret")
        self.assertEqual(
            outputs.run_metadata["predictive_shap_family"],
            "first_stage_price_model_prediction",
        )
        self.assertEqual(
            outputs.run_metadata["regret_predictive_shap_family"],
            "second_stage_regret_model_prediction",
        )
        self.assertEqual(
            outputs.run_metadata["decision_shap_family"],
            "first_stage_price_model_decision_value",
        )
        self.assertEqual(outputs.prediction_metrics["holdout"]["days"], 1)
        self.assertEqual(outputs.prediction_metrics["base_price_model_holdout"]["targets_per_day"], 24)

        daily_row = outputs.daily_shap.iloc[0]
        predictive_shap_sum = sum(
            daily_row[f"predictive_shap_{feature_name}"]
            for feature_name in outputs.run_metadata["feature_columns"]
        )
        decision_shap_sum = sum(
            daily_row[f"decision_shap_{feature_name}"]
            for feature_name in outputs.run_metadata["feature_columns"]
        )
        regret_predictive_shap_sum = sum(
            daily_row[f"regret_predictive_shap_{feature_name}"]
            for feature_name in outputs.run_metadata["feature_columns"]
        )

        self.assertAlmostEqual(
            predictive_shap_sum,
            daily_row["predictive_total_gain"],
            places=6,
        )
        self.assertAlmostEqual(
            decision_shap_sum,
            daily_row["decision_value_gain"],
            places=6,
        )
        self.assertAlmostEqual(
            regret_predictive_shap_sum,
            daily_row["regret_predictive_value_gain"],
            places=6,
        )
        self.assertAlmostEqual(
            daily_row["actual_daily_regret"],
            daily_row["oracle_obj"] - daily_row["decision_full_value"],
            places=6,
        )
        self.assertAlmostEqual(
            daily_row["baseline_daily_regret"],
            daily_row["oracle_obj"] - daily_row["decision_baseline_value"],
            places=6,
        )
        self.assertAlmostEqual(
            daily_row["decision_value_gain"],
            daily_row["decision_full_value"] - daily_row["decision_baseline_value"],
            places=6,
        )

        self.assertEqual(resolve_regret_model_name("xgb"), "xgb")
        with self.assertRaisesRegex(ValueError, "Only model_name='xgb'"):
            resolve_regret_model_name("mlp")

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "results"
            plot_outdir = Path(tmpdir) / "plots"
            write_caiso_regret_shap_case_study_outputs(outputs, outdir)
            self.assertTrue((outdir / "daily_shap.csv").exists())
            self.assertTrue((outdir / "regret_predictions.csv").exists())
            with (outdir / "run_metadata.json").open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["regret_model_name"], "xgb")

            plot_paths = create_comparison_plots(
                daily_shap_path=outdir / "daily_shap.csv",
                outdir=plot_outdir,
            )
            self.assertTrue(plot_paths)
            self.assertTrue((plot_outdir / "predictive_beeswarm.png").exists())
            self.assertTrue((plot_outdir / "regret_predictive_beeswarm.png").exists())
            written_daily_shap = pd.read_csv(outdir / "daily_shap.csv")
            self.assertIn("predicted_daily_regret", written_daily_shap.columns)
            self.assertIn("regret_predictive_shap_min_temp_c", written_daily_shap.columns)


if __name__ == "__main__":
    unittest.main()
