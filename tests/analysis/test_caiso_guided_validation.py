from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from dva.analysis.run_caiso_decision_shap_guided_validation import (
    compute_background_feature_replacements,
    compute_daily_sage_values,
    build_fixed_caiso_guided_validation_split,
    _build_config_for_record,
    _add_ante_infodva_metrics,
    _filter_manifest_by_model_family,
    _load_preserved_results,
    _select_lofo_validation_candidate_summary,
    _select_sage_validation_candidate_summary,
    make_L25_OA_symbols,
    make_model_manifest,
    select_harmful_feature_candidates,
    select_random_ineligible_feature,
    select_sage_disruptive_feature_candidates,
)
from dva.analysis.run_caiso_decision_shap_guided_validation_rolling_windows import (
    build_calendar_rolling_caiso_guided_validation_folds,
)
from dva.model.train import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_TARGET_COLUMNS,
    train_model,
)


def _write_synthetic_caiso_frame(
    path: Path,
    *,
    start: str,
    end: str,
) -> None:
    dates = pd.date_range(start=start, end=end, freq="D")
    frame = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    for feature_idx, feature_name in enumerate(DEFAULT_FEATURE_COLUMNS):
        if feature_name == "day_of_week":
            frame[feature_name] = dates.dayofweek
        else:
            frame[feature_name] = feature_idx + np.arange(len(frame), dtype=float) / 1000.0
    for target_idx, target_name in enumerate(DEFAULT_TARGET_COLUMNS):
        frame[target_name] = target_idx + np.arange(len(frame), dtype=float) / 100.0
    frame.to_csv(path, index=False)


class CaisoGuidedValidationDesignTests(unittest.TestCase):
    def test_l25_oa_has_pairwise_balance(self) -> None:
        oa = make_L25_OA_symbols()

        self.assertEqual(oa.shape, (25, 6))
        for first_col in range(oa.shape[1]):
            for second_col in range(first_col + 1, oa.shape[1]):
                pairs = set(map(tuple, oa[:, [first_col, second_col]]))
                self.assertEqual(len(pairs), 25)

    def test_model_manifest_has_expected_baseline_rows(self) -> None:
        manifest = make_model_manifest()
        rows_by_id = {
            str(record["model_id"]): record
            for record in manifest.to_dict(orient="records")
        }

        self.assertEqual(len(manifest), 50)
        self.assertEqual(rows_by_id["xgb_001"]["model_name"], "xgb")
        self.assertEqual(rows_by_id["xgb_001"]["n_estimators"], 100)
        self.assertEqual(rows_by_id["xgb_001"]["max_depth"], 3)
        self.assertEqual(rows_by_id["xgb_001"]["learning_rate"], 0.05)
        self.assertEqual(rows_by_id["nn_001"]["model_name"], "torch_mlp")
        self.assertEqual(rows_by_id["nn_001"]["hidden_layers"], 2)
        self.assertEqual(rows_by_id["nn_001"]["hidden_width"], 128)
        self.assertEqual(rows_by_id["nn_001"]["dropout"], 0.10)
        self.assertEqual(rows_by_id["nn_001"]["batch_size"], 64)
        self.assertEqual(rows_by_id["nn_001"]["max_epochs"], 300)
        self.assertEqual(rows_by_id["nn_001"]["early_stopping_patience"], 25)

    def test_model_family_filter_selects_only_nn_rows(self) -> None:
        manifest = make_model_manifest()
        nn_manifest = _filter_manifest_by_model_family(manifest, "nn")
        xgb_manifest = _filter_manifest_by_model_family(manifest, "xgb")

        self.assertEqual(len(nn_manifest), 25)
        self.assertEqual(set(nn_manifest["model_name"]), {"torch_mlp"})
        self.assertEqual(len(xgb_manifest), 25)
        self.assertEqual(set(xgb_manifest["model_name"]), {"xgb"})

    def test_preserved_results_exclude_current_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model_results.csv"
            pd.DataFrame(
                [
                    {"model_id": "xgb_001", "status": "completed"},
                    {"model_id": "nn_001", "status": "completed"},
                ]
            ).to_csv(path, index=False)

            preserved = _load_preserved_results(
                path,
                excluded_model_ids={"nn_001"},
            )

        self.assertEqual(preserved, [{"model_id": "xgb_001", "status": "completed"}])

    def test_model_config_accepts_storage_parameter_overrides(self) -> None:
        record = make_model_manifest().query("model_id == 'xgb_001'").iloc[0].to_dict()

        config = _build_config_for_record(
            record,
            dataset_path=Path("data.csv"),
            model_dir=Path("out"),
            background_days=365,
            throughput_penalty=10.0,
            energy_capacity=8.0,
            power_limit=1.0,
            charge_efficiency=0.85,
            discharge_efficiency=0.85,
            initial_state_of_charge=4.0,
            terminal_state_of_charge=4.0,
        )

        self.assertEqual(config.storage_parameters.throughput_penalty, 10.0)
        self.assertEqual(config.storage_parameters.energy_capacity, 8.0)
        self.assertEqual(config.storage_parameters.power_limit, 1.0)
        self.assertEqual(config.storage_parameters.charge_efficiency, 0.85)
        self.assertEqual(config.storage_parameters.discharge_efficiency, 0.85)
        self.assertEqual(config.storage_parameters.initial_state_of_charge, 4.0)
        self.assertEqual(config.storage_parameters.terminal_state_of_charge, 4.0)


class CaisoGuidedValidationSplitTests(unittest.TestCase):
    def test_default_split_matches_expected_dates(self) -> None:
        split = build_fixed_caiso_guided_validation_split()

        self.assertEqual(len(split.train_frame), 1093)
        self.assertEqual(len(split.validation_frame), 71)
        self.assertEqual(len(split.test_frame), 30)
        self.assertEqual(split.train_frame[split.date_column].iloc[0], "2023-01-26")
        self.assertEqual(split.train_frame[split.date_column].iloc[-1], "2026-01-25")
        self.assertEqual(split.validation_frame[split.date_column].iloc[0], "2026-01-26")
        self.assertEqual(split.validation_frame[split.date_column].iloc[-1], "2026-04-07")
        self.assertEqual(split.test_frame[split.date_column].iloc[0], "2026-04-08")
        self.assertEqual(split.test_frame[split.date_column].iloc[-1], "2026-05-07")

    def test_background_replacement_uses_last_training_year_mean(self) -> None:
        split = build_fixed_caiso_guided_validation_split()
        replacements = compute_background_feature_replacements(
            split.background_frame,
            ("mean_wind_speed",),
        )

        expected = float(split.background_frame["mean_wind_speed"].mean())
        self.assertAlmostEqual(
            replacements["mean_wind_speed"],
            expected,
            places=12,
        )

    def test_calendar_split_can_use_train_validation_months_and_rest_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "caiso_synthetic.csv"
            _write_synthetic_caiso_frame(
                dataset_path,
                start="2023-01-26",
                end="2026-05-07",
            )

            split = build_fixed_caiso_guided_validation_split(
                dataset_path,
                train_months=24,
                validation_months=12,
                test_rest=True,
            )

        self.assertEqual(len(split.train_frame), 731)
        self.assertEqual(len(split.validation_frame), 365)
        self.assertEqual(len(split.test_frame), 102)
        self.assertEqual(split.train_frame[split.date_column].iloc[0], "2023-01-26")
        self.assertEqual(split.train_frame[split.date_column].iloc[-1], "2025-01-25")
        self.assertEqual(
            split.validation_frame[split.date_column].iloc[0],
            "2025-01-26",
        )
        self.assertEqual(
            split.validation_frame[split.date_column].iloc[-1],
            "2026-01-25",
        )
        self.assertEqual(split.test_frame[split.date_column].iloc[0], "2026-01-26")
        self.assertEqual(split.test_frame[split.date_column].iloc[-1], "2026-05-07")

    def test_ante_infodva_metric_aliases_use_ead_decision_columns(self) -> None:
        metrics: dict[str, float] = {}
        daily_shap = pd.DataFrame(
            {
                "ead_decision_value_gain": [2.0, 4.0],
                "ead_decision_baseline_value": [10.0, 20.0],
                "ead_decision_full_value": [8.0, 16.0],
                "ead_decision_characteristic_full_value": [2.0, 4.0],
                "ead_decision_shap_a": [1.0, -3.0],
                "ead_decision_shap_b": [1.0, 7.0],
            }
        )
        summary_shap = pd.DataFrame({"ead_decision_mean_abs_shap": [2.0, 4.0]})

        _add_ante_infodva_metrics(metrics, daily_shap, summary_shap)

        self.assertEqual(metrics["mean_ante_infodva_value_gain"], 3.0)
        self.assertEqual(metrics["mean_ante_infodva_baseline_value"], 15.0)
        self.assertEqual(metrics["mean_ante_infodva_full_value"], 12.0)
        self.assertEqual(
            metrics["mean_ante_infodva_characteristic_full_value"],
            3.0,
        )
        self.assertEqual(metrics["mean_abs_ante_infodva_shap"], 3.0)
        self.assertEqual(metrics["mean_signed_ante_infodva_shap"], 1.5)
        self.assertEqual(metrics["max_feature_mean_abs_ante_infodva_shap"], 4.0)


class CaisoRollingGuidedValidationSplitTests(unittest.TestCase):
    def test_default_rolling_folds_use_latest_complete_seasonal_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "caiso_synthetic.csv"
            _write_synthetic_caiso_frame(
                dataset_path,
                start="2023-01-26",
                end="2026-05-07",
            )

            folds = build_calendar_rolling_caiso_guided_validation_folds(
                dataset_path,
                background_days=365,
            )

        self.assertEqual(len(folds), 4)
        self.assertEqual(
            [fold.season for fold in folds],
            ["spring", "summer", "fall", "winter"],
        )
        self.assertEqual(folds[0].train_start, "2023-01-26")
        self.assertEqual(folds[0].train_end, "2025-01-25")
        self.assertEqual(folds[0].validation_start, "2025-01-26")
        self.assertEqual(folds[0].validation_end, "2025-04-25")
        self.assertEqual(folds[0].test_start, "2025-04-26")
        self.assertEqual(folds[0].test_end, "2025-07-25")
        self.assertEqual(folds[3].train_start, "2023-10-26")
        self.assertEqual(folds[3].validation_start, "2025-10-26")
        self.assertEqual(folds[3].test_start, "2026-01-26")
        self.assertEqual(folds[3].test_end, "2026-04-25")
        self.assertGreaterEqual(len(folds[0].split.background_frame), 364)

    def test_rolling_folds_honor_explicit_start_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "caiso_synthetic.csv"
            _write_synthetic_caiso_frame(
                dataset_path,
                start="2023-01-26",
                end="2026-05-07",
            )

            folds = build_calendar_rolling_caiso_guided_validation_folds(
                dataset_path,
                start_date="2023-01-26",
            )

        self.assertEqual(folds[0].train_start, "2023-01-26")
        self.assertEqual(folds[0].validation_start, "2025-01-26")
        self.assertEqual(folds[0].test_start, "2025-04-26")
        self.assertEqual(folds[3].test_start, "2026-01-26")


class CaisoGuidedValidationCandidateTests(unittest.TestCase):
    def test_harmful_feature_candidates_require_negative_ci_and_non_tiny_abs(self) -> None:
        daily_shap = pd.DataFrame(
            {
                "decision_shap_clear_harm": [-1.0, -1.2, -0.9, -1.1, -1.0],
                "decision_shap_tiny_harm": [-0.001] * 5,
                "decision_shap_helpful": [0.5] * 5,
                "decision_shap_uncertain": [-2.0, 1.0, 1.0, -2.0, 1.0],
                "decision_shap_secondary_harm": [-0.25] * 5,
            }
        )

        candidates = select_harmful_feature_candidates(
            daily_shap,
            (
                "clear_harm",
                "tiny_harm",
                "helpful",
                "uncertain",
                "secondary_harm",
            ),
            bootstrap_replicates=200,
            random_state=7,
        )
        rows = {
            str(record["feature"]): record
            for record in candidates.to_dict(orient="records")
        }

        self.assertTrue(rows["clear_harm"]["eligible"])
        self.assertEqual(rows["clear_harm"]["candidate_rank"], 1)
        self.assertTrue(rows["secondary_harm"]["eligible"])
        self.assertEqual(rows["secondary_harm"]["candidate_rank"], 2)
        self.assertFalse(rows["tiny_harm"]["eligible"])
        self.assertEqual(rows["tiny_harm"]["ineligibility_reason_count"], 1)
        self.assertIn(
            "tiny_mean_abs_decision_shap",
            rows["tiny_harm"]["ineligibility_reason"],
        )
        self.assertFalse(rows["helpful"]["eligible"])
        self.assertEqual(rows["helpful"]["ineligibility_reason_count"], 2)
        self.assertIn(
            "non_negative_mean_decision_shap",
            rows["helpful"]["ineligibility_reason"],
        )
        self.assertFalse(rows["uncertain"]["eligible"])
        self.assertIn(
            "bootstrap_upper_ci_not_negative",
            rows["uncertain"]["ineligibility_reason"],
        )

    def test_no_candidate_when_all_means_are_non_negative(self) -> None:
        daily_shap = pd.DataFrame(
            {
                "decision_shap_a": [0.1, 0.2, 0.3],
                "decision_shap_b": [0.0, 0.1, 0.2],
            }
        )

        candidates = select_harmful_feature_candidates(
            daily_shap,
            ("a", "b"),
            bootstrap_replicates=20,
            random_state=3,
        )

        self.assertEqual(int(candidates["eligible"].sum()), 0)

    def test_random_ineligible_feature_selection_is_reproducible(self) -> None:
        candidates = pd.DataFrame(
            {
                "feature": [
                    "eligible",
                    "b_priority_ineligible",
                    "a_priority_ineligible",
                    "single_reason_ineligible",
                ],
                "eligible": [True, False, False, False],
                "ineligibility_reason": [
                    "",
                    "reason_b1;reason_b2",
                    "reason_a1;reason_a2",
                    "reason_single",
                ],
                "ineligibility_reason_count": [0, 2, 2, 1],
            }
        )

        first = select_random_ineligible_feature(
            candidates,
            random_mask_seed=20260510,
            model_id="xgb_001",
        )
        second = select_random_ineligible_feature(
            candidates.sample(frac=1.0, random_state=99),
            random_mask_seed=20260510,
            model_id="xgb_001",
        )
        different_model = select_random_ineligible_feature(
            candidates,
            random_mask_seed=20260510,
            model_id="xgb_002",
        )

        self.assertEqual(first, second)
        self.assertIn(first, {"a_priority_ineligible", "b_priority_ineligible"})
        self.assertIn(
            different_model,
            {"a_priority_ineligible", "b_priority_ineligible"},
        )
        self.assertNotEqual(first, "single_reason_ineligible")

    def test_random_ineligible_feature_counts_reasons_when_column_is_absent(self) -> None:
        candidates = pd.DataFrame(
            {
                "feature": ["one_reason", "two_reasons"],
                "eligible": [False, False],
                "ineligibility_reason": ["reason_one", "reason_a;reason_b"],
            }
        )

        selected = select_random_ineligible_feature(
            candidates,
            random_mask_seed=20260510,
            model_id="xgb_001",
        )

        self.assertEqual(selected, "two_reasons")

    def test_random_ineligible_feature_is_none_when_all_features_are_eligible(self) -> None:
        candidates = pd.DataFrame(
            {
                "feature": ["a", "b"],
                "eligible": [True, True],
            }
        )

        self.assertIsNone(
            select_random_ineligible_feature(
                candidates,
                random_mask_seed=20260510,
                model_id="xgb_001",
            )
        )


class _SumFeatureModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        feature_sum = X.to_numpy(dtype=float).sum(axis=1)
        return np.column_stack((feature_sum, feature_sum))


class CaisoGuidedValidationSageTests(unittest.TestCase):
    def test_daily_sage_uses_loss_null_minus_loss_coalition(self) -> None:
        background = pd.DataFrame(
            {
                "safe": [0.0, 0.0],
                "harmful": [0.0, 0.0],
            }
        )
        explain = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02"],
                "safe": [0.0, 0.0],
                "harmful": [1.0, 2.0],
                "y_1": [0.0, 0.0],
                "y_2": [0.0, 0.0],
            }
        )

        daily_sage = compute_daily_sage_values(
            model=_SumFeatureModel(),
            feature_columns=("safe", "harmful"),
            target_columns=("y_1", "y_2"),
            background_frame=background,
            explain_frame=explain,
            date_column="date",
        )

        np.testing.assert_allclose(daily_sage["sage_loss_null"], [0.0, 0.0])
        np.testing.assert_allclose(daily_sage["sage_loss_full"], [1.0, 4.0])
        np.testing.assert_allclose(daily_sage["sage_value_full"], [-1.0, -4.0])
        np.testing.assert_allclose(daily_sage["sage_shap_safe"], [0.0, 0.0])
        np.testing.assert_allclose(daily_sage["sage_shap_harmful"], [-1.0, -4.0])
        np.testing.assert_allclose(daily_sage["sage_efficiency_gap"], [0.0, 0.0])

    def test_sage_disruptive_candidates_use_negative_upper_ci_and_lowest_mean_rank(
        self,
    ) -> None:
        daily_sage = pd.DataFrame(
            {
                "sage_shap_clear_harm": [-1.0, -1.2, -0.9, -1.1, -1.0],
                "sage_shap_tiny_harm": [-0.001] * 5,
                "sage_shap_helpful": [0.5] * 5,
                "sage_shap_uncertain": [-2.0, 1.0, 1.0, -2.0, 1.0],
                "sage_shap_secondary_harm": [-0.25] * 5,
            }
        )

        candidates = select_sage_disruptive_feature_candidates(
            daily_sage,
            (
                "clear_harm",
                "tiny_harm",
                "helpful",
                "uncertain",
                "secondary_harm",
            ),
            bootstrap_replicates=200,
            random_state=7,
        )
        rows = {
            str(record["feature"]): record
            for record in candidates.to_dict(orient="records")
        }

        self.assertTrue(rows["clear_harm"]["disruptive"])
        self.assertEqual(rows["clear_harm"]["candidate_rank"], 1)
        self.assertTrue(rows["secondary_harm"]["disruptive"])
        self.assertEqual(rows["secondary_harm"]["candidate_rank"], 2)
        self.assertTrue(rows["tiny_harm"]["disruptive"])
        self.assertEqual(rows["tiny_harm"]["candidate_rank"], 3)
        self.assertFalse(rows["tiny_harm"]["mean_abs_sage_not_tiny"])
        self.assertIn("tiny_mean_abs_sage", rows["tiny_harm"]["failed_conditions"])
        self.assertFalse(rows["helpful"]["disruptive"])
        self.assertIn(
            "non_negative_mean_sage",
            rows["helpful"]["failed_conditions"],
        )
        self.assertFalse(rows["uncertain"]["disruptive"])
        self.assertIn(
            "bootstrap_upper_ci_not_negative",
            rows["uncertain"]["failed_conditions"],
        )

    def test_no_sage_candidate_when_upper_ci_is_non_negative(self) -> None:
        daily_sage = pd.DataFrame(
            {
                "sage_shap_a": [0.1, 0.2, 0.3],
                "sage_shap_b": [0.0, 0.1, 0.2],
            }
        )

        candidates = select_sage_disruptive_feature_candidates(
            daily_sage,
            ("a", "b"),
            bootstrap_replicates=20,
            random_state=3,
        )

        self.assertEqual(int(candidates["disruptive"].sum()), 0)

    def test_sage_selection_uses_validation_rmse_improvement_after_hiding(
        self,
    ) -> None:
        summaries = [
            {
                "feature": "lowest_mean_sage",
                "candidate_rank": 1,
                "validation_sage_rmse_improvement": 0.1,
                "validation_sage_regret_improvement": 10.0,
            },
            {
                "feature": "best_rmse",
                "candidate_rank": 2,
                "validation_sage_rmse_improvement": 0.5,
                "validation_sage_regret_improvement": 0.0,
            },
            {
                "feature": "tied_rmse_lower_rank",
                "candidate_rank": 3,
                "validation_sage_rmse_improvement": 0.5,
                "validation_sage_regret_improvement": -1.0,
            },
        ]

        selected = _select_sage_validation_candidate_summary(summaries)

        self.assertIsNotNone(selected)
        self.assertEqual(selected["feature"], "best_rmse")

    def test_lofo_selection_uses_minimum_validation_total_regret_delta(self) -> None:
        summaries = [
            {
                "feature": "small_rmse_gain",
                "validation_lofo_total_regret_delta": -2.0,
                "validation_lofo_rmse_improvement": 0.1,
            },
            {
                "feature": "best_regret",
                "validation_lofo_total_regret_delta": -3.0,
                "validation_lofo_rmse_improvement": -0.5,
            },
            {
                "feature": "tied_regret_better_rmse",
                "validation_lofo_total_regret_delta": -3.0,
                "validation_lofo_rmse_improvement": 0.2,
            },
        ]

        selected = _select_lofo_validation_candidate_summary(summaries)

        self.assertIsNotNone(selected)
        self.assertEqual(selected["feature"], "tied_regret_better_rmse")


class TorchMlpGuidedValidationTrainingTests(unittest.TestCase):
    def test_torch_mlp_accepts_guided_validation_training_knobs(self) -> None:
        X_train = pd.DataFrame(
            {
                "f0": np.linspace(0.0, 1.0, 12),
                "f1": np.linspace(1.0, 0.0, 12),
            }
        )
        y_train = pd.DataFrame(
            {
                "y0": np.linspace(0.0, 2.0, 12),
                "y1": np.linspace(2.0, 0.0, 12),
            }
        )

        artifacts = train_model(
            X_train,
            y_train,
            model_name="torch_mlp",
            random_state=7,
            mlp_hidden_layer_sizes=(8, 8),
            mlp_max_iter=3,
            mlp_dropout=0.1,
            mlp_weight_decay=1e-4,
            mlp_batch_size=4,
            mlp_early_stopping_patience=2,
            mlp_activation="relu",
            mlp_batch_norm=False,
            learning_rate=1e-3,
        )

        predictions = artifacts.model.predict(X_train.iloc[:3])
        self.assertEqual(artifacts.model_name, "torch_mlp")
        self.assertEqual(predictions.shape, (3, 2))
        self.assertEqual(artifacts.model.epochs_trained_, 3)


if __name__ == "__main__":
    unittest.main()
