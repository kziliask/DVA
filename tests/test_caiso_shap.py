from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from dva.analysis.caiso_shap import (
    CaisoShapCaseStudyConfig,
    compute_exact_shapley_values,
    run_caiso_shap_case_study,
    select_recent_background_frame,
    write_caiso_shap_case_study_outputs,
)
from dva.analysis.caiso_sweep_runs import (
    load_caiso_shap_case_study_sweep_manifest,
)
from dva.model.storage_dispatch import (
    StorageDispatchParameters,
    build_storage_dispatch_model,
    evaluate_storage_dispatch_result,
    solve_storage_dispatch,
    solve_storage_dispatch_lexicographic,
)
from dva.model.train import load_default_train_explain_split, train_model


class XGBoostTrainingTests(unittest.TestCase):
    def test_chronological_split_matches_expected_dates(self) -> None:
        split = load_default_train_explain_split(holdout_days=60)

        self.assertEqual(len(split.X_train), 1134)
        self.assertEqual(len(split.X_explain), 60)
        self.assertEqual(split.train_dates.iloc[0], "2023-01-26")
        self.assertEqual(split.train_dates.iloc[-1], "2026-03-07")
        self.assertEqual(split.explain_dates.iloc[0], "2026-03-09")
        self.assertEqual(split.explain_dates.iloc[-1], "2026-05-07")

    def test_default_background_uses_last_training_year(self) -> None:
        split = load_default_train_explain_split()
        background = select_recent_background_frame(
            split.train_frame,
            split.date_column,
            CaisoShapCaseStudyConfig().background_days,
        )

        self.assertEqual(len(background), 364)
        self.assertEqual(background[split.date_column].iloc[0], "2025-01-26")
        self.assertEqual(background[split.date_column].iloc[-1], "2026-01-25")

    def test_train_model_xgb_returns_price_predictions(self) -> None:
        X_train = pd.DataFrame(
            {
                "f0": [0.0, 1.0, 0.0, 1.0],
                "f1": [0.0, 0.0, 1.0, 1.0],
            }
        )
        y_train = pd.DataFrame(
            {
                "y0": [1.0, 2.0, 3.0, 4.0],
                "y1": [0.5, 1.5, 2.5, 3.5],
            }
        )

        artifacts = train_model(
            X_train,
            y_train,
            model_name="xgb",
            random_state=0,
            n_jobs=1,
            xgb_n_estimators=2,
            xgb_max_depth=1,
        )

        predictions = artifacts.model.predict(X_train.iloc[:2])
        self.assertEqual(artifacts.model_name, "xgb")
        self.assertEqual(predictions.shape, (2, 2))
        self.assertIn("XGBRegressor", artifacts.model_description)

    def test_train_model_rejects_archived_model_names(self) -> None:
        X_train = pd.DataFrame({"f0": [0.0, 1.0]})
        y_train = pd.DataFrame({"y0": [1.0, 2.0]})

        with self.assertRaisesRegex(ValueError, "Only model_name='xgb'"):
            train_model(X_train, y_train, model_name="mlp")

    def test_predictive_shap_efficiency_holds_for_vector_outputs(self) -> None:
        coalition_values = np.array(
            [
                [1.0, 10.0],
                [2.0, 11.0],
                [4.0, 14.0],
                [7.0, 19.0],
            ]
        )
        shap_values = compute_exact_shapley_values(coalition_values, feature_count=2)

        np.testing.assert_allclose(
            shap_values.sum(axis=0),
            coalition_values[-1] - coalition_values[0],
        )
        np.testing.assert_allclose(
            shap_values.sum(axis=1),
            np.array([5.0, 10.0]),
        )


class StorageDispatchLexicographicTests(unittest.TestCase):
    def test_throughput_penalty_below_one_keeps_big_m_constraints(self) -> None:
        parameters = StorageDispatchParameters(
            energy_capacity=2.0,
            power_limit=1.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            throughput_penalty=0.5,
            initial_state_of_charge=1.0,
            terminal_state_of_charge=1.0,
        )

        storage_model = build_storage_dispatch_model([1.0, -1.0], parameters)
        constraint_names = set(storage_model.model.component_map(pyo.Constraint))
        binary_count = sum(
            1
            for variable in storage_model.model.component_data_objects(pyo.Var)
            if variable.is_binary()
        )

        self.assertEqual(binary_count, 2)
        self.assertIn("charge_limit", constraint_names)
        self.assertIn("discharge_limit", constraint_names)
        self.assertNotIn("shared_power_limit", constraint_names)

    def test_throughput_penalty_at_least_one_uses_relaxed_constraints(self) -> None:
        parameters = StorageDispatchParameters(
            energy_capacity=2.0,
            power_limit=1.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            throughput_penalty=1.0,
            initial_state_of_charge=1.0,
            terminal_state_of_charge=1.0,
        )

        storage_model = build_storage_dispatch_model([1.0, -1.0], parameters)
        constraint_names = set(storage_model.model.component_map(pyo.Constraint))
        binary_count = sum(
            1
            for variable in storage_model.model.component_data_objects(pyo.Var)
            if variable.is_binary()
        )

        self.assertEqual(binary_count, 0)
        self.assertIsNone(storage_model.mode)
        self.assertIn("shared_power_limit", constraint_names)
        self.assertNotIn("charge_limit", constraint_names)
        self.assertNotIn("discharge_limit", constraint_names)

    def test_relaxed_dispatch_still_returns_binary_mode_proxy(self) -> None:
        parameters = StorageDispatchParameters(
            energy_capacity=2.0,
            power_limit=1.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            throughput_penalty=1.0,
            initial_state_of_charge=1.0,
            terminal_state_of_charge=1.0,
        )

        result = solve_storage_dispatch([0.0, 0.0], parameters)

        self.assertEqual(result.mode, (0, 0))

    def test_lexicographic_dispatch_is_reproducible_and_preserves_primary_objective(self) -> None:
        predicted_prices = [10.0, 10.0, 10.0, 10.0]
        parameters = StorageDispatchParameters(
            energy_capacity=2.0,
            power_limit=1.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            throughput_penalty=0.0,
            initial_state_of_charge=1.0,
            terminal_state_of_charge=1.0,
        )
        solver_params = {
            "Threads": 1,
            "Seed": 0,
            "MIPGap": 0.0,
            "MIPGapAbs": 1e-9,
        }

        primary_result = solve_storage_dispatch(
            predicted_prices,
            parameters,
            solver_params=solver_params,
        )
        lexicographic_result_a = solve_storage_dispatch_lexicographic(
            predicted_prices,
            parameters,
            solver_params=solver_params,
            objective_tolerance=1e-6,
        )
        lexicographic_result_b = solve_storage_dispatch_lexicographic(
            predicted_prices,
            parameters,
            solver_params=solver_params,
            objective_tolerance=1e-6,
        )

        self.assertGreaterEqual(
            lexicographic_result_a.objective_value + 1e-6,
            primary_result.objective_value,
        )
        self.assertEqual(lexicographic_result_a.charge, lexicographic_result_b.charge)
        self.assertEqual(lexicographic_result_a.discharge, lexicographic_result_b.discharge)
        self.assertEqual(lexicographic_result_a.mode, lexicographic_result_b.mode)


class CaisoCaseStudySmokeTests(unittest.TestCase):
    def test_case_study_config_defaults_to_xgb(self) -> None:
        config = CaisoShapCaseStudyConfig()

        self.assertEqual(config.model_name, "xgb")
        self.assertEqual(config.xgb_n_estimators, 100)
        self.assertEqual(config.xgb_max_depth, 3)
        self.assertEqual(config.holdout_mean_impute_features, ())
        self.assertFalse(config.compute_ead_decision_shap)

    def test_sweep_run_manifest_loads_xgb_case_study_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest = pd.DataFrame(
                [
                    {
                        "setting_id": "default",
                        "results_dir": "default_run",
                        "sweep_id": "throughput",
                        "parameter_name": "throughput_penalty",
                        "parameter_value": 0.0,
                        "step_index": 0,
                        "step_label": "0",
                        "model": "xgb",
                    },
                    {
                        "setting_id": "efficiency",
                        "results_dir": "efficiency_run",
                        "sweep_id": "efficiency",
                        "parameter_name": "charge_discharge_efficiency_pair",
                        "parameter_value": 1.0,
                        "step_index": 0,
                        "step_label": "0.8/1.0",
                        "model": "xgb",
                        "charge_efficiency": 0.8,
                        "discharge_efficiency": 1.0,
                        "energy_capacity": 4.0,
                        "throughput_penalty": 5.0,
                    },
                ]
            )
            manifest_path = tmp_path / "sweep_manifest.csv"
            manifest.to_csv(manifest_path, index=False)

            run_entries = load_caiso_shap_case_study_sweep_manifest(manifest_path)
            run_entries_by_setting = {
                run_entry.manifest_entry.setting_id: run_entry
                for run_entry in run_entries
            }

            self.assertEqual(len(run_entries), 2)
            default_entry = run_entries_by_setting["default"]
            efficiency_entry = run_entries_by_setting["efficiency"]

            self.assertEqual(default_entry.config.model_name, "xgb")
            self.assertEqual(
                default_entry.config.storage_parameters.energy_capacity,
                2.0,
            )
            self.assertEqual(
                efficiency_entry.config.storage_parameters.charge_efficiency,
                0.8,
            )
            self.assertEqual(
                efficiency_entry.config.storage_parameters.discharge_efficiency,
                1.0,
            )
            self.assertEqual(
                efficiency_entry.config.storage_parameters.energy_capacity,
                4.0,
            )
            self.assertEqual(
                efficiency_entry.config.storage_parameters.throughput_penalty,
                5.0,
            )
            self.assertEqual(
                default_entry.manifest_entry.results_dir,
                (tmp_path / "default_run").resolve(),
            )

    def test_one_day_case_study_smoke_run_produces_consistent_outputs(self) -> None:
        config = CaisoShapCaseStudyConfig(
            max_days=1,
            xgb_n_estimators=2,
            xgb_max_depth=1,
        )
        outputs = run_caiso_shap_case_study(config)
        split = load_default_train_explain_split(
            dataset_path=config.dataset_path,
            holdout_days=config.holdout_days,
        )
        training_artifacts = train_model(
            split.X_train,
            split.y_train,
            model_name=config.model_name,
            feature_columns=split.feature_columns,
            target_columns=split.target_columns,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            xgb_n_estimators=config.xgb_n_estimators,
            xgb_max_depth=config.xgb_max_depth,
            xgb_learning_rate=config.xgb_learning_rate,
            xgb_subsample=config.xgb_subsample,
            xgb_colsample_bytree=config.xgb_colsample_bytree,
            xgb_reg_lambda=config.xgb_reg_lambda,
            xgb_verbosity=config.xgb_verbosity,
        )
        true_prices = split.y_explain.iloc[0].to_numpy(dtype=float, copy=True)
        predicted_prices = np.asarray(
            training_artifacts.model.predict(split.X_explain.iloc[[0]]),
            dtype=float,
        ).reshape(-1)
        solver_params = {
            "Threads": 1,
            "Seed": config.solver_seed,
            "MIPGap": config.mip_gap,
            "MIPGapAbs": config.mip_gap_abs,
        }
        expected_oracle_obj = float(
            solve_storage_dispatch_lexicographic(
                true_prices,
                config.storage_parameters,
                name="storage_dispatch_expected_oracle",
                log_to_console=False,
                solver_params=solver_params,
                objective_tolerance=config.objective_tolerance,
            ).objective_value
        )
        expected_decision_full_value = float(
            evaluate_storage_dispatch_result(
                true_prices,
                solve_storage_dispatch_lexicographic(
                    tuple(float(value) for value in predicted_prices),
                    config.storage_parameters,
                    name="storage_dispatch_expected_prediction",
                    log_to_console=False,
                    solver_params=solver_params,
                    objective_tolerance=config.objective_tolerance,
                ),
                config.storage_parameters,
            ).objective_value
        )

        self.assertEqual(len(outputs.daily_shap), 1)
        self.assertEqual(len(outputs.predictive_hourly_shap), 8 * 24)
        self.assertEqual(len(outputs.daily_full_dispatch), 24)
        self.assertEqual(len(outputs.summary_shap), 8)
        self.assertEqual(outputs.run_metadata["model_name"], "xgb")
        self.assertEqual(outputs.run_metadata["xgb_params"]["n_estimators"], 2)
        self.assertEqual(outputs.run_metadata["xgb_params"]["max_depth"], 1)
        self.assertEqual(outputs.run_metadata["explain_rows"], 1)
        self.assertEqual(outputs.run_metadata["coalitions_per_day"], 256)
        self.assertEqual(
            outputs.run_metadata["coalition_expectation_method"],
            "empirical_background_marginalization",
        )
        self.assertEqual(
            set(outputs.evaluation_metrics),
            {"decision", "predictive"},
        )

        daily_row = outputs.daily_shap.iloc[0]
        self.assertAlmostEqual(
            daily_row["oracle_obj"],
            expected_oracle_obj,
            places=6,
        )
        self.assertAlmostEqual(
            outputs.prediction_metrics["holdout"]["mean_actual_daily_regret"],
            expected_oracle_obj - expected_decision_full_value,
            places=6,
        )
        self.assertAlmostEqual(
            outputs.prediction_metrics["holdout"]["mean_decision_value_gain"],
            float(outputs.daily_shap["decision_value_gain"].mean()),
            places=6,
        )
        self.assertGreaterEqual(outputs.prediction_metrics["holdout"]["mae"], 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            write_caiso_shap_case_study_outputs(outputs, tmpdir)
            self.assertTrue(pd.read_csv(f"{tmpdir}/daily_full_dispatch.csv").shape[0] > 0)
            written_daily_shap = pd.read_csv(f"{tmpdir}/daily_shap.csv")
            self.assertIn("oracle_obj", written_daily_shap.columns)
            with open(f"{tmpdir}/evaluation_metrics.json", encoding="utf-8") as handle:
                self.assertIn("decision", handle.read())
            with open(f"{tmpdir}/prediction_metrics.json", encoding="utf-8") as handle:
                prediction_metrics = json.load(handle)
            self.assertEqual(
                prediction_metrics["holdout"]["predictions"],
                outputs.prediction_metrics["holdout"]["predictions"],
            )

    def test_holdout_mean_impute_feature_records_background_replacement(self) -> None:
        config = CaisoShapCaseStudyConfig(
            max_days=1,
            xgb_n_estimators=2,
            xgb_max_depth=1,
            holdout_mean_impute_features=("mean_wind_speed",),
        )

        outputs = run_caiso_shap_case_study(config)
        split = load_default_train_explain_split(
            dataset_path=config.dataset_path,
            holdout_days=config.holdout_days,
        )
        background = select_recent_background_frame(
            split.train_frame,
            split.date_column,
            config.background_days,
        )
        expected_replacement = float(background["mean_wind_speed"].mean())

        self.assertEqual(
            outputs.run_metadata["holdout_mean_impute_features"],
            ["mean_wind_speed"],
        )
        self.assertEqual(
            outputs.run_metadata["holdout_feature_replacement_strategy"],
            "background_mean",
        )
        self.assertAlmostEqual(
            outputs.run_metadata["holdout_feature_replacements"]["mean_wind_speed"],
            expected_replacement,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
