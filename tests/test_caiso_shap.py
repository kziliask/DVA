from __future__ import annotations
import io
import json
import tempfile
import warnings
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning

from dva.analysis.caiso_shap import (
    CaisoShapCaseStudyConfig,
    ExactRandomForestCoalitionEvaluator,
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
    StorageDispatchSPOModel,
    build_storage_dispatch_model,
    build_spo_training_targets,
    evaluate_storage_dispatch_result,
    prices_to_spo_costs,
    solve_storage_dispatch,
    solve_storage_dispatch_lexicographic,
)
from dva.model.train import (
    load_default_train_explain_split,
    load_default_random_forest_train_explain_split,
    train_model,
)


class ExactRandomForestCoalitionEvaluatorTests(unittest.TestCase):
    def test_chronological_split_matches_expected_dates(self) -> None:
        split = load_default_random_forest_train_explain_split(holdout_days=60)

        self.assertEqual(len(split.X_train), 1134)
        self.assertEqual(len(split.X_explain), 60)
        self.assertEqual(split.train_dates.iloc[0], "2023-01-26")
        self.assertEqual(split.train_dates.iloc[-1], "2026-03-07")
        self.assertEqual(split.explain_dates.iloc[0], "2026-03-09")
        self.assertEqual(split.explain_dates.iloc[-1], "2026-05-07")

    def test_default_background_uses_last_training_year(self) -> None:
        split = load_default_random_forest_train_explain_split()
        background = select_recent_background_frame(
            split.train_frame,
            split.date_column,
            CaisoShapCaseStudyConfig().background_days,
        )

        self.assertEqual(len(background), 364)
        self.assertEqual(background[split.date_column].iloc[0], "2025-01-26")
        self.assertEqual(background[split.date_column].iloc[-1], "2026-01-25")

    def test_exact_coalition_predictions_match_empty_and_full_cases(self) -> None:
        X = pd.DataFrame(
            {
                "f0": [0.0, 1.0, 0.0, 1.0],
                "f1": [0.0, 0.0, 1.0, 1.0],
            }
        )
        y = pd.DataFrame(
            {
                "y0": [0.0, 1.0, 1.0, 2.0],
                "y1": [0.0, 2.0, 2.0, 4.0],
            }
        )
        model = RandomForestRegressor(
            n_estimators=5,
            max_depth=2,
            random_state=0,
            n_jobs=1,
        )
        model.fit(X, y)
        evaluator = ExactRandomForestCoalitionEvaluator(model, tuple(X.columns))

        observation = X.iloc[3]
        coalition_predictions = evaluator.evaluate_all_coalitions(observation)
        expected_empty = np.mean(
            [estimator.tree_.value[0, :, 0] for estimator in model.estimators_],
            axis=0,
        )
        expected_full = model.predict(observation.to_frame().T)[0]

        np.testing.assert_allclose(coalition_predictions[0], expected_empty)
        np.testing.assert_allclose(coalition_predictions[-1], expected_full)

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
        storage_model.model.update()
        constraint_names = {
            constraint.ConstrName for constraint in storage_model.model.getConstrs()
        }

        self.assertEqual(storage_model.model.NumBinVars, 2)
        self.assertIn("charge_limit[1]", constraint_names)
        self.assertIn("discharge_limit[1]", constraint_names)
        self.assertNotIn("shared_power_limit[1]", constraint_names)

        storage_model.model.dispose()

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
        storage_model.model.update()
        constraint_names = {
            constraint.ConstrName for constraint in storage_model.model.getConstrs()
        }

        self.assertEqual(storage_model.model.NumBinVars, 0)
        self.assertIsNone(storage_model.mode)
        self.assertIn("shared_power_limit[1]", constraint_names)
        self.assertNotIn("charge_limit[1]", constraint_names)
        self.assertNotIn("discharge_limit[1]", constraint_names)

        storage_model.model.dispose()

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

    def test_zero_penalty_lexicographic_dispatch_does_not_raise(self) -> None:
        split = load_default_random_forest_train_explain_split(holdout_days=60)
        training = RandomForestRegressor(random_state=0, n_jobs=1)
        training.fit(split.X_train, split.y_train)
        evaluator = ExactRandomForestCoalitionEvaluator(training, split.feature_columns)

        predicted_prices = evaluator.evaluate_all_coalitions(split.X_explain.iloc[0])[0]
        parameters = StorageDispatchParameters(
            energy_capacity=2.0,
            power_limit=1.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
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

        result = solve_storage_dispatch_lexicographic(
            predicted_prices,
            parameters,
            solver_params=solver_params,
            objective_tolerance=1e-6,
        )

        self.assertEqual(len(result.charge), 24)


class StorageDispatchSPOTests(unittest.TestCase):
    def test_spo_model_uses_relaxed_constraints_when_penalty_is_at_least_one(self) -> None:
        parameters = StorageDispatchParameters(
            energy_capacity=2.0,
            power_limit=1.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            throughput_penalty=5.0,
            initial_state_of_charge=1.0,
            terminal_state_of_charge=1.0,
        )

        spo_model, _ = StorageDispatchSPOModel(
            parameters=parameters,
            horizon=2,
        )._getModel()
        spo_model.update()
        constraint_names = {
            constraint.ConstrName for constraint in spo_model.getConstrs()
        }

        self.assertEqual(spo_model.NumBinVars, 0)
        self.assertIn("shared_power_limit[1]", constraint_names)
        self.assertNotIn("charge_limit[1]", constraint_names)
        self.assertNotIn("discharge_limit[1]", constraint_names)

        spo_model.dispose()

    def test_prices_to_spo_costs_maps_charge_and_discharge_coefficients(self) -> None:
        costs = prices_to_spo_costs(
            np.array([10.0, -5.0], dtype=np.float32),
            throughput_penalty=2.0,
        )

        np.testing.assert_allclose(
            costs,
            np.array([-12.0, 3.0, 8.0, -7.0], dtype=np.float32),
        )

    def test_spo_training_targets_match_dispatch_solution_and_objective(self) -> None:
        prices = np.array([[1.0, 10.0]], dtype=np.float32)
        parameters = StorageDispatchParameters(
            energy_capacity=1.0,
            power_limit=1.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            throughput_penalty=0.0,
            initial_state_of_charge=0.0,
            terminal_state_of_charge=0.0,
        )
        solver_params = {
            "Threads": 1,
            "Seed": 0,
            "MIPGap": 0.0,
            "MIPGapAbs": 1e-9,
        }

        costs, solutions, objectives = build_spo_training_targets(
            prices,
            parameters,
            solver_params=solver_params,
        )
        expected_dispatch = solve_storage_dispatch(
            prices[0],
            parameters,
            solver_params=solver_params,
        )
        expected_solution = np.concatenate(
            (
                np.asarray(expected_dispatch.charge, dtype=np.float32),
                np.asarray(expected_dispatch.discharge, dtype=np.float32),
            )
        )

        np.testing.assert_allclose(
            costs[0],
            prices_to_spo_costs(prices[0], parameters.throughput_penalty),
        )
        np.testing.assert_allclose(solutions[0], expected_solution)
        self.assertAlmostEqual(
            float(objectives[0, 0]),
            expected_dispatch.objective_value,
            places=6,
        )

    def test_spo_plus_loss_is_near_zero_for_perfect_price_predictions(self) -> None:
        import pyepo.func as pyepo_func
        import torch

        split = load_default_train_explain_split()
        y_array = split.y_train.iloc[:8].to_numpy(dtype=np.float32, copy=True)
        parameters = StorageDispatchParameters(
            energy_capacity=2.0,
            power_limit=1.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            throughput_penalty=2.0,
            initial_state_of_charge=1.0,
            terminal_state_of_charge=1.0,
        )
        solver_params = {
            "Threads": 1,
            "Seed": 0,
            "MIPGap": 0.0,
            "MIPGapAbs": 1e-9,
        }

        true_costs, true_solutions, true_objectives = build_spo_training_targets(
            y_array,
            parameters,
            solver_params=solver_params,
        )
        loss_fn = pyepo_func.SPOPlus(
            optmodel=StorageDispatchSPOModel(
                parameters=parameters,
                horizon=int(y_array.shape[1]),
                solver_params=solver_params,
            ),
            processes=1,
        )
        true_cost_tensor = torch.from_numpy(true_costs)
        true_solution_tensor = torch.from_numpy(true_solutions)
        true_objective_tensor = torch.from_numpy(true_objectives)
        throughput_penalty = torch.tensor(
            parameters.throughput_penalty,
            dtype=torch.float32,
        )

        with torch.no_grad():
            perfect = torch.from_numpy(y_array)
            perfect_costs = torch.cat(
                (
                    -(perfect + throughput_penalty),
                    perfect - throughput_penalty,
                ),
                dim=1,
            )
            loss = loss_fn(
                perfect_costs,
                true_cost_tensor,
                true_solution_tensor,
                true_objective_tensor,
            )

        self.assertLessEqual(
            abs(float(loss.detach().item())),
            1e-5,
            msg=f"Expected near-zero perfect-prediction SPO+ loss, got {float(loss):.6e}",
        )

    def test_train_model_spo_mlp_returns_price_predictions(self) -> None:
        X_train = pd.DataFrame(
            {
                "f0": [0.0, 1.0, 0.0],
                "f1": [0.0, 0.0, 1.0],
            }
        )
        y_train = pd.DataFrame(
            {
                "y0": [1.0, 2.0, 3.0],
                "y1": [0.5, 1.5, 2.5],
                "y2": [2.0, 1.0, 0.0],
                "y3": [1.0, 0.0, 1.0],
            }
        )
        parameters = StorageDispatchParameters(
            energy_capacity=2.0,
            power_limit=1.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            throughput_penalty=0.0,
            initial_state_of_charge=0.0,
            terminal_state_of_charge=0.0,
        )

        artifacts = train_model(
            X_train,
            y_train,
            model_name="spo_mlp",
            random_state=0,
            mlp_max_iter=1,
            storage_parameters=parameters,
            spo_processes=1,
        )

        predictions = artifacts.model.predict(X_train.iloc[:2])
        self.assertEqual(artifacts.model_name, "spo_mlp")
        self.assertEqual(predictions.shape, (2, 4))

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

    def test_train_model_spo_mlp_verbose_prints_progress(self) -> None:
        X_train = pd.DataFrame(
            {
                "f0": [0.0, 1.0],
                "f1": [1.0, 0.0],
            }
        )
        y_train = pd.DataFrame(
            {
                "y0": [1.0, 2.0],
                "y1": [0.5, 1.5],
            }
        )
        parameters = StorageDispatchParameters(
            energy_capacity=1.0,
            power_limit=1.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            throughput_penalty=0.0,
            initial_state_of_charge=0.0,
            terminal_state_of_charge=0.0,
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            train_model(
                X_train,
                y_train,
                model_name="spo_mlp",
                random_state=0,
                mlp_max_iter=1,
                storage_parameters=parameters,
                training_verbose=True,
                training_log_every=1,
                spo_processes=1,
            )

        output = buffer.getvalue()
        self.assertIn("[spo_targets] solving", output)
        self.assertIn("[spo_targets] solved 2/2", output)
        self.assertIn("[spo_mlp] epoch 1/1", output)
        self.assertIn("spo_processes=1", output)

    def test_train_model_spo_mlp_warm_start_with_mse_verbose_prints_both_phases(self) -> None:
        X_train = pd.DataFrame(
            {
                "f0": [0.0, 1.0],
                "f1": [1.0, 0.0],
            }
        )
        y_train = pd.DataFrame(
            {
                "y0": [1.0, 2.0],
                "y1": [0.5, 1.5],
            }
        )
        parameters = StorageDispatchParameters(
            energy_capacity=1.0,
            power_limit=1.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            throughput_penalty=0.0,
            initial_state_of_charge=0.0,
            terminal_state_of_charge=0.0,
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            train_model(
                X_train,
                y_train,
                model_name="spo_mlp",
                random_state=0,
                mlp_max_iter=1,
                learning_rate=1e-2,
                mse_learning_rate=5e-3,
                spo_learning_rate=2e-4,
                storage_parameters=parameters,
                training_verbose=True,
                training_log_every=1,
                spo_processes=1,
                spo_warm_start_with_mse=True,
            )

        output = buffer.getvalue()
        self.assertIn("MSE warm-start enabled", output)
        self.assertGreaterEqual(output.count("[spo_mlp] epoch 1/1"), 2)
        self.assertIn("learning_rate=0.005", output)
        self.assertIn("warm_start_with_mse=True", output)
        self.assertIn("mse_learning_rate=0.005", output)
        self.assertIn("spo_learning_rate=0.0002", output)


class CaisoCaseStudySmokeTests(unittest.TestCase):
    def test_case_study_config_defaults_to_xgb(self) -> None:
        config = CaisoShapCaseStudyConfig()

        self.assertEqual(config.model_name, "xgb")
        self.assertEqual(config.xgb_n_estimators, 100)
        self.assertEqual(config.xgb_max_depth, 3)
        self.assertEqual(config.holdout_mean_impute_features, ())
        self.assertFalse(config.compute_ead_decision_shap)

    def test_sweep_run_manifest_loads_case_study_configs_with_defaults_and_overrides(self) -> None:
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
                        "model": "mlp",
                        "mlp_max_iter": 10000,
                    },
                    {
                        "setting_id": "efficiency",
                        "results_dir": "efficiency_run",
                        "sweep_id": "efficiency",
                        "parameter_name": "charge_discharge_efficiency_pair",
                        "parameter_value": 1.0,
                        "step_index": 0,
                        "step_label": "0.8/1.0",
                        "model": "mlp",
                        "mlp_max_iter": 10000,
                        "lr": 0.02,
                        "mse_lr": 0.01,
                        "spo_lr": 0.001,
                        "spo_warm_start_with_mse": True,
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

            self.assertEqual(default_entry.config.model_name, "mlp")
            self.assertEqual(default_entry.config.mlp_max_iter, 10000)
            self.assertIsNone(default_entry.config.learning_rate)
            self.assertIsNone(default_entry.config.mse_learning_rate)
            self.assertIsNone(default_entry.config.spo_learning_rate)
            self.assertFalse(default_entry.config.spo_warm_start_with_mse)
            self.assertEqual(
                default_entry.config.storage_parameters.energy_capacity,
                2.0,
            )
            self.assertEqual(
                default_entry.config.storage_parameters.charge_efficiency,
                0.95,
            )
            self.assertEqual(
                default_entry.config.storage_parameters.discharge_efficiency,
                0.95,
            )
            self.assertEqual(
                default_entry.config.storage_parameters.throughput_penalty,
                0.0,
            )
            self.assertEqual(
                efficiency_entry.config.storage_parameters.charge_efficiency,
                0.8,
            )
            self.assertEqual(efficiency_entry.config.learning_rate, 0.02)
            self.assertEqual(efficiency_entry.config.mse_learning_rate, 0.01)
            self.assertEqual(efficiency_entry.config.spo_learning_rate, 0.001)
            self.assertTrue(efficiency_entry.config.spo_warm_start_with_mse)
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
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
                mlp_hidden_layer_sizes=config.mlp_hidden_layer_sizes,
                mlp_max_iter=config.mlp_max_iter,
                xgb_n_estimators=config.xgb_n_estimators,
                xgb_max_depth=config.xgb_max_depth,
                xgb_learning_rate=config.xgb_learning_rate,
                xgb_subsample=config.xgb_subsample,
                xgb_colsample_bytree=config.xgb_colsample_bytree,
                xgb_reg_lambda=config.xgb_reg_lambda,
                xgb_verbosity=config.xgb_verbosity,
                storage_parameters=config.storage_parameters,
                training_verbose=config.training_verbose,
                training_log_every=config.training_log_every,
                spo_warm_start_with_mse=config.spo_warm_start_with_mse,
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
        expected_actual_daily_regret = float(
            expected_oracle_obj - expected_decision_full_value
        )

        self.assertEqual(len(outputs.daily_shap), 1)
        self.assertEqual(len(outputs.predictive_hourly_shap), 8 * 24)
        self.assertEqual(len(outputs.daily_full_dispatch), 24)
        self.assertEqual(len(outputs.summary_shap), 8)
        self.assertEqual(outputs.run_metadata["model_name"], "xgb")
        self.assertEqual(outputs.run_metadata["xgb_params"]["n_estimators"], 2)
        self.assertEqual(outputs.run_metadata["xgb_params"]["max_depth"], 1)
        self.assertIn("holdout", outputs.prediction_metrics)
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
        self.assertEqual(
            outputs.run_metadata["evaluation_metric_parameters"][
                "decision_deletion_auc_clipping"
            ],
            "clip_to_[0,v_dec(N)]",
        )
        self.assertTrue(
            outputs.run_metadata["evaluation_metric_parameters"][
                "decision_deletion_auc_lower_is_better"
            ]
        )
        self.assertTrue(
            outputs.run_metadata["evaluation_metric_parameters"][
                "decision_deletion_auc_requires_positive_full_gain"
            ]
        )
        self.assertTrue(
            outputs.run_metadata["evaluation_metric_parameters"][
                "decision_deletion_auc_zero_if_all_strict_suffixes_nonpositive"
            ]
        )
        self.assertEqual(
            outputs.run_metadata["evaluation_metric_parameters"][
                "decision_insertion_auc_clipping"
            ],
            "clip_to_[0,v_dec(N)]",
        )
        self.assertTrue(
            outputs.run_metadata["evaluation_metric_parameters"][
                "decision_insertion_auc_requires_positive_full_gain"
            ]
        )
        self.assertTrue(
            outputs.run_metadata["evaluation_metric_parameters"][
                "decision_insertion_auc_zero_if_all_strict_prefixes_nonpositive"
            ]
        )
        self.assertEqual(
            outputs.evaluation_metrics["decision"]["decision_deletion_auc"][
                "clipping"
            ],
            "clip_to_[0,v_dec(N)]",
        )
        self.assertTrue(
            outputs.evaluation_metrics["decision"]["decision_deletion_auc"][
                "lower_is_better"
            ]
        )
        self.assertTrue(
            outputs.evaluation_metrics["decision"]["decision_deletion_auc"][
                "requires_positive_full_gain"
            ]
        )
        self.assertTrue(
            outputs.evaluation_metrics["decision"]["decision_deletion_auc"][
                "zero_if_all_strict_suffixes_nonpositive"
            ]
        )
        self.assertEqual(
            outputs.evaluation_metrics["decision"]["decision_insertion_auc"][
                "clipping"
            ],
            "clip_to_[0,v_dec(N)]",
        )
        self.assertTrue(
            outputs.evaluation_metrics["decision"]["decision_insertion_auc"][
                "requires_positive_full_gain"
            ]
        )
        self.assertTrue(
            outputs.evaluation_metrics["decision"]["decision_insertion_auc"][
                "zero_if_all_strict_prefixes_nonpositive"
            ]
        )

        daily_row = outputs.daily_shap.iloc[0]
        predictive_shap_sum = sum(
            daily_row[f"predictive_shap_{feature_name}"]
            for feature_name in outputs.run_metadata["feature_columns"]
        )
        decision_shap_sum = sum(
            daily_row[f"decision_shap_{feature_name}"]
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
        self.assertIn("predictive_decision_infidelity", daily_row.index)
        self.assertIn("decision_decision_infidelity", daily_row.index)
        self.assertIn("predictive_decision_deletion_auc", daily_row.index)
        self.assertIn("decision_decision_deletion_auc", daily_row.index)
        self.assertIn("predictive_decision_insertion_auc", daily_row.index)
        self.assertIn("decision_decision_insertion_auc", daily_row.index)
        self.assertIn("abs_rank_kendall_tau", daily_row.index)
        self.assertIn("oracle_obj", daily_row.index)
        for feature_name in outputs.run_metadata["feature_columns"]:
            activation_rate_column = f"decision_activation_rate_{feature_name}"
            activated_value_sum_column = f"decision_activated_value_sum_{feature_name}"
            activated_value_column = f"decision_activated_value_{feature_name}"
            self.assertIn(activation_rate_column, daily_row.index)
            self.assertIn(activated_value_sum_column, daily_row.index)
            self.assertIn(activated_value_column, daily_row.index)
            self.assertGreaterEqual(daily_row[activation_rate_column], 0.0)
            self.assertLessEqual(daily_row[activation_rate_column], 1.0)
            if daily_row[activation_rate_column] == 0.0:
                self.assertAlmostEqual(daily_row[activated_value_column], 0.0)
        self.assertIn("decision_activation_rate", outputs.summary_shap.columns)
        self.assertIn("decision_activated_value", outputs.summary_shap.columns)
        self.assertTrue(
            outputs.summary_shap["decision_activation_rate"]
            .between(0.0, 1.0)
            .all()
        )
        self.assertAlmostEqual(
            daily_row["oracle_obj"],
            expected_oracle_obj,
            places=6,
        )

        aggregated_hourly = (
            outputs.predictive_hourly_shap.groupby("feature", sort=False)["shap_value"].sum().to_dict()
        )
        for feature_name in outputs.run_metadata["feature_columns"]:
            self.assertAlmostEqual(
                aggregated_hourly[feature_name],
                daily_row[f"predictive_shap_{feature_name}"],
                places=6,
            )

        global_spearman = outputs.comparison_metrics["global_abs_rank_spearman"]
        self.assertTrue(global_spearman is None or -1.0 <= global_spearman <= 1.0)
        daily_spearman = outputs.comparison_metrics["daily_abs_rank_spearman"]["mean"]
        self.assertTrue(daily_spearman is None or -1.0 <= daily_spearman <= 1.0)
        global_kendall_tau = outputs.comparison_metrics["global_abs_rank_kendall_tau"]
        self.assertTrue(global_kendall_tau is None or -1.0 <= global_kendall_tau <= 1.0)
        daily_kendall_tau = outputs.comparison_metrics["daily_abs_rank_kendall_tau"]["mean"]
        self.assertTrue(daily_kendall_tau is None or -1.0 <= daily_kendall_tau <= 1.0)
        self.assertGreaterEqual(
            outputs.evaluation_metrics["decision"]["decision_infidelity"]["valid_days"],
            1,
        )
        self.assertGreaterEqual(
            outputs.evaluation_metrics["predictive"]["decision_infidelity"]["valid_days"],
            1,
        )
        self.assertGreaterEqual(outputs.prediction_metrics["holdout"]["mae"], 0.0)
        self.assertGreaterEqual(outputs.prediction_metrics["holdout"]["mse"], 0.0)
        self.assertGreaterEqual(outputs.prediction_metrics["holdout"]["rmse"], 0.0)
        self.assertEqual(outputs.prediction_metrics["holdout"]["days"], 1)
        self.assertEqual(outputs.prediction_metrics["holdout"]["targets_per_day"], 24)
        self.assertEqual(outputs.prediction_metrics["holdout"]["predictions"], 24)
        self.assertAlmostEqual(
            outputs.prediction_metrics["holdout"]["mean_decision_value_gain"],
            float(outputs.daily_shap["decision_value_gain"].mean()),
            places=6,
        )
        self.assertAlmostEqual(
            outputs.prediction_metrics["holdout"]["mean_actual_daily_regret"],
            expected_actual_daily_regret,
            places=6,
        )
        self.assertAlmostEqual(
            outputs.prediction_metrics["holdout"]["mean_actual_daily_regret"],
            float(
                (outputs.daily_shap["oracle_obj"] - outputs.daily_shap["decision_full_value"]).mean()
            ),
            places=6,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            write_caiso_shap_case_study_outputs(outputs, tmpdir)
            self.assertTrue(pd.read_csv(f"{tmpdir}/daily_full_dispatch.csv").shape[0] > 0)
            written_daily_shap = pd.read_csv(f"{tmpdir}/daily_shap.csv")
            self.assertIn("abs_rank_kendall_tau", written_daily_shap.columns)
            self.assertIn("oracle_obj", written_daily_shap.columns)
            self.assertIn(
                f"decision_activation_rate_{outputs.run_metadata['feature_columns'][0]}",
                written_daily_shap.columns,
            )
            with open(f"{tmpdir}/evaluation_metrics.json", encoding="utf-8") as handle:
                self.assertIn("decision", handle.read())
            with open(f"{tmpdir}/comparison_metrics.json", encoding="utf-8") as handle:
                comparison_metrics = json.load(handle)
            self.assertIn("global_abs_rank_kendall_tau", comparison_metrics)
            with open(f"{tmpdir}/prediction_metrics.json", encoding="utf-8") as handle:
                prediction_metrics = json.load(handle)
            self.assertEqual(
                prediction_metrics["holdout"]["predictions"],
                outputs.prediction_metrics["holdout"]["predictions"],
            )
            self.assertAlmostEqual(
                prediction_metrics["holdout"]["mean_decision_value_gain"],
                outputs.prediction_metrics["holdout"]["mean_decision_value_gain"],
                places=6,
            )
            self.assertAlmostEqual(
                prediction_metrics["holdout"]["mean_actual_daily_regret"],
                outputs.prediction_metrics["holdout"]["mean_actual_daily_regret"],
                places=6,
            )

    def test_one_day_mlp_case_study_smoke_run_produces_outputs(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            outputs = run_caiso_shap_case_study(
                CaisoShapCaseStudyConfig(
                    model_name="mlp",
                    mlp_max_iter=100,
                    max_days=1,
                )
            )

        self.assertEqual(outputs.run_metadata["model_name"], "mlp")
        self.assertEqual(
            outputs.run_metadata["coalition_expectation_method"],
            "empirical_background_marginalization",
        )
        self.assertEqual(len(outputs.daily_shap), 1)
        self.assertEqual(len(outputs.predictive_hourly_shap), 8 * 24)
        self.assertEqual(outputs.run_metadata["explain_rows"], 1)

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

    def test_holdout_mean_impute_allows_float_replacement_for_integer_feature(
        self,
    ) -> None:
        config = CaisoShapCaseStudyConfig(
            max_days=1,
            xgb_n_estimators=2,
            xgb_max_depth=1,
            holdout_mean_impute_features=("day_of_week",),
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
        expected_replacement = float(background["day_of_week"].mean())

        self.assertAlmostEqual(
            outputs.run_metadata["holdout_feature_replacements"]["day_of_week"],
            expected_replacement,
            places=12,
        )

    def test_one_day_torch_mlp_case_study_smoke_run_produces_outputs(self) -> None:
        outputs = run_caiso_shap_case_study(
            CaisoShapCaseStudyConfig(
                model_name="torch_mlp",
                mlp_max_iter=100,
                max_days=1,
            )
        )

        self.assertEqual(outputs.run_metadata["model_name"], "torch_mlp")
        self.assertEqual(
            outputs.run_metadata["coalition_expectation_method"],
            "empirical_background_marginalization",
        )
        self.assertEqual(len(outputs.daily_shap), 1)
        self.assertEqual(len(outputs.predictive_hourly_shap), 8 * 24)
        self.assertEqual(outputs.run_metadata["explain_rows"], 1)

    def test_one_day_spo_mlp_case_study_smoke_run_produces_outputs(self) -> None:
        outputs = run_caiso_shap_case_study(
            CaisoShapCaseStudyConfig(
                model_name="spo_mlp",
                holdout_days=360,
                mlp_max_iter=1,
                max_days=1,
                spo_processes=1,
            )
        )

        self.assertEqual(outputs.run_metadata["model_name"], "spo_mlp")
        self.assertEqual(
            outputs.run_metadata["coalition_expectation_method"],
            "empirical_background_marginalization",
        )
        self.assertEqual(len(outputs.daily_shap), 1)
        self.assertEqual(len(outputs.predictive_hourly_shap), 8 * 24)
        self.assertEqual(outputs.run_metadata["explain_rows"], 1)

    def test_one_day_spo_mlp_case_study_with_mse_warm_start_records_flag(self) -> None:
        outputs = run_caiso_shap_case_study(
            CaisoShapCaseStudyConfig(
                model_name="spo_mlp",
                holdout_days=360,
                mlp_max_iter=1,
                learning_rate=1e-2,
                mse_learning_rate=5e-3,
                spo_learning_rate=2e-4,
                max_days=1,
                spo_processes=1,
                spo_warm_start_with_mse=True,
            )
        )

        self.assertEqual(outputs.run_metadata["model_name"], "spo_mlp")
        self.assertEqual(outputs.run_metadata["learning_rate"], 1e-2)
        self.assertEqual(outputs.run_metadata["mse_learning_rate"], 5e-3)
        self.assertEqual(outputs.run_metadata["spo_learning_rate"], 2e-4)
        self.assertTrue(outputs.run_metadata["spo_warm_start_with_mse"])
        self.assertEqual(len(outputs.daily_shap), 1)

    def test_one_day_generic_background_models_case_study_smoke_run_produces_outputs(self) -> None:
        for model_name in ("ridge", "svr"):
            with self.subTest(model_name=model_name):
                outputs = run_caiso_shap_case_study(
                    CaisoShapCaseStudyConfig(
                        model_name=model_name,
                        max_days=1,
                    )
                )

                self.assertEqual(outputs.run_metadata["model_name"], model_name)
                self.assertEqual(
                    outputs.run_metadata["coalition_expectation_method"],
                    "empirical_background_marginalization",
                )
                self.assertEqual(len(outputs.daily_shap), 1)
                self.assertEqual(len(outputs.predictive_hourly_shap), 8 * 24)
                self.assertEqual(outputs.run_metadata["explain_rows"], 1)


if __name__ == "__main__":
    unittest.main()
