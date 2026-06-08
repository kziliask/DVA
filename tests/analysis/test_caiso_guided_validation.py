from __future__ import annotations

import unittest

from dva.analysis.run_caiso_decision_shap_guided_validation import (
    _build_config_for_record,
    _filter_manifest_by_model_family,
    make_model_manifest,
)


class CaisoGuidedValidationManifestTests(unittest.TestCase):
    def test_model_manifest_contains_xgb_grid_only(self) -> None:
        manifest = make_model_manifest()
        rows_by_id = {
            str(row["model_id"]): row
            for row in manifest.to_dict(orient="records")
        }

        self.assertEqual(len(manifest), 25)
        self.assertEqual(set(manifest["model_name"]), {"xgb"})
        self.assertEqual(rows_by_id["xgb_001"]["n_estimators"], 100)
        self.assertEqual(rows_by_id["xgb_001"]["max_depth"], 3)
        self.assertEqual(rows_by_id["xgb_001"]["learning_rate"], 0.05)
        self.assertNotIn("nn_001", rows_by_id)

    def test_model_family_filter_keeps_xgb_rows(self) -> None:
        manifest = make_model_manifest()

        all_manifest = _filter_manifest_by_model_family(manifest, "all")
        xgb_manifest = _filter_manifest_by_model_family(manifest, "xgb")

        self.assertEqual(len(all_manifest), 25)
        self.assertEqual(len(xgb_manifest), 25)
        self.assertEqual(set(xgb_manifest["model_name"]), {"xgb"})

    def test_model_family_filter_rejects_archived_nn_aliases(self) -> None:
        manifest = make_model_manifest()

        with self.assertRaisesRegex(ValueError, "all, xgb"):
            _filter_manifest_by_model_family(manifest, "nn")

    def test_config_from_model_record_uses_xgb_knobs(self) -> None:
        record = make_model_manifest().iloc[0]
        config = _build_config_for_record(
            record.to_dict(),
            model_dir=record["model_id"],
            dataset_path="data.csv",
            background_days=30,
            throughput_penalty=5.0,
            training_verbose=False,
            compute_ante_infodva=False,
            energy_capacity=None,
            power_limit=None,
            charge_efficiency=None,
            discharge_efficiency=None,
            initial_state_of_charge=None,
            terminal_state_of_charge=None,
        )

        self.assertEqual(config.model_name, "xgb")
        self.assertEqual(config.xgb_n_estimators, 100)
        self.assertEqual(config.xgb_max_depth, 3)
        self.assertEqual(config.xgb_learning_rate, 0.05)
        self.assertEqual(config.background_days, 30)


if __name__ == "__main__":
    unittest.main()
