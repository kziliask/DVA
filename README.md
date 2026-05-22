# Decision Value Attribution

Code for "Decision-Value Attribution in Predict-then-Optimize Systems" and the associated CAISO storage-dispatch and EMS staging case studies.

## Repository Layout

- `src/dva/attribution.py`: exact Shapley, Faith-SHAP/DVI,
  Shapley-Taylor, kernel SHAP, and permutation SHAP helpers.
- `src/dva/games.py`: InfoDVA, DesignDVA, JointDVA, and DVI game contracts.
- `src/dva/case_studies/caiso`: CAISO storage runners and design definitions.
- `src/dva/case_studies/ems`: EMS staging runners, designs, and model manifests.
- `src/dva/analysis`: experiment implementations, metrics, and bootstrap tools.
- `src/dva/plots`: plotting scripts used for paper figures.
- `data/cleaned`: processed CAISO price/weather table.
- `data/ems_data/processed`: processed EMS feature, target, geography, and
  distance-matrix inputs.
- `tests`: unit and other tests for attribution, optimization, runners, and
  output normalization.

## Setup

Prerequisites: Python 3.13 and `uv`.

```bash
uv sync --frozen
uv run pytest -q
```

The default optimization backend is HiGHS. To use Gurobi instead, install the
optional extra and pass `--optimization-solver gurobi` to EMS runners:

```bash
uv sync --frozen --extra gurobi
```

## Rerun The Baselines

Outputs are written under `results/` by default. Add `--overwrite` when replacing
an existing run. Most runners accept repeated `--model-id` arguments; omit
`--model-id` where supported to run the full configured XGBoost grid.

CAISO GDSI / InfoDVA:

```bash
uv run dva-caiso-gdsi --model-id xgb_012 --max-workers 1 --overwrite
```

CAISO JointDVA and DVI:

```bash
uv run dva-caiso-joint-dvi --model-id xgb_012 --baseline conservative --value-mode post
uv run dva-caiso-joint-dvi --model-id xgb_012 --baseline optimistic --value-mode ante
uv run dva-caiso-joint-dvi --model-id xgb_012 --target conservative --value-mode post
```

EMS decision baselines:

```bash
uv run dva-ems-baseline-experiment --model-id xgb_023 --regime 1:3 --overwrite
```

EMS InfoDVA over the 3-by-3 exact-solver design grid:

```bash
uv run dva-ems-infodva --model-id xgb_023 --radius 1 --radius 2 --radius 3 --staging 3 --staging 5 --staging 8 --overwrite
```

EMS DesignDVA and JointDVA:

```bash
uv run dva-ems-design-joint-dvi --analysis-kind designdva --model-id xgb_023 --solver greedy --value-mode post --overwrite
uv run dva-ems-design-joint-dvi --analysis-kind joint_dvi --model-id xgb_023 --solver naive --value-mode ante --optimization-solver highs --overwrite
```

EMS compute-time and approximation baselines:

```bash
uv run dva-ems-compute-benchmark --solver exact --solver greedy --max-hours 10
uv run dva-ems-kernel-permutation-benchmark --method permutation --method kernel --max-hours 10
```

Ranking and confidence-interval summaries:

```bash
uv run dva-additional-decision-ranking-baselines --all-models
uv run dva-validation-ranked-test-auc
uv run dva-paired-bootstrap-ci --input path/to/per_unit_metrics.csv --input-format long --reference-method DecisionSHAP
```

## Use The Methods

The attribution and game utilities can also be used directly:

```python
import numpy as np

from dva.attribution import compute_dvi_values, compute_exact_shapley_values
from dva.games import build_design_players, build_joint_players

coalition_values = np.array([0.0, 1.0, 2.0, 4.0])
shapley = compute_exact_shapley_values(coalition_values, feature_count=2)
dvi = compute_dvi_values(coalition_values, player_count=2, order=2)

design_players = build_design_players(
    actual={"capacity": 4, "power": 1},
    baseline={"capacity": 2, "power": 0.5},
)
players = build_joint_players(["temperature", "load"], design_players)
```

Case-study APIs are exposed through `dva.analysis`, including
`run_caiso_shap_case_study`, `run_ems_exact_shap`, and the decision-metric
helpers in `dva.analysis.evaluation_metrics`.

## Cluster Runs

Static PBS/shell launchers are generated through the following script:

```bash
uv run python scripts/generate_cluster_scripts.py
```

Generated launchers use `scripts/cluster/env.sh`, run `uv sync --frozen`,
write logs under `logs/cluster`, and write experiment artifacts under
`results/`. Set `OPTIMIZATION_SOLVER=gurobi` for Gurobi-backed EMS jobs; the
cluster environment will sync the optional Gurobi extra automatically.
