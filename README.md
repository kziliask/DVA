# DVA

Decision Value Attribution (DVA) experiments for the CAISO storage and EMS staging
case studies. This subproject is intended to be the online-ready package version of
the original working repo.

## Layout

- `src/dva/attribution.py`: exact Shapley, perturbation/permutation SHAP, kernel
  SHAP, Shapley-Taylor, and Faith-SHAP helpers.
- `src/dva/games.py`: shared InfoDVA, DesignDVA, JointDVA, and DVI game contracts.
- `src/dva/case_studies/caiso`: CAISO storage experiment runners.
- `src/dva/case_studies/ems`: EMS staging experiment runners.
- `data/`: processed, runner-ready datasets only.
- `scripts/cluster`: static PBS and shell launchers for heavier experiments.

## Quickstart

```bash
uv sync
uv run pytest -q
uv run dva-caiso-gdsi --dry-run
uv run dva-caiso-joint-dvi --baseline conservative --value-mode post --max-days 1 --dry-run
uv run dva-ems-infodva --dry-run
uv run dva-ems-design-joint-dvi --analysis-kind joint_dvi --solver naive --value-mode ante --dry-run
```

Cluster scripts source `scripts/cluster/env.sh`, run through `uv`, and write logs
under `logs/cluster`. EMS launchers default to `OPTIMIZATION_SOLVER=highs` and
`SOLVER_THREADS=1`; setting `OPTIMIZATION_SOLVER=gurobi` syncs the optional
Gurobi extra before the job runs. Regenerate the static launchers with:

```bash
uv run python scripts/generate_cluster_scripts.py
```
