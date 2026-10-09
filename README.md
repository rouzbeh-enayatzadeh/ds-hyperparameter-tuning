# Hyperparameter Tuning

How to choose the settings a model does not learn for itself.

## Learning path

| Notebook | What it covers |
|---|---|
| `1_manual_tuning.ipynb` | parameters vs hyperparameters, `get_params()` / `set_params()`, tuning by hand, and why a train/validation/test split is needed |
| `2_grid_search.ipynb` | `GridSearchCV`, `best_params_` / `best_estimator_`, reading `cv_results_` as a leaderboard, and the parallel-coordinate view of a search |
| `3_randomized_search.ipynb` | `RandomizedSearchCV`, sampling distributions, `n_iter`, and when random beats exhaustive |
| `4_nested_cross_validation.ipynb` | tuning *inside* an evaluation loop, and why a tuned model's `best_score_` is not an estimate of its performance |
| `5_hyperparameter_tuning_exercise.ipynb` | practice (solution in `solutions/`) |

## Prerequisites

Cross-validation from the **Bias-Variance Tradeoff** lesson, and pipelines from
**Feature Engineering**. Both are used from the first notebook onward.

## Dataset

`data/penguins.csv` — the Palmer Penguins, carried over from the Feature Engineering lesson
so the preprocessing pipeline is already familiar. The task is predicting `species`.

## Setup

```bash
uv sync
```
