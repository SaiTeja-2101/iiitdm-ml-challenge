# ML Challenge CS_CLUB — Online News Popularity

This repository contains a reproducible, competition-compliant pipeline for predicting the five `target_popularity` classes. The official selection metric is **Macro F1**, so model selection uses per-class F1 and stratified out-of-fold predictions rather than accuracy or training score.

## Competition constraints

- Only the organizer-provided `train.csv` and `test.csv` are used.
- No external dataset, hidden labels, pretrained model, target encoding, or leaderboard probing is used.
- `id` is retained only for the submission and is excluded from modeling.
- All random operations use seed `42`.

## Validation design

The test set has no missing predictor values, while 42.5% of training rows contain at least one missing value. The primary validation therefore partitions the complete training rows into stratified folds. In each fold, validation contains only complete rows and training contains every other row, including incomplete rows. This mimics the observable test distribution while still using all permitted training examples.

An optional secondary five-fold validation over every training row is available with `--run-secondary`.

For every candidate, the report records fold Macro F1, mean and standard deviation, per-class F1, a confusion matrix, runtime, and the conservative selection score:

```text
mean Macro F1 - 0.5 * fold standard deviation
```

## Models and features

The pipeline compares:

1. Median-imputed, standardized, class-balanced multinomial logistic regression.
2. CatBoost multiclass with a short comparison of `SqrtBalanced` and `Balanced` class weights.
3. Class-balanced LightGBM multiclass.
4. An ordinal CatBoost regressor using `E < D < C < B < A`, with thresholds tuned only from out-of-fold predictions.

The deterministic feature function adds safe content ratios, log transforms for skewed count/share features, LDA concentration and entropy, sentiment balances, and decoded channel/weekday values. Original predictors remain available to the models. Missing values are handled natively by the boosting models.

CatBoost and LightGBM probabilities are blended only if the out-of-fold gain is at least 0.005 Macro F1, no fold loses more than 0.01, and minority-class performance remains stable. Additive class biases are evaluated with the same gates using cross-fitted tuning.

## Data placement

Do not commit competition data. Put these files together in a local directory:

```text
train.csv
test.csv
sample_submission.csv
```

On Kaggle, the usual path is `/kaggle/input/ml-challenge-cs-club`. Locally, the commands below use the downloaded competition directory.

## Reproduce

Python 3.12 was used for local verification. Create an isolated environment and install the pinned direct dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the full five-fold pipeline:

```powershell
.\.venv\Scripts\python.exe solution.py `
  --data-dir "C:\Users\saite\Downloads\ml-challenge-cs-club" `
  --output-dir artifacts
```

For a faster smoke test:

```powershell
.\.venv\Scripts\python.exe solution.py `
  --data-dir "C:\Users\saite\Downloads\ml-challenge-cs-club" `
  --output-dir artifacts `
  --quick
```

The program writes:

- `submission.csv` — final `id,target_popularity` predictions in test order.
- `artifacts/cv_summary.json` — versions, validation metrics, model-selection evidence, prediction class counts, and submission SHA-256.

## Verified local result

The supplied `submission.csv` was generated with five test-like folds using LightGBM plus the accepted cross-fitted class-bias adjustment. CatBoost remains implemented for the Kaggle runtime but was not executed locally because its 102 MB Windows wheel was unavailable within the local verification window.

- LightGBM fold Macro F1: `0.31398 ± 0.00646`
- Cross-fitted bias-adjusted Macro F1: `0.32132`
- Bias-adjusted per-class F1: A `0.1820`, B `0.1955`, C `0.2931`, D `0.4119`, E `0.5240`
- Submission class counts: A `246`, B `389`, C `464`, D `767`, E `634`
- Submission SHA-256: `b34b46fc46535af5f0ebc64edc62309f91d37b70491094e2aff19158e5c4bcc7`

Exact command for the supplied artifact:

```powershell
.\.venv\Scripts\python.exe solution.py `
  --data-dir "C:\Users\saite\Downloads\ml-challenge-cs-club" `
  --output-dir artifacts `
  --skip-catboost --skip-ordinal --skip-baseline
```

The complete metrics, confusion matrices, fold scores, versions, final biases, and first/last submission records are stored in `artifacts/cv_summary.json`.

## Submission checks

The program stops unless all of the following hold:

- Train and test schemas match after excluding the target.
- Test IDs equal the sample-submission IDs in the same order.
- There are exactly 2,500 unique test IDs.
- Output columns are exactly `id,target_popularity`.
- Every prediction is one of `A`, `B`, `C`, `D`, or `E` and every class appears.
- The saved CSV can be read back and passes the same checks.

Before the team’s one Kaggle submission, rerun from a clean session and compare the SHA-256 in `artifacts/cv_summary.json` with the file being uploaded.
