"""Reproducible solution for Kaggle's ML Challenge CS_CLUB.

The pipeline uses only the organizer-provided train/test CSV files.  It builds
test-like folds whose validation rows are complete (matching the test set),
trains CatBoost and LightGBM candidates, tunes a conservative probability
ensemble for Macro F1, validates the output, and writes submission.csv plus a
machine-readable CV report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ID_COL = "id"
TARGET = "target_popularity"
CLASSES = np.array(["A", "B", "C", "D", "E"])
CLASS_TO_INT = {label: idx for idx, label in enumerate(CLASSES)}
ORDINAL_TO_LABEL = np.array(["E", "D", "C", "B", "A"])
ORDINAL_MAP = {label: idx for idx, label in enumerate(ORDINAL_TO_LABEL)}


@dataclass
class CVResult:
    name: str
    mean_macro_f1: float
    std_macro_f1: float
    selection_score: float
    fold_macro_f1: list[float]
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    fit_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--quick", action="store_true", help="Use 3 folds and shorter boosting runs.")
    parser.add_argument("--skip-catboost", action="store_true")
    parser.add_argument("--skip-lightgbm", action="store_true")
    parser.add_argument("--skip-ordinal", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip logistic regression after it has been benchmarked.")
    parser.add_argument("--run-secondary", action="store_true", help="Run all-row robustness CV for the winner.")
    return parser.parse_args()


def load_and_validate(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    for path in (train_path, test_path, sample_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path.resolve()}")

    train = pd.read_csv(train_path, na_values=["NA"])
    test = pd.read_csv(test_path, na_values=["NA"])
    sample = pd.read_csv(sample_path)

    assert TARGET in train and TARGET not in test
    assert train.columns.drop(TARGET).tolist() == test.columns.tolist()
    assert sample.columns.tolist() == [ID_COL, TARGET]
    assert test[ID_COL].tolist() == sample[ID_COL].tolist()
    assert train[ID_COL].is_unique and test[ID_COL].is_unique
    assert not set(train[ID_COL]).intersection(test[ID_COL])
    assert set(train[TARGET].unique()) == set(CLASSES)
    assert len(train) == 37_144 and len(test) == 2_500
    assert test.drop(columns=ID_COL).isna().sum().sum() == 0
    return train, test, sample


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.astype(float)
    result = numerator.astype(float) / denominator.where(denominator != 0)
    return result.replace([np.inf, -np.inf], np.nan)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.drop(columns=[ID_COL, TARGET], errors="ignore").copy()

    ratios = {
        "ratio_self_hrefs_hrefs": ("num_self_hrefs", "num_hrefs"),
        "ratio_imgs_content": ("num_imgs", "n_tokens_content"),
        "ratio_videos_content": ("num_videos", "n_tokens_content"),
        "ratio_keywords_content": ("num_keywords", "n_tokens_content"),
        "ratio_title_content": ("n_tokens_title", "n_tokens_content"),
    }
    for new_col, (num, den) in ratios.items():
        x[new_col] = safe_ratio(x[num], x[den])
        x[f"{new_col}_zero_den"] = x[den].eq(0).astype(np.int8)

    skewed = [
        "n_tokens_content", "num_hrefs", "num_self_hrefs", "num_imgs", "num_videos",
        "kw_max_min", "kw_avg_min", "kw_min_max", "kw_max_max", "kw_avg_max",
        "kw_min_avg", "kw_max_avg", "kw_avg_avg", "self_reference_min_shares",
        "self_reference_max_shares", "self_reference_avg_sharess",
    ]
    for col in skewed:
        x[f"log1p_{col}"] = np.log1p(x[col].clip(lower=0))

    lda_cols = [f"LDA_0{i}" for i in range(5)]
    # Preserve original NaNs for the learners, but use zero-filled topic weights
    # for aggregate features so argmax/entropy remain deterministic.
    lda = x[lda_cols].fillna(0).clip(lower=1e-12)
    x["lda_max"] = lda.max(axis=1)
    x["lda_argmax"] = lda.to_numpy().argmax(axis=1)
    x["lda_sum"] = lda.sum(axis=1)
    normalized = lda.div(lda.sum(axis=1), axis=0)
    x["lda_entropy"] = -(normalized * np.log(normalized)).sum(axis=1)

    x["word_rate_balance"] = x["global_rate_positive_words"] - x["global_rate_negative_words"]
    x["sentiment_magnitude"] = x["global_sentiment_polarity"].abs()
    x["title_sentiment_magnitude"] = x["title_sentiment_polarity"].abs()

    channel_cols = [
        "data_channel_is_lifestyle", "data_channel_is_entertainment", "data_channel_is_bus",
        "data_channel_is_socmed", "data_channel_is_tech", "data_channel_is_world",
    ]
    weekday_cols = [
        "weekday_is_monday", "weekday_is_tuesday", "weekday_is_wednesday",
        "weekday_is_thursday", "weekday_is_friday", "weekday_is_saturday",
        "weekday_is_sunday",
    ]
    x["channel_count"] = x[channel_cols].sum(axis=1)
    x["channel_code"] = np.where(x["channel_count"].gt(0), x[channel_cols].to_numpy().argmax(axis=1), -1)
    x["weekday_count"] = x[weekday_cols].sum(axis=1)
    x["weekday_code"] = np.where(x["weekday_count"].gt(0), x[weekday_cols].to_numpy().argmax(axis=1), -1)

    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def make_test_like_folds(raw_x: pd.DataFrame, y: np.ndarray, n_splits: int) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    clean_idx = np.flatnonzero(~raw_x.isna().any(axis=1).to_numpy())
    clean_y = y[clean_idx]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(len(raw_x))
    for _, clean_val_pos in splitter.split(clean_idx, clean_y):
        val_idx = clean_idx[clean_val_pos]
        train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=True)
        folds.append((train_idx, val_idx))
    return clean_idx, folds


def make_all_row_folds(y: np.ndarray, n_splits: int) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    all_idx = np.arange(len(y))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = [(tr, va) for tr, va in splitter.split(all_idx, y)]
    return all_idx, folds


def summarize_predictions(name: str, y_true: np.ndarray, y_pred: np.ndarray, fold_ids: np.ndarray, elapsed: float) -> CVResult:
    fold_scores = [f1_score(y_true[fold_ids == f], y_pred[fold_ids == f], average="macro") for f in sorted(np.unique(fold_ids))]
    report = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    per_class = {label: float(report[label]["f1-score"]) for label in CLASSES}
    mean = float(np.mean(fold_scores))
    std = float(np.std(fold_scores))
    return CVResult(
        name=name,
        mean_macro_f1=mean,
        std_macro_f1=std,
        selection_score=mean - 0.5 * std,
        fold_macro_f1=[float(v) for v in fold_scores],
        per_class_f1=per_class,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
        fit_seconds=elapsed,
    )


def baseline_logistic(x: pd.DataFrame, y: np.ndarray, eval_idx: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> CVResult:
    start = time.time()
    pos = {idx: i for i, idx in enumerate(eval_idx)}
    preds = np.empty(len(eval_idx), dtype="<U1")
    fold_ids = np.empty(len(eval_idx), dtype=int)
    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=800, solver="lbfgs", random_state=SEED)),
        ])
        model.fit(x.iloc[tr_idx], y[tr_idx])
        va_pred = model.predict(x.iloc[va_idx])
        for idx, pred in zip(va_idx, va_pred):
            p = pos[idx]
            preds[p] = pred
            fold_ids[p] = fold
    return summarize_predictions("logistic_balanced", y[eval_idx], preds, fold_ids, time.time() - start)


def catboost_cv(
    x: pd.DataFrame,
    y: np.ndarray,
    test_x: pd.DataFrame | None,
    eval_idx: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    class_weights: str,
    iterations: int,
    threads: int,
    name: str,
) -> tuple[CVResult, np.ndarray, np.ndarray | None, np.ndarray]:
    from catboost import CatBoostClassifier

    start = time.time()
    pos = {idx: i for i, idx in enumerate(eval_idx)}
    oof = np.zeros((len(eval_idx), len(CLASSES)), dtype=float)
    fold_ids = np.empty(len(eval_idx), dtype=int)
    test_prob = np.zeros((len(test_x), len(CLASSES)), dtype=float) if test_x is not None else None

    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=iterations,
            depth=7,
            learning_rate=0.05,
            l2_leaf_reg=5,
            auto_class_weights=class_weights,
            random_seed=SEED + fold,
            thread_count=threads,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x.iloc[tr_idx], y[tr_idx], eval_set=(x.iloc[va_idx], y[va_idx]), early_stopping_rounds=100, verbose=False)
        probs = model.predict_proba(x.iloc[va_idx])
        class_order = np.asarray(model.classes_)
        reorder = [int(np.where(class_order == label)[0][0]) for label in CLASSES]
        probs = probs[:, reorder]
        for row, idx in enumerate(va_idx):
            p = pos[idx]
            oof[p] = probs[row]
            fold_ids[p] = fold
        if test_prob is not None:
            test_prob += model.predict_proba(test_x)[:, reorder] / len(folds)

    pred = CLASSES[oof.argmax(axis=1)]
    result = summarize_predictions(name, y[eval_idx], pred, fold_ids, time.time() - start)
    return result, oof, test_prob, fold_ids


def lightgbm_cv(
    x: pd.DataFrame,
    y: np.ndarray,
    test_x: pd.DataFrame | None,
    eval_idx: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    iterations: int,
    threads: int,
    name: str,
) -> tuple[CVResult, np.ndarray, np.ndarray | None, np.ndarray]:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    start = time.time()
    y_int = np.array([CLASS_TO_INT[v] for v in y], dtype=int)
    pos = {idx: i for i, idx in enumerate(eval_idx)}
    oof = np.zeros((len(eval_idx), len(CLASSES)), dtype=float)
    fold_ids = np.empty(len(eval_idx), dtype=int)
    test_prob = np.zeros((len(test_x), len(CLASSES)), dtype=float) if test_x is not None else None
    n_jobs = threads if threads > 0 else -1

    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = LGBMClassifier(
            objective="multiclass",
            num_class=len(CLASSES),
            n_estimators=iterations,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            class_weight="balanced",
            random_state=SEED + fold,
            n_jobs=n_jobs,
            verbosity=-1,
        )
        model.fit(
            x.iloc[tr_idx], y_int[tr_idx],
            eval_set=[(x.iloc[va_idx], y_int[va_idx])],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        probs = model.predict_proba(x.iloc[va_idx])
        for row, idx in enumerate(va_idx):
            p = pos[idx]
            oof[p] = probs[row]
            fold_ids[p] = fold
        if test_prob is not None:
            test_prob += model.predict_proba(test_x) / len(folds)

    pred = CLASSES[oof.argmax(axis=1)]
    result = summarize_predictions(name, y[eval_idx], pred, fold_ids, time.time() - start)
    return result, oof, test_prob, fold_ids


def labels_from_ordinal(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return ORDINAL_TO_LABEL[np.digitize(values, thresholds)]


def tune_ordinal_thresholds(values: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    ordered_y = np.array([ORDINAL_MAP[v] for v in y])
    cumulative = [np.mean(ordered_y <= i) for i in range(4)]
    thresholds = np.quantile(values, cumulative)
    best = f1_score(y, labels_from_ordinal(values, thresholds), average="macro")
    for _ in range(4):
        changed = False
        for j in range(4):
            lower = thresholds[j - 1] + 1e-5 if j else float(np.nanmin(values))
            upper = thresholds[j + 1] - 1e-5 if j < 3 else float(np.nanmax(values))
            for candidate in np.linspace(lower, upper, 60):
                trial = thresholds.copy()
                trial[j] = candidate
                score = f1_score(y, labels_from_ordinal(values, trial), average="macro")
                if score > best + 1e-7:
                    thresholds, best, changed = trial, score, True
        if not changed:
            break
    return thresholds, float(best)


def ordinal_catboost_cv(
    x: pd.DataFrame,
    y: np.ndarray,
    test_x: pd.DataFrame,
    eval_idx: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    iterations: int,
    threads: int,
) -> tuple[CVResult, np.ndarray, np.ndarray]:
    from catboost import CatBoostRegressor

    start = time.time()
    y_num = np.array([ORDINAL_MAP[v] for v in y], dtype=float)
    pos = {idx: i for i, idx in enumerate(eval_idx)}
    oof = np.zeros(len(eval_idx), dtype=float)
    fold_ids = np.empty(len(eval_idx), dtype=int)
    test_pred = np.zeros(len(test_x), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=iterations,
            depth=7,
            learning_rate=0.05,
            l2_leaf_reg=5,
            random_seed=SEED + fold,
            thread_count=threads,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x.iloc[tr_idx], y_num[tr_idx], eval_set=(x.iloc[va_idx], y_num[va_idx]), early_stopping_rounds=100, verbose=False)
        values = model.predict(x.iloc[va_idx])
        for idx, value in zip(va_idx, values):
            p = pos[idx]
            oof[p] = value
            fold_ids[p] = fold
        test_pred += model.predict(test_x) / len(folds)
    thresholds, _ = tune_ordinal_thresholds(oof, y[eval_idx])
    pred = labels_from_ordinal(oof, thresholds)
    result = summarize_predictions("catboost_ordinal", y[eval_idx], pred, fold_ids, time.time() - start)
    return result, test_pred, thresholds


def tune_biases(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    logp = np.log(np.clip(prob, 1e-12, 1.0))
    biases = np.zeros(len(CLASSES), dtype=float)
    best = f1_score(y, CLASSES[np.argmax(logp + biases, axis=1)], average="macro")
    # D is the reference class; adding the same constant to all classes is redundant.
    for _ in range(4):
        changed = False
        for j, label in enumerate(CLASSES):
            if label == "D":
                continue
            for candidate in np.arange(-1.5, 1.5001, 0.1):
                trial = biases.copy()
                trial[j] = candidate
                pred = CLASSES[np.argmax(logp + trial, axis=1)]
                score = f1_score(y, pred, average="macro")
                if score > best + 1e-7:
                    biases, best, changed = trial, score, True
        if not changed:
            break
    return biases


def evaluate_adjustment(prob: np.ndarray, y: np.ndarray, fold_ids: np.ndarray) -> tuple[bool, np.ndarray, dict]:
    base_pred = CLASSES[prob.argmax(axis=1)]
    adjusted = np.empty(len(y), dtype="<U1")
    fold_deltas = []
    for fold in sorted(np.unique(fold_ids)):
        train_mask = fold_ids != fold
        val_mask = ~train_mask
        biases = tune_biases(prob[train_mask], y[train_mask])
        adjusted[val_mask] = CLASSES[np.argmax(np.log(np.clip(prob[val_mask], 1e-12, 1.0)) + biases, axis=1)]
        base = f1_score(y[val_mask], base_pred[val_mask], average="macro")
        tuned = f1_score(y[val_mask], adjusted[val_mask], average="macro")
        fold_deltas.append(float(tuned - base))
    base_macro = f1_score(y, base_pred, average="macro")
    tuned_macro = f1_score(y, adjusted, average="macro")
    base_pc = classification_report(y, base_pred, labels=CLASSES, output_dict=True, zero_division=0)
    tuned_pc = classification_report(y, adjusted, labels=CLASSES, output_dict=True, zero_division=0)
    minority_ok = all(tuned_pc[c]["f1-score"] >= base_pc[c]["f1-score"] - 0.02 for c in ["A", "B"])
    accepted = tuned_macro - base_macro >= 0.005 and min(fold_deltas) >= -0.01 and minority_ok
    final_biases = tune_biases(prob, y) if accepted else np.zeros(len(CLASSES))
    details = {
        "accepted": bool(accepted), "base_macro_f1": float(base_macro),
        "crossfit_macro_f1": float(tuned_macro), "fold_deltas": fold_deltas,
        "final_biases": final_biases.tolist(),
        "crossfit_per_class_f1": {
            label: float(tuned_pc[label]["f1-score"]) for label in CLASSES
        },
        "crossfit_confusion_matrix": confusion_matrix(y, adjusted, labels=CLASSES).tolist(),
    }
    return accepted, final_biases, details


def choose_probabilities(
    candidates: dict[str, tuple[CVResult, np.ndarray, np.ndarray]],
    y_eval: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[str, np.ndarray, np.ndarray, dict]:
    ranked = sorted(candidates.items(), key=lambda item: item[1][0].selection_score, reverse=True)
    if len(ranked) >= 2 and ranked[0][1][0].selection_score - ranked[1][1][0].selection_score < 0.003:
        cat_candidate = next((item for item in ranked[:2] if item[0].startswith("catboost")), None)
        if cat_candidate is not None and ranked[0] != cat_candidate:
            ranked[0], ranked[1] = cat_candidate, ranked[0]
    best_name, (best_result, best_oof, best_test) = ranked[0]
    selection = {"base_choice": best_name, "base_selection_score": best_result.selection_score}

    if len(ranked) >= 2:
        second_name, (_, second_oof, second_test) = ranked[1]
        base_pred = CLASSES[best_oof.argmax(axis=1)]
        base_score = f1_score(y_eval, base_pred, average="macro")
        base_report = classification_report(y_eval, base_pred, labels=CLASSES, output_dict=True, zero_division=0)
        best_score = base_score
        chosen = (best_name, best_oof, best_test, 1.0)
        ensemble_trials = []
        for weight in np.arange(0.0, 1.0001, 0.1):
            prob = weight * best_oof + (1 - weight) * second_oof
            pred = CLASSES[prob.argmax(axis=1)]
            score = f1_score(y_eval, pred, average="macro")
            fold_deltas = []
            for fold in sorted(np.unique(fold_ids)):
                mask = fold_ids == fold
                fold_deltas.append(float(
                    f1_score(y_eval[mask], pred[mask], average="macro")
                    - f1_score(y_eval[mask], base_pred[mask], average="macro")
                ))
            report = classification_report(y_eval, pred, labels=CLASSES, output_dict=True, zero_division=0)
            minority_ok = all(report[c]["f1-score"] >= base_report[c]["f1-score"] - 0.02 for c in ["A", "B"])
            acceptable = score - base_score >= 0.005 and min(fold_deltas) >= -0.01 and minority_ok
            ensemble_trials.append({
                "weight_primary": float(weight), "macro_f1": float(score),
                "fold_deltas": fold_deltas, "acceptable": bool(acceptable),
            })
            if acceptable and score > best_score:
                best_score = score
                chosen = (f"ensemble:{best_name}+{second_name}", prob, weight * best_test + (1 - weight) * second_test, float(weight))
        trial_name, trial_oof, trial_test, weight = chosen
        if trial_name.startswith("ensemble:"):
            best_name, best_oof, best_test = trial_name, trial_oof, trial_test
        selection["ensemble_trials"] = ensemble_trials
        selection["ensemble_weight_primary"] = weight

    accepted, biases, bias_details = evaluate_adjustment(best_oof, y_eval, fold_ids)
    selection["bias_adjustment"] = bias_details
    if accepted:
        best_test = np.exp(np.log(np.clip(best_test, 1e-12, 1.0)) + biases)
        best_test /= best_test.sum(axis=1, keepdims=True)
        best_name += "+bias"
    return best_name, best_oof, best_test, selection


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> None:
    assert submission.columns.tolist() == [ID_COL, TARGET]
    assert len(submission) == len(test) == 2_500
    assert submission[ID_COL].tolist() == test[ID_COL].tolist() == sample[ID_COL].tolist()
    assert submission[ID_COL].is_unique
    assert not submission.isna().any().any()
    assert set(submission[TARGET].unique()).issubset(set(CLASSES))
    assert set(submission[TARGET].unique()) == set(CLASSES), "Every class must appear in the submission."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    import sklearn
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
    for package in ("catboost", "lightgbm"):
        try:
            module = __import__(package)
            versions[package] = module.__version__
        except ImportError:
            versions[package] = "not-installed"
    return versions


def main() -> None:
    args = parse_args()
    if args.quick:
        args.folds = 3
    iterations = 700 if args.quick else 1_500
    lgb_iterations = 900 if args.quick else 2_000
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train, test, sample = load_and_validate(args.data_dir)
    y = train[TARGET].to_numpy()
    raw_x = train.drop(columns=[ID_COL, TARGET])
    x = engineer_features(train)
    test_x = engineer_features(test)
    assert x.columns.tolist() == test_x.columns.tolist()

    clean_idx, folds = make_test_like_folds(raw_x, y, args.folds)
    eval_y = y[clean_idx]
    diagnostics: dict = {
        "versions": package_versions(),
        "seed": SEED,
        "mode": "quick" if args.quick else "full",
        "train_shape": list(train.shape), "test_shape": list(test.shape),
        "clean_train_rows": int(len(clean_idx)),
        "class_counts": train[TARGET].value_counts().sort_index().to_dict(),
        "candidate_results": [],
    }

    majority = np.repeat("D", len(eval_y))
    diagnostics["constant_d_macro_f1"] = float(f1_score(eval_y, majority, average="macro"))
    if not args.skip_baseline:
        logistic = baseline_logistic(x, y, clean_idx, folds)
        diagnostics["candidate_results"].append(asdict(logistic))
        print(f"{logistic.name}: {logistic.mean_macro_f1:.5f} ± {logistic.std_macro_f1:.5f}")

    candidates: dict[str, tuple[CVResult, np.ndarray, np.ndarray]] = {}
    best_policy = "SqrtBalanced"
    fold_ids: np.ndarray | None = None
    if not args.skip_catboost:
        # Short three-fold comparison of CatBoost weighting policies.
        probe_splits = min(3, args.folds)
        probe_idx, probe_folds = make_test_like_folds(raw_x, y, probe_splits)
        probes = []
        for policy in ("SqrtBalanced", "Balanced"):
            result, _, _, _ = catboost_cv(x, y, None, probe_idx, probe_folds, policy, min(iterations, 800), args.threads, f"catboost_probe_{policy}")
            probes.append((result, policy))
            diagnostics["candidate_results"].append(asdict(result))
            print(f"{result.name}: {result.mean_macro_f1:.5f} ± {result.std_macro_f1:.5f}")
        best_policy = max(probes, key=lambda pair: pair[0].selection_score)[1]

        cat_result, cat_oof, cat_test, fold_ids = catboost_cv(
            x, y, test_x, clean_idx, folds, best_policy, iterations, args.threads, f"catboost_{best_policy}"
        )
        assert cat_test is not None
        candidates[cat_result.name] = (cat_result, cat_oof, cat_test)
        diagnostics["candidate_results"].append(asdict(cat_result))
        print(f"{cat_result.name}: {cat_result.mean_macro_f1:.5f} ± {cat_result.std_macro_f1:.5f}")

    if not args.skip_lightgbm:
        lgb_result, lgb_oof, lgb_test, lgb_fold_ids = lightgbm_cv(
            x, y, test_x, clean_idx, folds, lgb_iterations, args.threads, "lightgbm_balanced"
        )
        assert lgb_test is not None
        if fold_ids is not None:
            assert np.array_equal(fold_ids, lgb_fold_ids)
        fold_ids = lgb_fold_ids
        candidates[lgb_result.name] = (lgb_result, lgb_oof, lgb_test)
        diagnostics["candidate_results"].append(asdict(lgb_result))
        print(f"{lgb_result.name}: {lgb_result.mean_macro_f1:.5f} ± {lgb_result.std_macro_f1:.5f}")

    if not args.skip_ordinal and not args.skip_catboost:
        ord_result, ord_test, thresholds = ordinal_catboost_cv(x, y, test_x, clean_idx, folds, iterations, args.threads)
        diagnostics["candidate_results"].append(asdict(ord_result))
        diagnostics["ordinal_thresholds"] = thresholds.tolist()
        print(f"{ord_result.name}: {ord_result.mean_macro_f1:.5f} ± {ord_result.std_macro_f1:.5f}")

    if not candidates or fold_ids is None:
        raise RuntimeError("At least one boosting model must be enabled.")
    chosen_name, chosen_oof, chosen_test, selection = choose_probabilities(candidates, eval_y, fold_ids)
    diagnostics["selection"] = selection
    diagnostics["selected_model"] = chosen_name

    if args.run_secondary:
        _, all_folds = make_all_row_folds(y, args.folds)
        if chosen_name.startswith("lightgbm"):
            secondary, _, _, _ = lightgbm_cv(x, y, None, np.arange(len(y)), all_folds, lgb_iterations, args.threads, "secondary_all_rows_lightgbm")
        else:
            secondary, _, _, _ = catboost_cv(x, y, None, np.arange(len(y)), all_folds, best_policy, iterations, args.threads, "secondary_all_rows_catboost")
        diagnostics["secondary_robustness"] = asdict(secondary)

    test_labels = CLASSES[chosen_test.argmax(axis=1)]
    submission = sample.copy()
    submission[TARGET] = test_labels
    validate_submission(submission, test, sample)
    submission_path = Path("submission.csv")
    submission.to_csv(submission_path, index=False, lineterminator="\n")
    validate_submission(pd.read_csv(submission_path), test, sample)
    diagnostics["submission_sha256"] = sha256_file(submission_path)
    diagnostics["submission_class_counts"] = submission[TARGET].value_counts().sort_index().to_dict()
    diagnostics["submission_head"] = submission.head().to_dict(orient="records")
    diagnostics["submission_tail"] = submission.tail().to_dict(orient="records")

    report_path = args.output_dir / "cv_summary.json"
    report_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"Selected: {chosen_name}")
    print(f"Submission: {submission_path.resolve()}")
    print(f"SHA-256: {diagnostics['submission_sha256']}")
    print(submission[TARGET].value_counts().sort_index())


if __name__ == "__main__":
    main()
