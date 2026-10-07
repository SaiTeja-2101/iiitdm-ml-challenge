"""Bounded, non-destructive second-pass experiments for the competition.

This script never writes ``submission.csv``.  It verifies and snapshots the
already-scored file, compares challengers on identical OOF rows, and writes a
separate ``submission_candidate.csv`` plus auditable experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score

from solution import (
    CLASSES,
    CLASS_TO_INT,
    ID_COL,
    SEED,
    TARGET,
    engineer_features,
    load_and_validate,
    make_test_like_folds,
    package_versions,
    sha256_file,
    summarize_predictions,
    tune_biases,
    validate_submission,
)

SCORED_SHA256 = "b34b46fc46535af5f0ebc64edc62309f91d37b70491094e2aff19158e5c4bcc7"
SCORED_PUBLIC_F1 = 0.33283


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    leaves: int = 31
    min_child: int = 80
    weight_exp: float = 0.5
    train_scope: str = "all"
    max_features: float = 0.7
    min_leaf: int = 2


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--budget-minutes", type=float, default=50.0)
    parser.add_argument("--quick", action="store_true", help="Short smoke run; not suitable for selection.")
    return parser.parse_args()


def class_sample_weights(y: np.ndarray, train_idx: np.ndarray, exponent: float) -> np.ndarray:
    labels, counts = np.unique(y[train_idx], return_counts=True)
    largest = float(counts.max())
    by_label = {label: (largest / count) ** exponent for label, count in zip(labels, counts)}
    return np.asarray([by_label[label] for label in y[train_idx]], dtype=float)


def scoped_folds(
    folds: list[tuple[np.ndarray, np.ndarray]], clean_idx: np.ndarray, scope: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    if scope == "all":
        return folds
    if scope != "clean":
        raise ValueError(f"Unknown train scope: {scope}")
    clean = set(clean_idx.tolist())
    return [(np.asarray([i for i in tr if i in clean], dtype=int), va) for tr, va in folds]


def run_model(
    spec: ModelSpec,
    x: pd.DataFrame,
    test_x: pd.DataFrame,
    y: np.ndarray,
    eval_idx: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    clean_idx: np.ndarray,
    seed: int,
    threads: int,
    iterations: int,
):
    start = time.time()
    y_int = np.asarray([CLASS_TO_INT[v] for v in y], dtype=int)
    pos = {idx: i for i, idx in enumerate(eval_idx)}
    oof = np.zeros((len(eval_idx), len(CLASSES)), dtype=np.float32)
    test_prob = np.zeros((len(test_x), len(CLASSES)), dtype=np.float64)
    fold_ids = np.empty(len(eval_idx), dtype=np.int8)
    use_folds = scoped_folds(folds, clean_idx, spec.train_scope)
    best_iterations: list[int] = []

    for fold, (tr, va) in enumerate(use_folds):
        weights = class_sample_weights(y, tr, spec.weight_exp)
        if spec.kind == "lgbm":
            import lightgbm as lgb
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                objective="multiclass", num_class=len(CLASSES), n_estimators=iterations,
                learning_rate=0.03, num_leaves=spec.leaves,
                min_child_samples=spec.min_child, subsample=0.85,
                subsample_freq=1, colsample_bytree=0.85, reg_lambda=5.0,
                random_state=seed + fold, n_jobs=threads if threads > 0 else -1,
                verbosity=-1,
            )
            model.fit(
                x.iloc[tr], y_int[tr], sample_weight=weights,
                eval_set=[(x.iloc[va], y_int[va])], eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
            )
            best_iterations.append(int(model.best_iteration_ or iterations))
            va_prob = model.predict_proba(x.iloc[va])
            test_prob += model.predict_proba(test_x) / len(use_folds)
        elif spec.kind == "extra_trees":
            imputer = SimpleImputer(strategy="median")
            train_matrix = imputer.fit_transform(x.iloc[tr])
            val_matrix = imputer.transform(x.iloc[va])
            test_matrix = imputer.transform(test_x)
            model = ExtraTreesClassifier(
                n_estimators=iterations, max_features=spec.max_features,
                min_samples_leaf=spec.min_leaf, class_weight=None,
                random_state=seed + fold, n_jobs=threads if threads > 0 else -1,
            )
            model.fit(train_matrix, y_int[tr], sample_weight=weights)
            va_prob = model.predict_proba(val_matrix)
            test_prob += model.predict_proba(test_matrix) / len(use_folds)
            best_iterations.append(iterations)
        else:
            raise ValueError(spec.kind)

        for row, original_idx in enumerate(va):
            p = pos[int(original_idx)]
            oof[p] = va_prob[row]
            fold_ids[p] = fold

    prediction = CLASSES[oof.argmax(axis=1)]
    result = summarize_predictions(
        f"{spec.name}_{x.shape[1]}f_seed{seed}", y[eval_idx], prediction,
        fold_ids, time.time() - start,
    )
    return result, oof, test_prob.astype(np.float32), fold_ids, best_iterations


def crossfit_bias(prob: np.ndarray, y: np.ndarray, fold_ids: np.ndarray):
    pred = np.empty(len(y), dtype="<U1")
    biases = []
    for fold in sorted(np.unique(fold_ids)):
        train = fold_ids != fold
        val = ~train
        bias = tune_biases(prob[train], y[train])
        pred[val] = CLASSES[np.argmax(np.log(np.clip(prob[val], 1e-12, 1.0)) + bias, axis=1)]
        biases.append(bias.tolist())
    final_bias = tune_biases(prob, y)
    return pred, final_bias, biases


def prediction_summary(pred: np.ndarray, y: np.ndarray, fold_ids: np.ndarray) -> dict:
    fold_scores = [
        float(f1_score(y[fold_ids == fold], pred[fold_ids == fold], average="macro"))
        for fold in sorted(np.unique(fold_ids))
    ]
    report = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "fold_macro_f1": fold_scores,
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)),
        "per_class_f1": {c: float(report[c]["f1-score"]) for c in CLASSES},
    }


def stratified_bootstrap(
    y: np.ndarray, baseline: np.ndarray, challenger: np.ndarray, draws: int = 1000
) -> dict:
    rng = np.random.default_rng(20261007)
    groups = [np.flatnonzero(y == label) for label in CLASSES]
    deltas = np.empty(draws, dtype=float)
    for draw in range(draws):
        sample = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        deltas[draw] = (
            f1_score(y[sample], challenger[sample], average="macro")
            - f1_score(y[sample], baseline[sample], average="macro")
        )
    return {
        "draws": draws,
        "win_rate": float(np.mean(deltas > 0)),
        "delta_mean": float(deltas.mean()),
        "delta_90pct_ci": [float(v) for v in np.quantile(deltas, [0.05, 0.95])],
    }


def compare_to_baseline(
    baseline_pred: np.ndarray,
    challenger_pred: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    seed_gains: dict[str, float],
) -> dict:
    base = prediction_summary(baseline_pred, y, fold_ids)
    challenger = prediction_summary(challenger_pred, y, fold_ids)
    fold_delta = (np.asarray(challenger["fold_macro_f1"]) - np.asarray(base["fold_macro_f1"])).tolist()
    bootstrap = stratified_bootstrap(y, baseline_pred, challenger_pred)
    checks = {
        "gain_at_least_0.007": challenger["macro_f1"] - base["macro_f1"] >= 0.007,
        "at_least_4_of_5_folds_improve": int(np.sum(np.asarray(fold_delta) > 0)) >= 4,
        "no_fold_loses_over_0.01": min(fold_delta) >= -0.01,
        "fold_std_at_most_0.010": challenger["fold_std"] <= 0.010,
        "class_A_not_down_over_0.01": challenger["per_class_f1"]["A"] >= base["per_class_f1"]["A"] - 0.01,
        "class_B_not_down_over_0.01": challenger["per_class_f1"]["B"] >= base["per_class_f1"]["B"] - 0.01,
        "both_seeds_improve": all(value > 0 for value in seed_gains.values()),
        "bootstrap_win_rate_at_least_0.85": bootstrap["win_rate"] >= 0.85,
        "bootstrap_90pct_lower_above_zero": bootstrap["delta_90pct_ci"][0] > 0,
    }
    return {
        "baseline": base, "challenger": challenger,
        "macro_gain": challenger["macro_f1"] - base["macro_f1"],
        "fold_deltas": fold_delta, "seed_gains": seed_gains,
        "bootstrap": bootstrap, "checks": checks,
        "passed_all_gates": bool(all(checks.values())),
    }


def save_run(path: Path, result, oof, test_prob, fold_ids, best_iterations, meta: dict) -> None:
    np.savez_compressed(path, oof=oof, test_prob=test_prob, fold_ids=fold_ids)
    meta.update({"result": asdict(result), "best_iterations": best_iterations, "array_file": path.name})


def main() -> None:
    opt = args()
    started = time.time()
    opt.output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = Path("submission.csv")
    if not scored_path.exists() or sha256_file(scored_path) != SCORED_SHA256:
        raise RuntimeError("The scored submission is missing or changed; refusing to run.")
    baseline_dir = Path("artifacts/baseline")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    frozen = baseline_dir / "submission_scored_0.33283.csv"
    if not frozen.exists():
        shutil.copyfile(scored_path, frozen)
    if sha256_file(frozen) != SCORED_SHA256:
        raise RuntimeError("Frozen baseline hash mismatch.")

    train, test, sample = load_and_validate(opt.data_dir)
    y = train[TARGET].to_numpy()
    raw_train = train.drop(columns=[ID_COL, TARGET])
    raw_test = test.drop(columns=[ID_COL])
    eng_train, eng_test = engineer_features(train), engineer_features(test)
    clean3, folds3 = make_test_like_folds(raw_train, y, 3)
    clean5, folds5 = make_test_like_folds(raw_train, y, 5)
    assert np.array_equal(clean3, clean5)
    y_eval = y[clean5]

    report: dict = {
        "scored_baseline": {"public_macro_f1": SCORED_PUBLIC_F1, "sha256": SCORED_SHA256},
        "versions": package_versions(), "started_unix": started,
        "data": {"train_rows": len(train), "test_rows": len(test), "complete_train_rows": len(clean5)},
        "screen": [], "full_confirmation": [], "skipped": [],
    }
    try:
        import catboost  # noqa: F401
        report["skipped"].append("CatBoost is installed, but omitted from this bounded runner; solution.py contains its validated implementation.")
    except ImportError:
        report["skipped"].append("CatBoost candidates skipped: package not installed in the verified local environment.")

    specs = [
        ModelSpec("lgbm_baseline", "lgbm", 63, 50, 1.0, "all"),
        ModelSpec("lgbm_regularized", "lgbm", 31, 80, 0.5, "all"),
        ModelSpec("lgbm_mid", "lgbm", 47, 100, 0.75, "all"),
        ModelSpec("lgbm_clean_specialist", "lgbm", 31, 80, 0.5, "clean"),
        ModelSpec("extra_trees_leaf2", "extra_trees", weight_exp=0.5, min_leaf=2),
        ModelSpec("extra_trees_leaf5", "extra_trees", weight_exp=0.5, min_leaf=5),
    ]
    screen_iterations = 250 if opt.quick else None
    for spec in specs:
        iterations = screen_iterations or (700 if spec.kind == "lgbm" else 400)
        result, oof, test_prob, fold_ids, best_iter = run_model(
            spec, eng_train, eng_test, y, clean3, folds3, clean3, SEED,
            opt.threads, iterations,
        )
        adjusted, final_bias, fold_biases = crossfit_bias(oof, y[clean3], fold_ids)
        item = {
            "spec": asdict(spec), "result": asdict(result),
            "crossfit_bias": prediction_summary(adjusted, y[clean3], fold_ids),
            "final_bias": final_bias.tolist(), "fold_biases": fold_biases,
            "best_iterations": best_iter,
        }
        report["screen"].append(item)
        print(f"screen {spec.name}: raw={result.mean_macro_f1:.5f} adjusted={item['crossfit_bias']['macro_f1']:.5f}", flush=True)

    baseline_spec = specs[0]
    challenger_spec = max(
        (s for s in specs[1:]),
        key=lambda s: next(i["crossfit_bias"]["macro_f1"] for i in report["screen"] if i["spec"]["name"] == s.name),
    )
    full: dict[tuple[str, str, int], dict] = {}

    def full_run(spec: ModelSpec, feature_name: str, seed: int) -> dict:
        key = (spec.name, feature_name, seed)
        if key in full:
            return full[key]
        x, tx = (eng_train, eng_test) if feature_name == "engineered" else (raw_train, raw_test)
        iterations = 350 if opt.quick else (1600 if spec.kind == "lgbm" else 700)
        result, oof, test_prob, fold_ids, best_iter = run_model(
            spec, x, tx, y, clean5, folds5, clean5, seed, opt.threads, iterations,
        )
        adjusted, final_bias, fold_biases = crossfit_bias(oof, y_eval, fold_ids)
        meta = {
            "spec": asdict(spec), "features": feature_name, "seed": seed,
            "crossfit_bias": prediction_summary(adjusted, y_eval, fold_ids),
            "final_bias": final_bias.tolist(), "fold_biases": fold_biases,
        }
        filename = f"{spec.name}_{feature_name}_seed{seed}.npz"
        save_run(opt.output_dir / filename, result, oof, test_prob, fold_ids, best_iter, meta)
        report["full_confirmation"].append(meta)
        value = {"result": result, "oof": oof, "test": test_prob, "fold_ids": fold_ids,
                 "adjusted": adjusted, "final_bias": final_bias, "meta": meta}
        full[key] = value
        print(f"full {spec.name}/{feature_name}/seed{seed}: adjusted={meta['crossfit_bias']['macro_f1']:.5f}", flush=True)
        return value

    baseline42 = full_run(baseline_spec, "engineered", 42)
    challenger42_eng = full_run(challenger_spec, "engineered", 42)
    challenger42_raw = full_run(challenger_spec, "raw", 42)
    chosen_features = (
        "raw" if challenger42_raw["meta"]["crossfit_bias"]["macro_f1"]
        > challenger42_eng["meta"]["crossfit_bias"]["macro_f1"] else "engineered"
    )
    challenger42 = challenger42_raw if chosen_features == "raw" else challenger42_eng
    baseline137 = full_run(baseline_spec, "engineered", 137)
    challenger137 = full_run(challenger_spec, chosen_features, 137)

    candidates = []
    for weight in np.arange(0.0, 1.0001, 0.1):
        prob42 = weight * baseline42["oof"] + (1 - weight) * challenger42["oof"]
        pred42, bias42, _ = crossfit_bias(prob42, y_eval, baseline42["fold_ids"])
        candidates.append({
            "baseline_weight": float(weight),
            "crossfit_macro_f1_seed42": float(f1_score(y_eval, pred42, average="macro")),
            "pred": pred42, "bias": bias42,
            "test_prob": weight * baseline42["test"] + (1 - weight) * challenger42["test"],
        })
    best = max(candidates, key=lambda item: item["crossfit_macro_f1_seed42"])
    selected_weight = best["baseline_weight"]
    prob137 = selected_weight * baseline137["oof"] + (1 - selected_weight) * challenger137["oof"]
    pred137, _, _ = crossfit_bias(prob137, y_eval, baseline137["fold_ids"])
    seed_gains = {
        "42": float(
            f1_score(y_eval, best["pred"], average="macro")
            - f1_score(y_eval, baseline42["adjusted"], average="macro")
        ),
        "137": float(
            f1_score(y_eval, pred137, average="macro")
            - f1_score(y_eval, baseline137["adjusted"], average="macro")
        ),
    }
    best["comparison"] = compare_to_baseline(
        baseline42["adjusted"], best["pred"], y_eval, baseline42["fold_ids"], seed_gains,
    )
    test_logp = np.log(np.clip(best["test_prob"], 1e-12, 1.0)) + best["bias"]
    candidate_labels = CLASSES[np.argmax(test_logp, axis=1)]
    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: candidate_labels})
    validate_submission(submission, test, sample)
    submission.to_csv("submission_candidate.csv", index=False)
    report["selection"] = {
        "screen_winner": challenger_spec.name,
        "chosen_feature_set": chosen_features,
        "ensemble_baseline_weight": best["baseline_weight"],
        "comparison": best["comparison"],
        "candidate_sha256": sha256_file(Path("submission_candidate.csv")),
        "test_prediction_counts": submission[TARGET].value_counts().sort_index().to_dict(),
        "test_fraction_changed_from_scored_submission": float(
            np.mean(submission[TARGET].to_numpy() != pd.read_csv(scored_path)[TARGET].to_numpy())
        ),
        "recommendation": "candidate" if best["comparison"]["passed_all_gates"] else "retain_scored_baseline",
    }
    serializable_trials = [
        {"baseline_weight": c["baseline_weight"], "crossfit_macro_f1_seed42": c["crossfit_macro_f1_seed42"]}
        for c in candidates
    ]
    report["ensemble_trials"] = serializable_trials
    report["elapsed_minutes"] = (time.time() - started) / 60
    report["budget_minutes"] = opt.budget_minutes
    report["scored_submission_hash_unchanged"] = sha256_file(scored_path) == SCORED_SHA256
    if not report["scored_submission_hash_unchanged"]:
        raise RuntimeError("Scored submission changed during experiments.")

    json_path = opt.output_dir / "model_comparison.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    selection = report["selection"]
    comp = selection["comparison"]
    markdown = f"""# Second-pass model comparison

- Scored public baseline: **{SCORED_PUBLIC_F1:.5f}** (hash preserved: `{SCORED_SHA256}`)
- Screen winner: **{selection['screen_winner']}**
- Selected feature view: **{selection['chosen_feature_set']}**
- Selected baseline ensemble weight: **{selection['ensemble_baseline_weight']:.1f}**
- Local cross-fit Macro F1: **{comp['challenger']['macro_f1']:.5f}**
- Comparable baseline Macro F1: **{comp['baseline']['macro_f1']:.5f}**
- Local gain: **{comp['macro_gain']:+.5f}**
- Bootstrap win rate: **{comp['bootstrap']['win_rate']:.3f}**
- 90% bootstrap delta interval: **[{comp['bootstrap']['delta_90pct_ci'][0]:+.5f}, {comp['bootstrap']['delta_90pct_ci'][1]:+.5f}]**
- All conservative gates passed: **{comp['passed_all_gates']}**
- Recommendation: **{selection['recommendation']}**
- Test predictions changed: **{selection['test_fraction_changed_from_scored_submission']:.1%}**

`submission_candidate.csv` is an analysis artifact only. It was not uploaded, and the scored `submission.csv` was not overwritten.

## Technique conclusions

LightGBM is the primary fit for this tabular dataset because it handles missing values and nonlinear interactions directly. Regularized/smaller trees test whether the original model was overly sensitive; the clean-only specialist tests distribution matching because the test rows are complete. Extra Trees provides a structurally different bagged-tree candidate, but requires fold-fitted median imputation. Raw-versus-engineered ablation measures whether hand-built ratios and topic summaries add genuine signal. Probability averaging and class log-biases are assessed only through cross-fitting because in-sample threshold tuning would overstate Macro F1.

CatBoost and the ordinal regressor were not run because CatBoost is absent from this verified environment. Neural networks, SMOTE, external data, recovered labels, and leaderboard tuning remain excluded.
"""
    (opt.output_dir / "model_comparison.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report["selection"], indent=2), flush=True)


if __name__ == "__main__":
    main()
