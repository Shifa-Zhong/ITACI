"""Strict target-specific clean-source stacking with nested feature construction."""

from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from clean_source_common import (
    CLEAN_RESULTS,
    ENDPOINTS,
    add_common_fields,
    baseline_probability,
    clean_train_indices,
    evaluate,
    load_inputs,
    quick_lgb_model,
    quick_lgb_predict,
    read_json,
    removal_audit,
    result_key,
    smiles_array,
    write_json_atomic,
)
from run_all_clean_source_neural import scaffold_groups


CANDIDATES = (
    dict(n_estimators=200, max_depth=5, num_leaves=31, learning_rate=0.05, min_child_samples=20),
    dict(n_estimators=300, max_depth=7, num_leaves=63, learning_rate=0.05, min_child_samples=20),
    dict(n_estimators=500, max_depth=7, num_leaves=63, learning_rate=0.03, min_child_samples=10),
    dict(n_estimators=300, max_depth=9, num_leaves=127, learning_rate=0.05, min_child_samples=20),
    dict(n_estimators=500, max_depth=-1, num_leaves=63, learning_rate=0.03, min_child_samples=30),
    dict(n_estimators=700, max_depth=5, num_leaves=31, learning_rate=0.01, min_child_samples=10),
)


def train_source_models(datasets, splits, target, rep, excluded_extra=None, seed=101):
    models = []
    for source in ENDPOINTS:
        if source == target:
            continue
        indices = clean_train_indices(
            datasets,
            splits,
            source,
            target,
            rep,
            excluded_extra=excluded_extra,
        )
        models.append(
            quick_lgb_model(
                datasets[source]["X"][indices],
                datasets[source]["y"][indices],
                seed,
            )
        )
    return models


def augmented_features(models, fingerprints):
    auxiliary = np.column_stack(
        [model.predict_proba(fingerprints)[:, 1] for model in models]
    )
    return np.hstack([fingerprints, auxiliary])


def fit_candidate(params, x_train, y_train, seed):
    model = lgb.LGBMClassifier(
        **params,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
        n_jobs=8,
    )
    model.fit(x_train, y_train)
    return model


def run_one(datasets, splits, baseline, target, rep, smoke=False):
    seed = 101 + rep
    train_index, test_index = splits[target][rep]
    fingerprints = datasets[target]["X"]
    labels = datasets[target]["y"]
    target_smiles = smiles_array(datasets, target)
    x_train = fingerprints[train_index]
    y_train = labels[train_index]
    x_test = fingerprints[test_index]
    y_test = labels[test_index]
    groups = scaffold_groups(target_smiles[train_index])
    n_folds = 2 if smoke else 3
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(splitter.split(x_train, y_train, groups))
    fold_data = []
    for fold, (inner_train, inner_valid) in enumerate(folds):
        validation_smiles = set(target_smiles[train_index][inner_valid].tolist())
        source_models = train_source_models(
            datasets,
            splits,
            target,
            rep,
            excluded_extra=validation_smiles,
            seed=seed + fold,
        )
        fold_data.append(
            (
                inner_train,
                inner_valid,
                augmented_features(source_models, x_train[inner_train]),
                augmented_features(source_models, x_train[inner_valid]),
            )
        )
    candidates = CANDIDATES[:1] if smoke else CANDIDATES
    candidate_scores = []
    for candidate in candidates:
        scores = []
        for fold, (inner_train, inner_valid, train_augmented, valid_augmented) in enumerate(fold_data):
            model = fit_candidate(candidate, train_augmented, y_train[inner_train], seed + fold)
            scores.append(
                roc_auc_score(
                    y_train[inner_valid], model.predict_proba(valid_augmented)[:, 1]
                )
            )
        candidate_scores.append(float(np.mean(scores)))
    best_index = int(np.argmax(candidate_scores))
    best_params = dict(candidates[best_index])
    oof_stacking = np.zeros(len(train_index), dtype=float)
    oof_baseline = np.zeros(len(train_index), dtype=float)
    for fold, (inner_train, inner_valid, train_augmented, valid_augmented) in enumerate(fold_data):
        model = fit_candidate(best_params, train_augmented, y_train[inner_train], seed + fold)
        oof_stacking[inner_valid] = model.predict_proba(valid_augmented)[:, 1]
        oof_baseline[inner_valid] = quick_lgb_predict(
            x_train[inner_train], y_train[inner_train], x_train[inner_valid], seed + fold
        )
    best_lambda = 0.5
    best_auc = -np.inf
    for weight in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        auc = roc_auc_score(
            y_train, weight * oof_baseline + (1 - weight) * oof_stacking
        )
        if auc > best_auc:
            best_auc = auc
            best_lambda = weight
    outer_source_models = train_source_models(
        datasets, splits, target, rep, seed=seed
    )
    outer_train_augmented = augmented_features(outer_source_models, x_train)
    outer_test_augmented = augmented_features(outer_source_models, x_test)
    final_stacking = fit_candidate(best_params, outer_train_augmented, y_train, seed)
    stacking_probability = final_stacking.predict_proba(outer_test_augmented)[:, 1]
    base_probability = baseline_probability(baseline, target, rep)
    probability = best_lambda * base_probability + (1 - best_lambda) * stacking_probability
    removed = removal_audit(datasets, splits, target, rep)
    metrics = evaluate(y_test, probability)
    metrics.update(
        {
            "lambda": float(best_lambda),
            "inner_selection_AUC": float(best_auc),
            "best_candidate_index": best_index,
            "best_params": best_params,
            "inner_source_models_retrained": True,
        }
    )
    return add_common_fields(
        metrics, baseline, target, rep, removed, "S1_stacking_clean", seed
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=10)
    parser.add_argument("--target", choices=ENDPOINTS)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets, splits, baseline = load_inputs()
    suffix = "_smoke" if args.smoke else ""
    path = CLEAN_RESULTS / f"stacking_clean_results{suffix}.json"
    payload = read_json(path)
    targets = [args.target] if args.target else ENDPOINTS
    for rep in range(args.rep_start, min(args.rep_end, 10)):
        for target in targets:
            key = result_key(target, rep)
            if key in payload:
                continue
            started = time.time()
            result = run_one(datasets, splits, baseline, target, rep, args.smoke)
            payload[key] = result
            write_json_atomic(payload, path)
            print(
                f"stacking rep={rep} target={target} delta={result['delta_AUC']:+.4f} "
                f"removed={result['source_rows_removed']} seconds={time.time()-started:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
