"""Shared utilities for strict target-specific clean-source experiments."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from safe_project_pickle import safe_load


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CLEAN_RESULTS = ROOT / "results" / "revision" / "clean_source_independent"
CLEAN_RESULTS.mkdir(parents=True, exist_ok=True)

ENDPOINTS = [
    "prenatal_development_C",
    "TSHR_agonist_activity_C",
    "respiratory_toxicity_C",
    "ocular_toxicity_C",
    "ames_mutagenicity_C",
    "reproductive_toxicity_C",
    "skin_corrosion_C",
    "neurotoxicity_C",
    "Estrogen_Receptor_α_C",
    "Androgen_Receptor_C",
    "cytotoxicity_C",
    "Carcinogenicity_C",
    "Hepatotoxicity_C",
]

N_REPS = 10

CHOICES = {
    "n_estimators": [100, 200, 300, 500, 700, 1000],
    "max_depth": [3, 5, 7, 9, 11, -1],
    "num_leaves": [15, 31, 63, 127, 255],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "min_child_samples": [5, 10, 20, 30, 50],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.01, 0.1, 1.0],
    "reg_lambda": [0, 0.01, 0.1, 1.0],
}


def load_inputs():
    datasets = safe_load(RESULTS / "datasets.pkl")
    splits = safe_load(RESULTS / "scaffold_splits_record.pkl")
    baseline = safe_load(RESULTS / "scaffold_baseline_results.pkl")
    if list(datasets) != ENDPOINTS:
        raise AssertionError("Endpoint order differs from the prespecified order")
    return datasets, splits, baseline


def smiles_array(datasets, endpoint):
    return np.asarray(datasets[endpoint]["smiles"], dtype=str)


def target_test_smiles(datasets, splits, target, rep):
    _train, test = splits[target][rep]
    return set(smiles_array(datasets, target)[test].tolist())


def clean_train_indices(datasets, splits, endpoint, target, rep, excluded_extra=None):
    train, _test = splits[endpoint][rep]
    train = np.asarray(train, dtype=int)
    excluded = set(target_test_smiles(datasets, splits, target, rep))
    if excluded_extra:
        excluded.update(str(item) for item in excluded_extra)
    if endpoint == target and not excluded_extra:
        kept = train
    else:
        source_smiles = smiles_array(datasets, endpoint)[train]
        kept = train[~np.isin(source_smiles, list(excluded))]
    overlap = set(smiles_array(datasets, endpoint)[kept].tolist()) & excluded
    if overlap:
        raise AssertionError(f"Leakage remains for source={endpoint}, target={target}, rep={rep}")
    return kept


def removal_audit(datasets, splits, target, rep, excluded_extra=None):
    removed = {}
    target_test = target_test_smiles(datasets, splits, target, rep)
    for endpoint in ENDPOINTS:
        train, _ = splits[endpoint][rep]
        clean = clean_train_indices(datasets, splits, endpoint, target, rep, excluded_extra)
        removed[endpoint] = int(len(train) - len(clean))
        if endpoint != target:
            clean_smiles = set(smiles_array(datasets, endpoint)[clean].tolist())
            if clean_smiles & target_test:
                raise AssertionError("Post-cleaning source/test identity overlap")
    return removed


def calibration_stats(y, probability):
    probability = np.asarray(probability, dtype=float)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    if np.unique(y).size < 2:
        return math.nan, math.nan, math.nan
    calibrator = LogisticRegression(C=1e6, solver="lbfgs")
    calibrator.fit(logit, y)
    edges = np.linspace(0, 1, 11)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            ece += mask.mean() * abs(float(np.mean(y[mask])) - float(np.mean(probability[mask])))
    return float(calibrator.intercept_[0]), float(calibrator.coef_[0, 0]), float(ece)


def evaluate(y, probability):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    intercept, slope, ece = calibration_stats(y, probability)
    return {
        "AUC": float(roc_auc_score(y, probability)),
        "PR_AUC": float(average_precision_score(y, probability)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, predicted)),
        "Sensitivity": float(tp / (tp + fn)) if tp + fn else math.nan,
        "Specificity": float(tn / (tn + fp)) if tn + fp else math.nan,
        "Brier": float(brier_score_loss(y, probability)),
        "Calibration_Intercept": intercept,
        "Calibration_Slope": slope,
        "ECE_10": ece,
        "y_pred": probability.astype(float).tolist(),
    }


def quick_lgb_predict(x_train, y_train, x_test, seed, n_jobs=8):
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=7,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def quick_lgb_model(x_train, y_train, seed, n_jobs=8):
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=7,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train)
    return model


def tune_lgb_model(x_train, y_train, seed, max_evals=15, n_folds=3, groups=None, n_jobs=8):
    space = {key: hp.choice(key, values) for key, values in CHOICES.items()}
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = list(splitter.split(x_train, y_train, groups))
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = list(splitter.split(x_train, y_train))

    def objective(params):
        scores = []
        for train_index, valid_index in folds:
            if np.unique(y_train[valid_index]).size < 2:
                scores.append(0.5)
                continue
            model = lgb.LGBMClassifier(
                **params,
                random_state=seed,
                verbose=-1,
                n_jobs=n_jobs,
            )
            model.fit(x_train[train_index], y_train[train_index])
            scores.append(roc_auc_score(y_train[valid_index], model.predict_proba(x_train[valid_index])[:, 1]))
        return {"loss": -float(np.mean(scores)), "status": STATUS_OK}

    trials = Trials()
    best_indices = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(seed),
        verbose=False,
    )
    params = {key: CHOICES[key][int(index)] for key, index in best_indices.items()}
    model = lgb.LGBMClassifier(
        **params,
        random_state=seed,
        verbose=-1,
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train)
    return model, params


def baseline_probability(baseline, endpoint, rep):
    return np.asarray(baseline[endpoint][rep]["y_pred"], dtype=float)


def result_key(endpoint, rep):
    return f"{endpoint}|{rep}"


def read_json(path):
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(payload, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temp, path)


def add_common_fields(metrics, baseline, endpoint, rep, removed, method, seed):
    metrics.update(
        {
            "endpoint": endpoint,
            "rep": int(rep),
            "seed": int(seed),
            "method": method,
            "baseline_AUC": float(baseline[endpoint][rep]["AUC"]),
            "delta_AUC": float(metrics["AUC"] - baseline[endpoint][rep]["AUC"]),
            "source_rows_removed": int(sum(value for key, value in removed.items() if key != endpoint)),
            "max_source_test_overlap_after": 0,
        }
    )
    return metrics
