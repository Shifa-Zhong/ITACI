"""Primary DL-1/DL-2/DL-3 runs on independent nested scaffold-group folds."""

from __future__ import annotations

import math
import sys

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import run_all_clean_source_data_level as runner
from clean_source_independent_common_v3 import (
    INDEPENDENT_RESULTS,
    evaluate_clipped,
    load_independent_inputs,
    outer_metadata,
)


PARAM_CANDIDATES = (
    dict(n_estimators=250, max_depth=5, num_leaves=31, learning_rate=0.05,
         min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
         reg_alpha=0.0, reg_lambda=0.0),
    dict(n_estimators=500, max_depth=7, num_leaves=63, learning_rate=0.03,
         min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
         reg_alpha=0.0, reg_lambda=0.1),
    dict(n_estimators=350, max_depth=9, num_leaves=127, learning_rate=0.05,
         min_child_samples=10, subsample=0.9, colsample_bytree=0.7,
         reg_alpha=0.01, reg_lambda=0.1),
    dict(n_estimators=700, max_depth=5, num_leaves=31, learning_rate=0.02,
         min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
         reg_alpha=0.1, reg_lambda=1.0),
    dict(n_estimators=300, max_depth=-1, num_leaves=63, learning_rate=0.05,
         min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
         reg_alpha=0.1, reg_lambda=0.1),
    dict(n_estimators=500, max_depth=11, num_leaves=127, learning_rate=0.03,
         min_child_samples=50, subsample=0.7, colsample_bytree=0.9,
         reg_alpha=1.0, reg_lambda=0.1),
)


def nested_candidate_model(x_train, y_train, seed, max_evals=6, n_folds=3, groups=None, n_jobs=8):
    if groups is None:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        folds = list(splitter.split(x_train, y_train))
    else:
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        folds = list(splitter.split(x_train, y_train, groups))
    scores = []
    for candidate_id, params in enumerate(PARAM_CANDIDATES):
        fold_scores = []
        for train_index, valid_index in folds:
            if np.unique(y_train[valid_index]).size < 2:
                continue
            model = lgb.LGBMClassifier(
                **params, random_state=seed + candidate_id, verbose=-1, n_jobs=n_jobs
            )
            model.fit(x_train[train_index], y_train[train_index])
            fold_scores.append(
                roc_auc_score(
                    y_train[valid_index], model.predict_proba(x_train[valid_index])[:, 1]
                )
            )
        scores.append(float(np.mean(fold_scores)) if fold_scores else 0.5)
    best_index = int(np.argmax(scores))
    best_params = dict(PARAM_CANDIDATES[best_index])
    final_model = lgb.LGBMClassifier(
        **best_params, random_state=seed, verbose=-1, n_jobs=n_jobs
    )
    final_model.fit(x_train, y_train)
    best_params["candidate_index"] = best_index
    best_params["inner_fold_auc"] = scores
    return final_model, best_params


def with_outer_metadata(function):
    def wrapped(datasets, splits, baseline, target, rep, max_evals):
        result = function(datasets, splits, baseline, target, rep, max_evals)
        result.update(outer_metadata(rep))
        return result
    return wrapped


runner.load_inputs = load_independent_inputs
runner.CLEAN_RESULTS = INDEPENDENT_RESULTS
runner.tune_lgb_model = nested_candidate_model
runner.evaluate = evaluate_clipped
runner.math = math
runner.RUNNERS = {
    "dl1": with_outer_metadata(runner.run_dl1),
    "dl2": with_outer_metadata(runner.run_dl2),
    "dl3": with_outer_metadata(runner.run_dl3),
}


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    runner.main()
