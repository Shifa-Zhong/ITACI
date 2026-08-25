"""Fully nested primary FEFA rerun under strict clean-source evaluation.

For every target outer fold, each inner validation fold receives embeddings
from an encoder trained after removing both the outer target-test identities
and that inner fold's target-validation identities from every endpoint pool.
The same fold-specific embeddings are used to select the LightGBM
hyperparameters and the baseline/FEFA ensemble weight.  Only after those
choices are fixed is an outer encoder trained on the complete cleaned outer
training pool and used to fit the final FEFA model.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import run_all_clean_source_neural as neural
from clean_source_common import (
    CHOICES,
    ENDPOINTS,
    add_common_fields,
    baseline_probability,
    evaluate,
    quick_lgb_predict,
    read_json,
    removal_audit,
    result_key,
    smiles_array,
    write_json_atomic,
)
from clean_source_independent_common_v3 import (
    INDEPENDENT_RESULTS,
    evaluate_clipped,
    load_independent_inputs,
    outer_metadata,
)


neural.evaluate = evaluate_clipped


def fit_lgb(x_train, y_train, params, seed):
    model = lgb.LGBMClassifier(
        **params,
        random_state=int(seed),
        verbose=-1,
        n_jobs=8,
    )
    model.fit(x_train, y_train)
    return model


def tune_on_fold_specific_embeddings(fold_cache, seed, max_evals):
    """Choose one parameter set using only fully nested inner-fold features."""
    space = {key: hp.choice(key, values) for key, values in CHOICES.items()}

    def objective(params):
        scores = []
        for fold, cache in enumerate(fold_cache):
            model = fit_lgb(
                cache["x_train_augmented"],
                cache["y_train"],
                params,
                seed + fold,
            )
            probability = model.predict_proba(cache["x_valid_augmented"])[:, 1]
            scores.append(roc_auc_score(cache["y_valid"], probability))
        return {"loss": -float(np.mean(scores)), "status": STATUS_OK}

    trials = Trials()
    best_indices = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=int(max_evals),
        trials=trials,
        rstate=np.random.default_rng(seed),
        verbose=False,
    )
    params = {key: CHOICES[key][int(index)] for key, index in best_indices.items()}
    best_inner_auc = -float(trials.best_trial["result"]["loss"])
    return params, float(best_inner_auc)


def run_one(
    datasets,
    splits,
    baseline,
    target,
    rep,
    encoder_epochs=100,
    encoder_patience=5,
    n_inner_folds=3,
    max_evals=15,
):
    seed = 101 + int(rep)
    train_index, test_index = splits[target][rep]
    target_x = datasets[target]["X"].astype(np.float32)
    target_y = np.asarray(datasets[target]["y"], dtype=int)
    target_smiles = smiles_array(datasets, target)
    x_train = target_x[train_index]
    y_train = target_y[train_index]
    x_test = target_x[test_index]
    y_test = target_y[test_index]

    inner_groups = neural.scaffold_groups(target_smiles[train_index])
    splitter = StratifiedGroupKFold(
        n_splits=n_inner_folds,
        shuffle=True,
        random_state=seed,
    )
    fold_cache = []
    oof_baseline = np.zeros(len(train_index), dtype=float)
    inner_audit = []

    for fold, (inner_train_position, inner_valid_position) in enumerate(
        splitter.split(x_train, y_train, inner_groups)
    ):
        inner_validation_smiles = set(
            target_smiles[train_index][inner_valid_position].tolist()
        )
        pool_x, pool_y, pool_smiles = neural.multitask_pool(
            datasets,
            splits,
            target,
            rep,
            excluded_extra=inner_validation_smiles,
        )
        if set(pool_smiles) & inner_validation_smiles:
            raise AssertionError("Inner validation identity remains in encoder pool")
        inner_encoder, epochs_run = neural.train_encoder(
            pool_x,
            pool_y,
            seed + 1000 + fold,
            max_epochs=encoder_epochs,
            patience=encoder_patience,
        )
        inner_train_x = x_train[inner_train_position]
        inner_valid_x = x_train[inner_valid_position]
        inner_train_y = y_train[inner_train_position]
        inner_valid_y = y_train[inner_valid_position]
        inner_train_embeddings = neural.embeddings(inner_encoder, inner_train_x)
        inner_valid_embeddings = neural.embeddings(inner_encoder, inner_valid_x)
        fold_cache.append(
            {
                "train_position": inner_train_position,
                "valid_position": inner_valid_position,
                "x_train_augmented": np.hstack(
                    [inner_train_x, inner_train_embeddings]
                ).astype(np.float32, copy=False),
                "x_valid_augmented": np.hstack(
                    [inner_valid_x, inner_valid_embeddings]
                ).astype(np.float32, copy=False),
                "y_train": inner_train_y,
                "y_valid": inner_valid_y,
            }
        )
        oof_baseline[inner_valid_position] = quick_lgb_predict(
            inner_train_x,
            inner_train_y,
            inner_valid_x,
            seed + fold,
        )
        inner_audit.append(
            {
                "fold": int(fold + 1),
                "target_train_n": int(len(inner_train_position)),
                "target_validation_n": int(len(inner_valid_position)),
                "encoder_pool_n": int(len(pool_smiles)),
                "excluded_validation_identities": int(len(inner_validation_smiles)),
                "validation_identity_overlap_after": 0,
                "encoder_epochs": int(epochs_run),
            }
        )
        del inner_encoder, pool_x, pool_y
        gc.collect()
        if neural.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    best_params, inner_tuning_auc = tune_on_fold_specific_embeddings(
        fold_cache,
        seed,
        max_evals,
    )
    oof_fefa = np.zeros(len(train_index), dtype=float)
    for fold, cache in enumerate(fold_cache):
        model = fit_lgb(
            cache["x_train_augmented"],
            cache["y_train"],
            best_params,
            seed + fold,
        )
        oof_fefa[cache["valid_position"]] = model.predict_proba(
            cache["x_valid_augmented"]
        )[:, 1]
    ensemble_weight, inner_ensemble_auc = neural.select_lambda(
        y_train,
        oof_baseline,
        oof_fefa,
    )

    outer_pool_x, outer_pool_y, outer_pool_smiles = neural.multitask_pool(
        datasets,
        splits,
        target,
        rep,
    )
    outer_encoder, outer_epochs = neural.train_encoder(
        outer_pool_x,
        outer_pool_y,
        seed,
        max_epochs=encoder_epochs,
        patience=encoder_patience,
    )
    outer_train_embeddings = neural.embeddings(outer_encoder, x_train)
    outer_test_embeddings = neural.embeddings(outer_encoder, x_test)
    final_model = fit_lgb(
        np.hstack([x_train, outer_train_embeddings]),
        y_train,
        best_params,
        seed,
    )
    transfer_probability = final_model.predict_proba(
        np.hstack([x_test, outer_test_embeddings])
    )[:, 1]
    probability = (
        ensemble_weight * baseline_probability(baseline, target, rep)
        + (1.0 - ensemble_weight) * transfer_probability
    )

    removed = removal_audit(datasets, splits, target, rep)
    metrics = evaluate(y_test, probability)
    metrics.update(
        {
            "lambda": float(ensemble_weight),
            "inner_selection_AUC": float(inner_ensemble_auc),
            "inner_hyperparameter_AUC": float(inner_tuning_auc),
            "outer_pool_size": int(len(outer_pool_smiles)),
            "outer_encoder_epochs": int(outer_epochs),
            "inner_encoder_epochs": [item["encoder_epochs"] for item in inner_audit],
            "inner_fold_audit": inner_audit,
            "best_params": best_params,
            "fully_nested_feature_encoder_for_hyperparameter_selection": True,
            "fully_nested_feature_encoder_for_ensemble_weight_selection": True,
            "inner_grouping": "Bemis-Murcko scaffold",
            "outer_protocol": "two independent randomizations x five scaffold-group folds",
        }
    )
    metrics = add_common_fields(
        metrics,
        baseline,
        target,
        rep,
        removed,
        "FEFA_clean_fully_nested",
        seed,
    )
    del outer_encoder, outer_pool_x, outer_pool_y, fold_cache
    gc.collect()
    if neural.DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def merge_shards(output_path: Path, shard_count: int):
    merged = {}
    for shard_id in range(shard_count):
        shard_path = output_path.with_name(
            f"{output_path.stem}_shard_{shard_id}{output_path.suffix}"
        )
        payload = read_json(shard_path)
        for key, record in payload.items():
            if key in merged and merged[key] != record:
                raise AssertionError(f"Conflicting duplicate FEFA record: {key}")
            merged[key] = record
    if len(merged) != 130:
        raise AssertionError(f"Expected 130 FEFA records, found {len(merged)}")
    if any(int(record["max_source_test_overlap_after"]) != 0 for record in merged.values()):
        raise AssertionError("Nonzero source/test identity overlap in merged FEFA")
    if not all(
        record.get("fully_nested_feature_encoder_for_hyperparameter_selection") is True
        and record.get("fully_nested_feature_encoder_for_ensemble_weight_selection") is True
        for record in merged.values()
    ):
        raise AssertionError("Fully nested FEFA audit flag is incomplete")
    write_json_atomic(merged, output_path)
    print(f"merged {output_path.name}: {len(merged)}/130; fully nested; max overlap=0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--target", choices=ENDPOINTS)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=10)
    parser.add_argument("--encoder-epochs", type=int, default=100)
    parser.add_argument("--encoder-patience", type=int, default=5)
    parser.add_argument("--max-evals", type=int, default=15)
    parser.add_argument("--output-name", default="fefa_clean_results_fully_nested.json")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.shard_count:
        raise ValueError("shard-id must be in [0, shard-count)")
    output_path = INDEPENDENT_RESULTS / args.output_name
    if args.merge_only:
        merge_shards(output_path, args.shard_count)
        return
    if args.shard_count > 1:
        output_path = output_path.with_name(
            f"{output_path.stem}_shard_{args.shard_id}{output_path.suffix}"
        )
    payload = read_json(output_path)
    datasets, splits, baseline = load_independent_inputs()
    targets = [args.target] if args.target else ENDPOINTS
    for rep in range(args.rep_start, min(args.rep_end, 10)):
        for target_index, target in enumerate(ENDPOINTS):
            if target not in targets:
                continue
            task_index = rep * len(ENDPOINTS) + target_index
            if task_index % args.shard_count != args.shard_id:
                continue
            key = result_key(target, rep)
            if key in payload:
                continue
            started = time.time()
            result = run_one(
                datasets,
                splits,
                baseline,
                target,
                rep,
                encoder_epochs=args.encoder_epochs,
                encoder_patience=args.encoder_patience,
                max_evals=args.max_evals,
            )
            result.update(outer_metadata(rep))
            payload[key] = result
            write_json_atomic(payload, output_path)
            print(
                f"fully_nested_fefa shard={args.shard_id} rep={rep} target={target} "
                f"delta={result['delta_AUC']:+.4f} lambda={result['lambda']:.1f} "
                f"removed={result['source_rows_removed']} seconds={time.time()-started:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    main()
