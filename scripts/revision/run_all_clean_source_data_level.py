"""Strict target-specific clean-source rerun for DL-1, DL-2, and DL-3.

For every target endpoint and outer repetition, every source endpoint is
restricted to its outer-training indices and every row whose canonical SMILES
appears in the target outer-test set is removed before pair selection, feature
construction, tuning, or model fitting.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from clean_source_common import (
    CLEAN_RESULTS,
    ENDPOINTS,
    add_common_fields,
    baseline_probability,
    clean_train_indices,
    evaluate,
    load_inputs,
    quick_lgb_model,
    read_json,
    removal_audit,
    result_key,
    smiles_array,
    tune_lgb_model,
    write_json_atomic,
)


def target_arrays(datasets, splits, target, rep):
    train, test = splits[target][rep]
    return (
        datasets[target]["X"][train],
        datasets[target]["y"][train],
        datasets[target]["X"][test],
        datasets[target]["y"][test],
        smiles_array(datasets, target)[train],
        smiles_array(datasets, target)[test],
    )


def run_dl1(datasets, splits, baseline, target, rep, max_evals):
    seed = 101 + rep
    x_train, y_train, x_test, y_test, target_train_smiles, _ = target_arrays(
        datasets, splits, target, rep
    )
    target_train_map = {
        smile: int(label) for smile, label in zip(target_train_smiles, y_train)
    }
    target_full = set(smiles_array(datasets, target).tolist())
    rules = []
    augmented_x = []
    augmented_y = []
    augmented_groups = []
    for source in ENDPOINTS:
        if source == target:
            continue
        source_indices = clean_train_indices(datasets, splits, source, target, rep)
        source_smiles = smiles_array(datasets, source)[source_indices]
        source_labels = datasets[source]["y"][source_indices]
        source_map = {
            smile: int(label) for smile, label in zip(source_smiles, source_labels)
        }
        overlap = set(source_map) & set(target_train_map)
        for source_label in (0, 1):
            target_labels = [
                target_train_map[smile]
                for smile in overlap
                if source_map[smile] == source_label
            ]
            if len(target_labels) < 10:
                continue
            count_0 = target_labels.count(0)
            count_1 = target_labels.count(1)
            dominant = 0 if count_0 >= count_1 else 1
            fraction = max(count_0, count_1) / len(target_labels)
            if fraction < 0.90:
                continue
            rules.append(
                {
                    "source": source,
                    "source_label": source_label,
                    "target_label": dominant,
                    "n_overlap_train_only": len(target_labels),
                    "dominant_fraction": fraction,
                }
            )
            source_x = datasets[source]["X"][source_indices]
            for row, smile, label in zip(source_x, source_smiles, source_labels):
                if smile in target_full or int(label) != source_label:
                    continue
                augmented_x.append(row)
                augmented_y.append(dominant)
                augmented_groups.append(smile)
    removed = removal_audit(datasets, splits, target, rep)
    if not augmented_x:
        probability = baseline_probability(baseline, target, rep)
        metrics = evaluate(y_test, probability)
        metrics.update({"n_rules": len(rules), "n_added": 0, "used_baseline": True})
    else:
        x_merged = np.vstack([x_train, np.asarray(augmented_x)])
        y_merged = np.concatenate([y_train, np.asarray(augmented_y, dtype=y_train.dtype)])
        groups = np.concatenate(
            [np.asarray(target_train_smiles, dtype=str), np.asarray(augmented_groups, dtype=str)]
        )
        model, params = tune_lgb_model(
            x_merged,
            y_merged,
            seed,
            max_evals=max_evals,
            n_folds=5,
            groups=groups,
        )
        probability = model.predict_proba(x_test)[:, 1]
        metrics = evaluate(y_test, probability)
        metrics.update(
            {
                "n_rules": len(rules),
                "n_added": len(augmented_x),
                "used_baseline": False,
                "best_params": params,
            }
        )
    return add_common_fields(metrics, baseline, target, rep, removed, "DL1_clean", seed)


def run_dl2(datasets, splits, baseline, target, rep, max_evals):
    seed = 101 + rep
    _x_train, _y_train, x_test, y_test, _target_train_smiles, _ = target_arrays(
        datasets, splits, target, rep
    )
    pooled_x = []
    pooled_y = []
    pooled_groups = []
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        indices = clean_train_indices(datasets, splits, endpoint, target, rep)
        task_id = np.zeros(len(ENDPOINTS), dtype=np.uint8)
        task_id[endpoint_index] = 1
        x = datasets[endpoint]["X"][indices]
        pooled_x.append(np.hstack([x, np.tile(task_id, (len(indices), 1))]))
        pooled_y.append(datasets[endpoint]["y"][indices])
        pooled_groups.extend(smiles_array(datasets, endpoint)[indices].tolist())
    x_merged = np.vstack(pooled_x)
    y_merged = np.concatenate(pooled_y)
    model, params = tune_lgb_model(
        x_merged,
        y_merged,
        seed,
        max_evals=max_evals,
        n_folds=5,
        groups=np.asarray(pooled_groups, dtype=str),
    )
    target_index = ENDPOINTS.index(target)
    target_id = np.zeros(len(ENDPOINTS), dtype=np.uint8)
    target_id[target_index] = 1
    x_test_task = np.hstack([x_test, np.tile(target_id, (len(x_test), 1))])
    probability = model.predict_proba(x_test_task)[:, 1]
    removed = removal_audit(datasets, splits, target, rep)
    metrics = evaluate(y_test, probability)
    metrics.update(
        {
            "n_training_rows": int(len(y_merged)),
            "n_unique_training_smiles": int(len(set(pooled_groups))),
            "best_params": params,
        }
    )
    return add_common_fields(metrics, baseline, target, rep, removed, "DL2_clean", seed)


def shap_profile(model, x_train):
    contributions = model.predict(x_train, pred_contrib=True)[:, :-1]
    mean_signed = contributions.mean(axis=0)
    mean_abs = np.abs(contributions).mean(axis=0)
    top = np.argsort(mean_abs)[::-1][:10]
    return mean_signed, mean_abs, top


def run_dl3(datasets, splits, baseline, target, rep, max_evals):
    seed = 101 + rep
    x_target, y_target, x_test, y_test, target_train_smiles, _ = target_arrays(
        datasets, splits, target, rep
    )
    profiles = {}
    clean_indices = {}
    for endpoint in ENDPOINTS:
        indices = clean_train_indices(datasets, splits, endpoint, target, rep)
        clean_indices[endpoint] = indices
        model = quick_lgb_model(
            datasets[endpoint]["X"][indices],
            datasets[endpoint]["y"][indices],
            seed,
        )
        profiles[endpoint] = shap_profile(model, datasets[endpoint]["X"][indices])
    target_signed, _target_abs, target_top = profiles[target]
    candidates = []
    target_set = set(target_top.tolist())
    for source in ENDPOINTS:
        if source == target:
            continue
        source_signed, _source_abs, source_top = profiles[source]
        shared = target_set & set(source_top.tolist())
        if not shared:
            continue
        concordance = sum(
            int(np.sign(target_signed[index]) == np.sign(source_signed[index]))
            for index in shared
        ) / len(shared)
        if len(shared) >= 3 and concordance >= 2.0 / 3.0:
            candidates.append((len(shared), concordance, source))
    removed = removal_audit(datasets, splits, target, rep)
    if not candidates:
        probability = baseline_probability(baseline, target, rep)
        metrics = evaluate(y_test, probability)
        metrics.update(
            {
                "selected_source": None,
                "shared_top10": 0,
                "directional_concordance": math.nan,
                "used_baseline": True,
            }
        )
        return add_common_fields(metrics, baseline, target, rep, removed, "DL3_clean", seed)
    shared_count, directional_concordance, source = sorted(
        candidates, key=lambda item: (-item[0], -item[1], item[2])
    )[0]
    source_indices = clean_indices[source]
    x_source = datasets[source]["X"][source_indices]
    y_source = datasets[source]["y"][source_indices]
    source_smiles = smiles_array(datasets, source)[source_indices]
    target_id = np.tile(np.asarray([1, 0], dtype=np.uint8), (len(x_target), 1))
    source_id = np.tile(np.asarray([0, 1], dtype=np.uint8), (len(x_source), 1))
    x_merged = np.vstack(
        [np.hstack([x_target, target_id]), np.hstack([x_source, source_id])]
    )
    y_merged = np.concatenate([y_target, y_source])
    groups = np.concatenate(
        [np.asarray(target_train_smiles, dtype=str), np.asarray(source_smiles, dtype=str)]
    )
    model, params = tune_lgb_model(
        x_merged,
        y_merged,
        seed,
        max_evals=max_evals,
        n_folds=5,
        groups=groups,
    )
    x_test_task = np.hstack(
        [x_test, np.tile(np.asarray([1, 0], dtype=np.uint8), (len(x_test), 1))]
    )
    probability = model.predict_proba(x_test_task)[:, 1]
    metrics = evaluate(y_test, probability)
    metrics.update(
        {
            "selected_source": source,
            "shared_top10": int(shared_count),
            "directional_concordance": float(directional_concordance),
            "used_baseline": False,
            "best_params": params,
        }
    )
    return add_common_fields(metrics, baseline, target, rep, removed, "DL3_clean", seed)


RUNNERS = {"dl1": run_dl1, "dl2": run_dl2, "dl3": run_dl3}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["dl1", "dl2", "dl3", "all"], default="all")
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=10)
    parser.add_argument("--target", choices=ENDPOINTS)
    parser.add_argument("--max-evals", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets, splits, baseline = load_inputs()
    methods = list(RUNNERS) if args.method == "all" else [args.method]
    targets = [args.target] if args.target else ENDPOINTS
    for method in methods:
        default_evals = 50
        max_evals = args.max_evals or default_evals
        suffix = "_smoke" if args.smoke else ""
        path = CLEAN_RESULTS / f"{method}_clean_results{suffix}.json"
        payload = read_json(path)
        for rep in range(args.rep_start, min(args.rep_end, 10)):
            for target in targets:
                key = result_key(target, rep)
                if key in payload:
                    continue
                started = time.time()
                result = RUNNERS[method](
                    datasets, splits, baseline, target, rep, max_evals
                )
                payload[key] = result
                write_json_atomic(payload, path)
                print(
                    f"{method} rep={rep} target={target} "
                    f"AUC={result['AUC']:.4f} delta={result['delta_AUC']:+.4f} "
                    f"removed={result['source_rows_removed']} "
                    f"seconds={time.time()-started:.1f}",
                    flush=True,
                )


if __name__ == "__main__":
    import math

    main()
