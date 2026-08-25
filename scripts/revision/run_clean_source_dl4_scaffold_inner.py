"""Rerun DL-4 with true Bemis-Murcko scaffold-group inner CV."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from clean_source_common import (
    ENDPOINTS,
    add_common_fields,
    clean_train_indices,
    read_json,
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
from run_all_clean_source_neural import scaffold_groups
from run_clean_source_data_level_independent import nested_candidate_model


def run_one(datasets, splits, baseline, target, rep):
    seed = 101 + rep
    train_index, test_index = splits[target][rep]
    x_target = datasets[target]["X"]
    y_target = datasets[target]["y"]
    target_smiles = smiles_array(datasets, target)
    matched_rows = sum(
        len(clean_train_indices(datasets, splits, endpoint, target, rep))
        for endpoint in ENDPOINTS
    )
    rng = np.random.default_rng(seed + 8140)
    sampled_local = rng.choice(len(train_index), size=matched_rows, replace=True)
    sampled_index = np.asarray(train_index, dtype=int)[sampled_local]
    x_bootstrap = x_target[sampled_index]
    y_bootstrap = y_target[sampled_index]
    groups = scaffold_groups(target_smiles[sampled_index])
    model, parameters = nested_candidate_model(
        x_bootstrap, y_bootstrap, seed, groups=groups, n_jobs=8
    )
    probability = model.predict_proba(x_target[test_index])[:, 1]
    metrics = evaluate_clipped(y_target[test_index], probability)
    metrics.update(
        {
            "n_bootstrap_rows": int(matched_rows),
            "n_unique_target_rows_sampled": int(np.unique(sampled_index).size),
            "duplicate_fraction": float(1 - np.unique(sampled_index).size / matched_rows),
            "best_params": parameters,
            "inner_grouping": "Bemis-Murcko scaffold",
            **outer_metadata(rep),
        }
    )
    return add_common_fields(metrics, baseline, target, rep, {}, "DL4_bootstrap", seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=10)
    parser.add_argument("--target", choices=ENDPOINTS)
    args = parser.parse_args()
    datasets, splits, baseline = load_independent_inputs()
    path = INDEPENDENT_RESULTS / "dl4_clean_results_v2.json"
    payload = read_json(path)
    targets = [args.target] if args.target else ENDPOINTS
    for rep in range(args.rep_start, min(args.rep_end, 10)):
        for target in targets:
            key = result_key(target, rep)
            if key in payload:
                continue
            started = time.time()
            result = run_one(datasets, splits, baseline, target, rep)
            payload[key] = result
            write_json_atomic(payload, path)
            print(
                f"dl4_v2 rep={rep} target={target} delta={result['delta_AUC']:+.4f} "
                f"rows={result['n_bootstrap_rows']} duplicate={result['duplicate_fraction']:.3f} "
                f"seconds={time.time()-started:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    main()
