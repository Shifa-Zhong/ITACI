"""Leakage-free cross-endpoint diagnostic on independent scaffold-group folds."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from clean_source_common import (
    ENDPOINTS,
    clean_train_indices,
    quick_lgb_model,
    result_key,
    smiles_array,
    write_json_atomic,
)
from clean_source_independent_common_v3 import (
    INDEPENDENT_RESULTS,
    load_independent_inputs,
    outer_metadata,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=10)
    parser.add_argument("--target", choices=ENDPOINTS)
    args = parser.parse_args()
    datasets, splits, _baseline = load_independent_inputs()
    path = INDEPENDENT_RESULTS / "cross_prediction_clean_results.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    targets = [args.target] if args.target else ENDPOINTS
    for rep in range(args.rep_start, min(args.rep_end, 10)):
        for target in targets:
            _target_train, target_test = splits[target][rep]
            x_test = datasets[target]["X"][target_test]
            y_test = datasets[target]["y"][target_test]
            test_smiles = set(smiles_array(datasets, target)[target_test].tolist())
            for source in ENDPOINTS:
                key = f"{source}|{target}|{rep}"
                if key in payload:
                    continue
                started = time.time()
                source_original_train, _ = splits[source][rep]
                source_train = clean_train_indices(datasets, splits, source, target, rep)
                clean_smiles = set(smiles_array(datasets, source)[source_train].tolist())
                overlap = clean_smiles & test_smiles
                if overlap:
                    raise AssertionError(f"Cross-prediction leakage: {source} -> {target}, {rep}")
                model = quick_lgb_model(
                    datasets[source]["X"][source_train],
                    datasets[source]["y"][source_train],
                    seed=101 + rep,
                )
                probability = model.predict_proba(x_test)[:, 1]
                auc = float(roc_auc_score(y_test, probability))
                payload[key] = {
                    "source": source,
                    "target": target,
                    "rep": rep,
                    **outer_metadata(rep),
                    "AUC": auc,
                    "source_rows_before": int(len(source_original_train)),
                    "source_rows_after": int(len(source_train)),
                    "source_rows_removed": int(len(source_original_train) - len(source_train)),
                    "source_target_test_overlap_after": 0,
                }
                write_json_atomic(payload, path)
                print(
                    f"cross rep={rep} {source}->{target} AUC={auc:.4f} "
                    f"removed={payload[key]['source_rows_removed']} "
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
