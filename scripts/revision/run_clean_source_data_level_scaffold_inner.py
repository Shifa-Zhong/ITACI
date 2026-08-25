"""Rerun DL-1/DL-2/DL-3 with true Bemis-Murcko scaffold-group inner CV."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import run_all_clean_source_data_level as runner
from clean_source_common import ENDPOINTS, read_json, result_key, write_json_atomic
from clean_source_independent_common_v3 import (
    INDEPENDENT_RESULTS,
    evaluate_clipped,
    load_independent_inputs,
    outer_metadata,
)
from run_all_clean_source_neural import scaffold_groups
from run_clean_source_data_level_independent import nested_candidate_model


def scaffold_nested_candidate_model(
    x_train, y_train, seed, max_evals=6, n_folds=3, groups=None, n_jobs=8
):
    if groups is None:
        raise AssertionError("Primary data-level inner selection requires molecular groups")
    molecular_scaffolds = scaffold_groups(np.asarray(groups, dtype=str))
    return nested_candidate_model(
        x_train, y_train, seed, max_evals=max_evals, n_folds=n_folds,
        groups=molecular_scaffolds, n_jobs=n_jobs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dl1", "dl2", "dl3", "all"), default="all")
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=10)
    parser.add_argument("--target", choices=ENDPOINTS)
    args = parser.parse_args()

    datasets, splits, baseline = load_independent_inputs()
    runner.tune_lgb_model = scaffold_nested_candidate_model
    runner.evaluate = evaluate_clipped
    targets = [args.target] if args.target else ENDPOINTS
    methods = ("dl1", "dl2", "dl3") if args.method == "all" else (args.method,)
    functions = {"dl1": runner.run_dl1, "dl2": runner.run_dl2, "dl3": runner.run_dl3}
    for method in methods:
        path = INDEPENDENT_RESULTS / f"{method}_clean_results_v2.json"
        payload = read_json(path)
        for rep in range(args.rep_start, min(args.rep_end, 10)):
            for target in targets:
                key = result_key(target, rep)
                if key in payload:
                    continue
                started = time.time()
                result = functions[method](datasets, splits, baseline, target, rep, 6)
                result.update(outer_metadata(rep))
                result["inner_grouping"] = "Bemis-Murcko scaffold"
                payload[key] = result
                write_json_atomic(payload, path)
                print(
                    f"{method}_v2 rep={rep} target={target} AUC={result['AUC']:.4f} "
                    f"delta={result['delta_AUC']:+.4f} removed={result['source_rows_removed']} "
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
