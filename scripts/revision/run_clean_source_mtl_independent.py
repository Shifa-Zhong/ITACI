"""Final clean-source MTL run on the independent outer scaffold folds."""

from __future__ import annotations

import argparse
import sys
import time

import run_all_clean_source_neural as runner
from clean_source_common import ENDPOINTS, read_json, result_key, write_json_atomic
from clean_source_independent_common_v3 import (
    INDEPENDENT_RESULTS,
    evaluate_clipped,
    load_independent_inputs,
    outer_metadata,
)


runner.evaluate = evaluate_clipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=10)
    parser.add_argument("--target", choices=ENDPOINTS)
    args = parser.parse_args()
    datasets, splits, baseline = load_independent_inputs()
    path = INDEPENDENT_RESULTS / "mtl_clean_results_v3.json"
    payload = read_json(path)
    targets = [args.target] if args.target else ENDPOINTS
    for rep in range(args.rep_start, min(args.rep_end, 10)):
        for target in targets:
            key = result_key(target, rep)
            if key in payload:
                continue
            started = time.time()
            mtl_result, _unused_intermediate_fefa = runner.run_mtl_fefa(
                datasets, splits, baseline, target, rep,
                100, 3, 30, 3, 15,
            )
            mtl_result.update(outer_metadata(rep))
            payload[key] = mtl_result
            write_json_atomic(payload, path)
            print(
                f"mtl rep={rep} target={target} delta={mtl_result['delta_AUC']:+.4f} "
                f"removed={mtl_result['source_rows_removed']} seconds={time.time()-started:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    main()
