"""Final clean-source MAML run on the independent outer scaffold folds."""

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
    path = INDEPENDENT_RESULTS / "maml_clean_results.json"
    payload = read_json(path)
    targets = [args.target] if args.target else ENDPOINTS
    for rep in range(args.rep_start, min(args.rep_end, 10)):
        for target in targets:
            key = result_key(target, rep)
            if key in payload:
                continue
            started = time.time()
            result = runner.run_maml(datasets, splits, baseline, target, rep, 500, 100)
            result.update(outer_metadata(rep))
            payload[key] = result
            write_json_atomic(payload, path)
            print(
                f"maml rep={rep} target={target} delta={result['delta_AUC']:+.4f} "
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
