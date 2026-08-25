"""Primary nested clean-source stacking on independent scaffold-group folds."""

from __future__ import annotations

import sys

import run_clean_source_stacking_nested as runner
from clean_source_independent_common_v3 import (
    INDEPENDENT_RESULTS,
    evaluate_clipped,
    load_independent_inputs,
    outer_metadata,
)


original_run_one = runner.run_one


def run_one_with_metadata(datasets, splits, baseline, target, rep, smoke=False):
    result = original_run_one(datasets, splits, baseline, target, rep, smoke)
    result.update(outer_metadata(rep))
    return result


runner.load_inputs = load_independent_inputs
runner.CLEAN_RESULTS = INDEPENDENT_RESULTS
runner.evaluate = evaluate_clipped
runner.run_one = run_one_with_metadata


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    runner.main()
