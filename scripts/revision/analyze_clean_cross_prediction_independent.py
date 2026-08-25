"""Summarize the leakage-free independent-fold cross-prediction diagnostic."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from clean_source_common import ENDPOINTS
from clean_source_independent_common_v3 import INDEPENDENT_RESULTS as OUT


def main():
    path = OUT / "cross_prediction_clean_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload) != 13 * 13 * 10:
        raise AssertionError(f"Expected 1690 source-target-fold records, found {len(payload)}")
    records = pd.DataFrame(payload.values())
    if int(records.source_target_test_overlap_after.max()) != 0:
        raise AssertionError("Post-clean overlap is nonzero")
    pair = (
        records.groupby(["source", "target"], sort=False)
        .agg(AUC_mean=("AUC", "mean"), AUC_sd=("AUC", "std"),
             source_rows_removed=("source_rows_removed", "sum"))
        .reset_index()
    )
    off = pair.loc[~pair.source.eq(pair.target)].copy()
    matrix = pair.pivot(index="source", columns="target", values="AUC_mean").reindex(
        index=ENDPOINTS, columns=ENDPOINTS
    )
    summary = {
        "n_ordered_off_diagonal_pairs": 156,
        "n_outer_folds_per_pair": 10,
        "off_diagonal_mean_auc": float(off.AUC_mean.mean()),
        "off_diagonal_sd_auc": float(off.AUC_mean.std(ddof=1)),
        "pairs_below_0_55": int((off.AUC_mean < 0.55).sum()),
        "pairs_above_0_70": int((off.AUC_mean > 0.70).sum()),
        "source_rows_removed_total_across_pair_folds": int(records.source_rows_removed.sum()),
        "max_post_clean_source_target_test_overlap": 0,
    }
    pair.to_csv(OUT / "cross_prediction_pair_summary.csv", index=False, encoding="utf-8-sig")
    matrix.to_csv(OUT / "cross_prediction_auc_matrix.csv", encoding="utf-8-sig")
    (OUT / "cross_prediction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
