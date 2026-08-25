"""Exact inputs for the primary independent clean-source comparison.

Both the Morgan features and outer-fold baseline predictions are loaded from
the completed two-repeat, five-fold nested scaffold-group analysis.  This
avoids mixing those baseline predictions with a legacy fingerprint array whose
canonicalization differs for three organometallic records.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

import clean_source_common as common


ROOT = Path(__file__).resolve().parents[2]
OOD = ROOT / "results" / "revision" / "ood_analysis"
INDEPENDENT_RESULTS = ROOT / "results" / "revision" / "clean_source_independent"
INDEPENDENT_RESULTS.mkdir(parents=True, exist_ok=True)

OUTER_KEYS = tuple((repeat, fold) for repeat in (1, 2) for fold in range(1, 6))


def evaluate_clipped(y, probability):
    return common.evaluate(
        np.asarray(y, dtype=int),
        np.clip(np.asarray(probability, dtype=float), 0.0, 1.0),
    )


def load_independent_inputs():
    records = pd.read_csv(OOD / "records.csv")
    predictions = pd.read_csv(OOD / "independent_cv_predictions.csv")
    global_x = load_npz(OOD / "morgan_features.npz").tocsr()
    if global_x.shape != (len(records), 2048):
        raise AssertionError("Independent Morgan feature matrix has unexpected shape")

    datasets = {}
    splits = {}
    baseline = {}
    for endpoint in common.ENDPOINTS:
        endpoint_records = (
            records.loc[records.endpoint.eq(endpoint)]
            .sort_values("record_id")
            .reset_index(drop=True)
        )
        endpoint_records["endpoint_row"] = np.arange(len(endpoint_records), dtype=int)
        global_rows = endpoint_records.record_id.to_numpy(dtype=int)
        endpoint_x = global_x[global_rows].toarray().astype(np.uint8, copy=False)
        endpoint_y = endpoint_records.label.to_numpy(dtype=np.int8)
        endpoint_smiles = endpoint_records.smiles.astype(str).to_numpy()
        datasets[endpoint] = {"X": endpoint_x, "y": endpoint_y, "smiles": endpoint_smiles}

        expected_rows = np.arange(len(endpoint_records), dtype=int)
        endpoint_splits = []
        endpoint_baseline = []
        for repeat, fold in OUTER_KEYS:
            frame = predictions.loc[
                predictions.endpoint.eq(endpoint)
                & predictions["repeat"].eq(repeat)
                & predictions.fold.eq(fold)
            ].sort_values("endpoint_row")
            test_index = frame.endpoint_row.to_numpy(dtype=int)
            train_index = np.setdiff1d(expected_rows, test_index, assume_unique=True)
            if not len(test_index) or len(train_index) + len(test_index) != len(expected_rows):
                raise AssertionError(f"Invalid outer fold for {endpoint}, r{repeat}f{fold}")
            if set(train_index).intersection(test_index):
                raise AssertionError("Outer train/test index overlap")
            if not np.array_equal(endpoint_y[test_index], frame.label.to_numpy(dtype=int)):
                raise AssertionError(f"Label alignment mismatch for {endpoint}, r{repeat}f{fold}")
            if not np.array_equal(endpoint_smiles[test_index], frame.smiles.astype(str).to_numpy()):
                raise AssertionError(f"SMILES alignment mismatch for {endpoint}, r{repeat}f{fold}")
            probability = np.clip(frame.probability.to_numpy(dtype=float), 0.0, 1.0)
            metrics = evaluate_clipped(endpoint_y[test_index], probability)
            metrics.update({"outer_repeat": repeat, "outer_fold": fold})
            endpoint_splits.append((train_index, test_index))
            endpoint_baseline.append(metrics)

        if len(endpoint_splits) != 10:
            raise AssertionError(f"Expected 10 outer folds for {endpoint}")
        splits[endpoint] = endpoint_splits
        baseline[endpoint] = endpoint_baseline

    if list(datasets) != common.ENDPOINTS:
        raise AssertionError("Endpoint order differs from the prespecified order")
    return datasets, splits, baseline


def outer_metadata(rep_index: int) -> dict[str, int]:
    repeat, fold = OUTER_KEYS[int(rep_index)]
    return {"outer_repeat": int(repeat), "outer_fold": int(fold)}
