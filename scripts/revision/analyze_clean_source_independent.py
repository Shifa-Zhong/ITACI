"""Final statistics and scaffold-bootstrap audit for independent primary runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import clean_source_common as common
from clean_source_independent_common_v3 import (
    INDEPENDENT_RESULTS as OUT,
    OOD,
    evaluate_clipped,
    load_independent_inputs,
)


METHOD_FILES = {
    "DL-1": "dl1_clean_results_v2.json",
    "DL-2": "dl2_clean_results_v2.json",
    "DL-3": "dl3_clean_results_v2.json",
    "DL-4": "dl4_clean_results_v2.json",
    "S1 Stacking": "stacking_clean_results.json",
    "S2 MTL": "mtl_clean_results_v3.json",
    "S3 MAML": "maml_clean_results.json",
    "FEFA": "fefa_clean_results_fully_nested.json",
}
METRICS = (
    "AUC",
    "PR_AUC",
    "Balanced_Accuracy",
    "Sensitivity",
    "Specificity",
    "Brier",
    "Calibration_Intercept",
    "Calibration_Slope",
    "ECE_10",
)


def load_method(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload) != 130:
        raise AssertionError(f"{path.name}: expected 130 results, found {len(payload)}")
    records = list(payload.values())
    keys = {(record["endpoint"], int(record["rep"])) for record in records}
    expected = {(endpoint, rep) for endpoint in common.ENDPOINTS for rep in range(10)}
    if keys != expected:
        raise AssertionError(f"{path.name}: endpoint/fold keys are incomplete")
    if any(int(record["max_source_test_overlap_after"]) != 0 for record in records):
        raise AssertionError(f"{path.name}: nonzero post-clean source/test overlap")
    return records


def endpoint_probabilities(entries, splits, endpoint, n_records):
    sums = np.zeros(n_records, dtype=float)
    counts = np.zeros(n_records, dtype=int)
    for rep, entry in enumerate(entries):
        _train, test = splits[endpoint][rep]
        probability = np.asarray(entry["y_pred"], dtype=float)
        if len(probability) != len(test):
            raise AssertionError(f"Prediction length mismatch for {endpoint}, outer index {rep}")
        sums[test] += np.clip(probability, 0.0, 1.0)
        counts[test] += 1
    if not np.all(counts == 2):
        raise AssertionError(f"Each compound must have two OOF predictions for {endpoint}")
    return sums / counts


def scaffold_bootstrap(y, baseline_p, method_p, scaffold, seed, n_boot=1000):
    rng = np.random.default_rng(seed)
    unique = np.unique(scaffold)
    grouped = {value: np.flatnonzero(scaffold == value) for value in unique}
    baseline_auc = []
    method_auc = []
    delta_auc = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([grouped[value] for value in sampled])
        sampled_y = y[indices]
        if np.unique(sampled_y).size < 2:
            continue
        base = float(roc_auc_score(sampled_y, baseline_p[indices]))
        method = float(roc_auc_score(sampled_y, method_p[indices]))
        baseline_auc.append(base)
        method_auc.append(method)
        delta_auc.append(method - base)
    if len(delta_auc) < int(0.95 * n_boot):
        raise AssertionError("Too many single-class scaffold bootstrap samples")
    return {
        "baseline_auc": np.asarray(baseline_auc),
        "method_auc": np.asarray(method_auc),
        "delta_auc": np.asarray(delta_auc),
    }


def ci(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main():
    datasets, splits, baseline = load_independent_inputs()
    method_records = {
        method: load_method(OUT / filename) for method, filename in METHOD_FILES.items()
    }
    source_records = pd.read_csv(OOD / "records.csv")

    endpoint_rows = []
    extended_rows = []
    bootstrap_rows = []
    bootstrap_delta_by_method = {method: [] for method in METHOD_FILES}

    for endpoint_index, endpoint in enumerate(common.ENDPOINTS):
        n_records = len(datasets[endpoint]["y"])
        labels = np.asarray(datasets[endpoint]["y"], dtype=int)
        endpoint_scaffolds = (
            source_records.loc[source_records.endpoint.eq(endpoint)]
            .sort_values("record_id")
            .scaffold.astype(str)
            .to_numpy()
        )
        baseline_entries = baseline[endpoint]
        baseline_fold_auc = np.asarray([entry["AUC"] for entry in baseline_entries], dtype=float)
        baseline_pooled = endpoint_probabilities(baseline_entries, splits, endpoint, n_records)
        baseline_metrics = evaluate_clipped(labels, baseline_pooled)
        endpoint_row = {
            "endpoint": endpoint,
            "Baseline_mean": float(np.mean(baseline_fold_auc)),
            "Baseline_sd": float(np.std(baseline_fold_auc, ddof=1)),
        }
        extended_rows.append(
            {
                "endpoint": endpoint,
                "method": "Baseline",
                **{metric: float(baseline_metrics[metric]) for metric in METRICS},
            }
        )

        for method_index, (method, records) in enumerate(method_records.items()):
            selected = sorted(
                [record for record in records if record["endpoint"] == endpoint],
                key=lambda record: int(record["rep"]),
            )
            values = np.asarray([record["AUC"] for record in selected], dtype=float)
            deltas = values - baseline_fold_auc
            pooled = endpoint_probabilities(selected, splits, endpoint, n_records)
            pooled_metrics = evaluate_clipped(labels, pooled)
            endpoint_row[f"{method}_mean"] = float(np.mean(values))
            endpoint_row[f"{method}_sd"] = float(np.std(values, ddof=1))
            endpoint_row[f"{method}_delta"] = float(np.mean(deltas))
            extended_rows.append(
                {
                    "endpoint": endpoint,
                    "method": method,
                    **{metric: float(pooled_metrics[metric]) for metric in METRICS},
                }
            )
            distribution = scaffold_bootstrap(
                labels,
                baseline_pooled,
                pooled,
                endpoint_scaffolds,
                seed=814000 + 100 * endpoint_index + method_index,
            )
            base_low, base_high = ci(distribution["baseline_auc"])
            method_low, method_high = ci(distribution["method_auc"])
            delta_low, delta_high = ci(distribution["delta_auc"])
            bootstrap_rows.append(
                {
                    "endpoint": endpoint,
                    "method": method,
                    "resampling_unit": "complete Bemis-Murcko scaffold group",
                    "n_bootstrap": 1000,
                    "baseline_auc_low": base_low,
                    "baseline_auc_high": base_high,
                    "method_auc_low": method_low,
                    "method_auc_high": method_high,
                    "delta_auc_low": delta_low,
                    "delta_auc_high": delta_high,
                }
            )
            bootstrap_delta_by_method[method].append(distribution["delta_auc"])
        endpoint_rows.append(endpoint_row)

    per_endpoint = pd.DataFrame(endpoint_rows)
    extended = pd.DataFrame(extended_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    # The primary estimand is one pooled repeated-OOF ROC-AUC per endpoint.
    pooled_auc = extended.pivot(index="endpoint", columns="method", values="AUC")
    per_endpoint = per_endpoint.set_index("endpoint")
    for endpoint in per_endpoint.index:
        per_endpoint.loc[endpoint, "Baseline_mean"] = pooled_auc.loc[endpoint, "Baseline"]
        for method in METHOD_FILES:
            method_auc = float(pooled_auc.loc[endpoint, method])
            baseline_auc = float(pooled_auc.loc[endpoint, "Baseline"])
            per_endpoint.loc[endpoint, f"{method}_mean"] = method_auc
            per_endpoint.loc[endpoint, f"{method}_delta"] = method_auc - baseline_auc
    per_endpoint = per_endpoint.reset_index()

    summary_rows = []
    for method, records in method_records.items():
        deltas = per_endpoint[f"{method}_delta"].to_numpy(dtype=float)
        test = stats.wilcoxon(deltas, zero_method="pratt", alternative="two-sided")
        boot_arrays = bootstrap_delta_by_method[method]
        common_length = min(len(values) for values in boot_arrays)
        macro_boot = np.mean(
            np.vstack([values[:common_length] for values in boot_arrays]), axis=0
        )
        low, high = ci(macro_boot)
        summary_rows.append(
            {
                "method": method,
                "macro_delta_auc": float(np.mean(deltas)),
                "macro_delta_auc_estimand": "mean endpoint-level pooled repeated-OOF ROC-AUC difference",
                "macro_delta_auc_scaffold_boot_low": low,
                "macro_delta_auc_scaffold_boot_high": high,
                "median_endpoint_delta": float(np.median(deltas)),
                "endpoints_improved": int(np.sum(deltas > 0)),
                "endpoints_unchanged": int(np.sum(np.isclose(deltas, 0))),
                "wilcoxon_unit": "endpoint pooled repeated-OOF ROC-AUC difference; Pratt zero handling",
                "wilcoxon_n": 13,
                "wilcoxon_statistic": float(test.statistic),
                "wilcoxon_p_nominal": float(test.pvalue),
                "source_rows_removed_total": int(
                    sum(int(record["source_rows_removed"]) for record in records)
                ),
                "post_clean_source_test_overlap": 0,
            }
        )
    summary = pd.DataFrame(summary_rows)
    values = summary["wilcoxon_p_nominal"].to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    summary["wilcoxon_p_holm"] = adjusted
    summary["wilcoxon_p"] = adjusted

    macro_metrics = (
        extended.groupby("method", sort=False)[list(METRICS)]
        .mean()
        .reset_index()
    )
    removal_audit = pd.DataFrame(
        [
            {
                "method": method,
                "target_outer_folds": len(records),
                "source_rows_removed_total": int(sum(r["source_rows_removed"] for r in records)),
                "source_rows_removed_min_per_target_fold": int(min(r["source_rows_removed"] for r in records)),
                "source_rows_removed_max_per_target_fold": int(max(r["source_rows_removed"] for r in records)),
                "max_post_clean_source_test_overlap": int(max(r["max_source_test_overlap_after"] for r in records)),
            }
            for method, records in method_records.items()
        ]
    )

    per_endpoint.to_csv(OUT / "primary_per_endpoint_auc.csv", index=False, encoding="utf-8-sig")
    extended.to_csv(OUT / "primary_per_endpoint_extended_metrics.csv", index=False, encoding="utf-8-sig")
    macro_metrics.to_csv(OUT / "primary_macro_extended_metrics.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(OUT / "primary_scaffold_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "primary_method_summary.csv", index=False, encoding="utf-8-sig")
    removal_audit.to_csv(OUT / "primary_leakage_audit.csv", index=False, encoding="utf-8-sig")
    payload = {
        "design": "two independent randomizations x five outer scaffold-group folds",
        "inner_selection": "three scaffold-group folds",
        "statistical_unit": "endpoint pooled repeated-OOF ROC-AUC difference (n=13)",
        "multiplicity": "Holm family-wise correction across eight primary comparisons",
        "all_primary_methods_target_specific_clean_source": True,
        "max_post_clean_source_test_overlap": 0,
        "summary": summary.to_dict(orient="records"),
        "macro_extended_metrics": macro_metrics.to_dict(orient="records"),
    }
    (OUT / "primary_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\nMacro metrics")
    print(macro_metrics.to_string(index=False))
    print("\nLeakage audit")
    print(removal_audit.to_string(index=False))


if __name__ == "__main__":
    main()
