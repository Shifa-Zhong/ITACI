"""Compute a safe Mordred-space nearest-neighbour sensitivity analysis.

Descriptors are rebuilt from canonical SMILES and saved as ordinary NPZ/CSV
artifacts.  No historical pickle checkpoint is loaded.  Within every outer
scaffold fold, descriptor preprocessing (median imputation, standardization,
and 32-component PCA) is fitted on the training compounds only.  The primary
Morgan-model out-of-fold predictions are then stratified by both Tanimoto
support and Mordred-PCA Euclidean nearest-neighbour distance.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem, RDLogger
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "revision" / "ood_analysis"
RAW = OUT / "mordred_raw_safe.npz"


def compute_descriptors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if RAW.exists():
        data = np.load(RAW, allow_pickle=False)
        return data["smiles"], data["names"], data["values"]
    records = pd.read_csv(OUT / "records.csv", encoding="utf-8-sig")
    smiles = np.array(sorted(records.smiles.unique()), dtype=str)
    mols = [Chem.MolFromSmiles(value) for value in smiles]
    calculator = Calculator(descriptors, ignore_3D=True)
    started = time.time()
    frame = calculator.pandas(mols, nproc=12, quiet=False)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    names = np.asarray(frame.columns.astype(str), dtype=str)
    values = frame.to_numpy(dtype=np.float32)
    np.savez_compressed(RAW, smiles=smiles, names=names, values=values)
    print(f"Computed {values.shape} Mordred matrix in {(time.time()-started)/60:.1f} min", flush=True)
    return smiles, names, values


def safe_auc(y, probability):
    return float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else np.nan


def main() -> None:
    smiles, names, values = compute_descriptors()
    records = pd.read_csv(OUT / "records.csv", encoding="utf-8-sig")
    predictions = pd.read_csv(OUT / "independent_cv_predictions.csv", encoding="utf-8-sig")
    smile_to_index = {value: index for index, value in enumerate(smiles)}
    record_descriptor_index = records.smiles.map(smile_to_index).to_numpy(dtype=int)
    distance_rows = []

    for endpoint_number, endpoint in enumerate(records.endpoint.drop_duplicates(), start=1):
        endpoint_records = records.loc[records.endpoint.eq(endpoint)].reset_index(drop=True)
        endpoint_global = records.index[records.endpoint.eq(endpoint)].to_numpy()
        endpoint_values = values[record_descriptor_index[endpoint_global]]
        print(f"[{endpoint_number}/13] {endpoint}", flush=True)
        for (repeat, fold), prediction_fold in predictions.loc[
            predictions.endpoint.eq(endpoint)
        ].groupby(["repeat", "fold"], sort=True):
            test_indices = prediction_fold.endpoint_row.to_numpy(dtype=int)
            train_mask = np.ones(len(endpoint_records), dtype=bool)
            train_mask[test_indices] = False
            train_indices = np.flatnonzero(train_mask)

            # Training-only filtering and preprocessing.
            train_raw = endpoint_values[train_indices]
            test_raw = endpoint_values[test_indices]
            finite_train = np.where(np.isfinite(train_raw), train_raw, np.nan)
            finite_test = np.where(np.isfinite(test_raw), test_raw, np.nan)
            nan_fraction = np.isnan(finite_train).mean(axis=0)
            keep = nan_fraction <= 0.05
            finite_train = finite_train[:, keep]
            finite_test = finite_test[:, keep]
            imputer = SimpleImputer(strategy="median")
            train_imputed = imputer.fit_transform(finite_train)
            test_imputed = imputer.transform(finite_test)
            variance = np.var(train_imputed, axis=0)
            nonconstant = variance > 1e-12
            train_imputed = train_imputed[:, nonconstant]
            test_imputed = test_imputed[:, nonconstant]
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_imputed)
            test_scaled = scaler.transform(test_imputed)
            components = min(32, train_scaled.shape[0] - 1, train_scaled.shape[1])
            pca = PCA(n_components=components, svd_solver="randomized", random_state=1000 + int(repeat) * 10 + int(fold))
            train_pc = pca.fit_transform(train_scaled)
            test_pc = pca.transform(test_scaled)
            neighbour = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=8)
            neighbour.fit(train_pc)
            distances = neighbour.kneighbors(test_pc, return_distance=True)[0][:, 0]
            for endpoint_row, distance in zip(test_indices, distances):
                distance_rows.append(
                    {
                        "endpoint": endpoint,
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "endpoint_row": int(endpoint_row),
                        "mordred_pca32_nn_distance": float(distance),
                        "n_descriptors_kept": int(nonconstant.sum()),
                        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
                    }
                )

    distances = pd.DataFrame(distance_rows)
    distances.to_csv(OUT / "mordred_nn_distances.csv", index=False, encoding="utf-8-sig")
    combined = predictions.merge(
        distances, on=["endpoint", "repeat", "fold", "endpoint_row"], how="left", validate="one_to_one"
    )
    pooled = combined.groupby(
        ["endpoint", "endpoint_name", "record_id", "label", "smiles", "scaffold"], as_index=False
    ).agg(
        probability=("probability", "mean"),
        nn_tanimoto=("nn_tanimoto", "mean"),
        mordred_pca32_nn_distance=("mordred_pca32_nn_distance", "mean"),
    )
    # Retain the raw Euclidean distance as the scientific quantity and add an
    # explicit display transform. A reproducible extreme Ames observation
    # (record_id 26570) makes a raw linear-scale histogram unreadable; the
    # log10(1 + d) column permits transparent plotting without deleting or
    # winsorizing any record.
    pooled["log10_1p_mordred_pca32_nn_distance"] = np.log10(
        1.0 + pooled["mordred_pca32_nn_distance"]
    )
    pooled["mordred_distance_rank_descending"] = pooled[
        "mordred_pca32_nn_distance"
    ].rank(method="first", ascending=False).astype(int)
    pooled["is_max_mordred_distance"] = pooled[
        "mordred_distance_rank_descending"
    ].eq(1)
    pooled.to_csv(OUT / "pooled_oof_predictions_with_mordred_distance.csv", index=False, encoding="utf-8-sig")

    rows = []
    for endpoint, frame in pooled.groupby("endpoint", sort=False):
        frame = frame.copy()
        frame["tanimoto_quartile"] = pd.qcut(
            frame.nn_tanimoto.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]
        )
        frame["mordred_distance_quartile"] = pd.qcut(
            frame.mordred_pca32_nn_distance.rank(method="first"),
            4,
            labels=["Q1 nearest", "Q2", "Q3", "Q4 farthest"],
        )
        for metric_name, column in [
            ("Tanimoto similarity", "tanimoto_quartile"),
            ("Mordred-PCA distance", "mordred_distance_quartile"),
        ]:
            for quartile, stratum in frame.groupby(column, observed=True):
                rows.append(
                    {
                        "endpoint": endpoint,
                        "endpoint_name": frame.endpoint_name.iloc[0],
                        "distance_metric": metric_name,
                        "quartile": str(quartile),
                        "n": len(stratum),
                        "median_tanimoto": float(stratum.nn_tanimoto.median()),
                        "median_mordred_distance": float(stratum.mordred_pca32_nn_distance.median()),
                        "roc_auc": safe_auc(stratum.label, stratum.probability),
                        "pr_auc": float(average_precision_score(stratum.label, stratum.probability)),
                        "brier_score": float(brier_score_loss(stratum.label, stratum.probability)),
                    }
                )
    trend = pd.DataFrame(rows)
    trend.to_csv(OUT / "performance_by_fp_and_mordred_support.csv", index=False, encoding="utf-8-sig")

    summary = {
        "median_mordred_pca32_nn_distance": float(pooled.mordred_pca32_nn_distance.median()),
        "q25_mordred_pca32_nn_distance": float(pooled.mordred_pca32_nn_distance.quantile(0.25)),
        "q75_mordred_pca32_nn_distance": float(pooled.mordred_pca32_nn_distance.quantile(0.75)),
    }
    (OUT / "mordred_similarity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
