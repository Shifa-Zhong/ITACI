# ITACI — Cross-Endpoint Toxicity Transfer under Scaffold-Held-Out Evaluation

Reproducibility code for the manuscript *"Why Does Cross-Endpoint Transfer
Sometimes Fail in Computational Toxicology? Diagnosing Decision-Boundary
Incompatibility across 13 Endpoints under Scaffold-Held-Out Evaluation"*.

This repository contains only the **data-generation and statistical-analysis
pipeline**: scripts that load the toxicity datasets, construct scaffold-group
outer folds, train the baseline and cross-endpoint models, remove target-test
compound identities from every source-training branch, and calculate the
reported performance and support diagnostics. Figure rendering, manuscript or
response-letter generation, and submission-audit code are intentionally not
included.

Revised scripts write machine-readable CSV, JSON, and NPZ outputs under
`results/revision/`. Legacy scripts retain their original pickle checkpoint
format. Long-running model scripts checkpoint completed endpoint-fold records
so they can be interrupted and resumed.

---

## Quickstart

```bash
# 1. clone & enter
git clone https://github.com/Shifa-Zhong/ITACI.git
cd ITACI

# 2. create environment (Python 3.12 recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. scripts in scripts/revision use repository-relative data/results paths

mkdir -p results
```

---

## Data

`data/toxnew_right.xlsx` contains the 13 binary toxicity datasets used in the
study (one sheet per endpoint, columns `new_s` = canonical SMILES, `label`
= 0/1). 41,160 compound–endpoint records total; 24,428 unique SMILES.

| Endpoint                | N     |
| ----------------------- | ----- |
| cytotoxicity_C          |   342 |
| neurotoxicity_C         |   681 |
| prenatal_development_C  | 1,187 |
| Carcinogenicity_C       | 1,697 |
| skin_corrosion_C        | 1,931 |
| reproductive_toxicity_C | 2,040 |
| Estrogen_Receptor_α_C   | 2,379 |
| respiratory_toxicity_C  | 2,517 |
| Hepatotoxicity_C        | 2,586 |
| Androgen_Receptor_C     | 2,769 |
| ocular_toxicity_C       | 5,112 |
| TSHR_agonist_activity_C | 8,580 |
| ames_mutagenicity_C     | 9,339 |

---

## Revised primary pipeline

The final revised analysis is in `scripts/revision/`. It uses two independent
randomizations of five-fold stratified scaffold-group cross-validation. For
each target outer fold, every source-dependent method removes all compounds
whose canonical identity occurs in the target test fold. Model and ensemble
selection are confined to the corresponding outer-training data.

Run from the repository root:

```bash
# Build Morgan features and the two-repeat x five-fold baseline OOF predictions
python scripts/revision/revision_ood_analysis.py --build-features
python scripts/revision/revision_ood_analysis.py --run

# Leakage-controlled data-level methods and size-matched target-only control
python scripts/revision/run_clean_source_data_level_scaffold_inner.py --method all
python scripts/revision/run_clean_source_dl4_scaffold_inner.py

# Higher-level methods
python scripts/revision/run_clean_source_stacking_independent.py
python scripts/revision/run_clean_source_mtl_independent.py
python scripts/revision/run_clean_source_maml_independent.py
python scripts/revision/run_clean_source_fefa_fully_nested.py

# Leakage-free ordered source-target cross-prediction matrix
python scripts/revision/run_clean_cross_prediction_independent.py

# Final pooled repeated-OOF metrics, scaffold bootstrap, Wilcoxon/Holm results
python scripts/revision/analyze_clean_source_independent.py
python scripts/revision/analyze_clean_cross_prediction_independent.py

# Optional Mordred-PCA nearest-neighbour support sensitivity
python scripts/revision/revision_mordred_similarity.py
```

The revised outputs are written under:

- `results/revision/ood_analysis/`
- `results/revision/clean_source_independent/`

The fully nested FEFA runner can be distributed across workers with
`--shard-id` and `--shard-count`; use `--merge-only` with the same shard count
after all shards finish. The default single-process invocation writes the
canonical `fefa_clean_results_fully_nested.json` directly.

No script in `scripts/revision/` creates a figure, edits a DOCX file, or audits
submission formatting.

---

## Legacy/original pipeline order

Run scripts in the order below; later scripts consume the pickle files
produced by earlier ones. Anything Mordred-related is optional — the central
findings are reported on Morgan fingerprints (FP); Mordred re-runs are used as
the cross-representation robustness check.

### 1. Data + splits

| # | Script                              | Produces                          |
| - | ----------------------------------- | --------------------------------- |
| 1 | `step1_baseline.py`                 | `datasets.pkl` (FP + random splits) |
| 2 | `step_scaffold_splits_record.py`    | `scaffold_splits_record.pkl`     |
| 3 | `step_scaffold_baseline.py`         | `scaffold_baseline_results.pkl`  |

### 2. Mordred 2D descriptors (optional cross-representation track)

| # | Script                              | Produces                                  |
| - | ----------------------------------- | ----------------------------------------- |
| 4 | `step_mordred_features.py`          | `mordred_datasets.pkl`                    |
| 5 | `step_scaffold_baseline_mordred.py` | `scaffold_baseline_mordred_results.pkl`   |

### 3. Data-level integration strategies (DL-1, DL-2, DL-3, DL-4)

| # | Script                                    | Produces                                                                         |
| - | ----------------------------------------- | -------------------------------------------------------------------------------- |
| 6 | `step_scaffold_data_level.py`             | `scaffold_dl_corr_aug.pkl`, `scaffold_dl_global_merge.pkl`, `scaffold_dl_shap_pairs.pkl` |
| 7 | `step_scaffold_dl4_bootstrap.py`          | `scaffold_dl4_bootstrap.pkl`                                                     |
| 8 | `step_scaffold_data_level_mordred.py`     | Mordred variants of DL-1 / DL-2                                                  |
| 9 | `step_scaffold_dl3_mordred.py`            | `scaffold_dl_shap_pairs_mordred.pkl`                                             |
| 10| `step_scaffold_dl4_bootstrap_mordred.py`  | `scaffold_dl4_bootstrap_mordred.pkl`                                             |

### 4. Higher-level transfer strategies (S1 stacking, S2 MTL, S3 MAML, FEFA)

| # | Script                                | Produces                                            |
| - | ------------------------------------- | --------------------------------------------------- |
| 11| `step_scaffold_stacking.py`           | `scaffold_strategy1.pkl`                            |
| 12| `step_scaffold_mtl.py`                | `scaffold_strategy2.pkl`                            |
| 13| `step_scaffold_maml.py`               | `scaffold_strategy3.pkl`                            |
| 14| `step_scaffold_fefa.py`               | `scaffold_strategy3_fefa.pkl`                       |
| 15| `step_scaffold_stacking_mordred.py`   | `scaffold_strategy1_mordred.pkl`                    |
| 16| `step_scaffold_mtl_mordred.py`        | `scaffold_strategy2_mordred.pkl`                    |
| 17| `step_scaffold_maml_mordred.py`       | `scaffold_strategy3_mordred.pkl`                    |

### 5. Robustness controls

| # | Script                                 | Produces                                                                     |
| - | -------------------------------------- | ---------------------------------------------------------------------------- |
| 18| `step_scaffold_diversity_matched.py`   | `scaffold_diversity_control.pkl`, `scaffold_matched_control.pkl`             |
| 19| `step_scaffold_clean_source_v2.py`     | `scaffold_clean_source_v2.pkl`                                               |
| 20| `step_scaffold_rf_arch.py`             | `scaffold_rf_baseline.pkl`, `scaffold_rf_stacking.pkl`, `scaffold_arch_sensitivity.pkl` |

### 6. Root-cause diagnostics + ITACI / SMD / overlap analysis

| # | Script                                 | Produces                                                       |
| - | -------------------------------------- | -------------------------------------------------------------- |
| 21| `step_scaffold_rootcause.py`           | `scaffold_rootcause_results.pkl`, `shap_results.pkl`           |
| 22| `step_scaffold_diagnostics_mordred.py` | `scaffold_diagnostics_mordred.pkl`                             |
| 23| `step_itaci_smd.py`                    | `scaffold_itaci_smd.pkl`                                       |
| 24| `step_itaci_smd_mordred.py`            | `scaffold_itaci_smd_mordred.pkl`                               |
| 25| `step_overlap_analysis.py`             | `scaffold_overlap_analysis.pkl`                                |
| 26| `step_scaffold_final_analysis.py`      | `scaffold_final_summary.pkl` (aggregate paired Wilcoxon stats) |

---

## Reproducibility notes

- The revised primary protocol in `scripts/revision/` is **two independent
  randomizations × five outer scaffold-group folds**. The older
  `step_scaffold_*` scripts below use repeated scaffold-disjoint 80/20 splits
  and are retained for historical reproducibility and sensitivity analyses;
  they are not the revised primary estimand.
- Revised LightGBM selection uses a prespecified finite candidate set within
  scaffold-grouped inner folds; fully nested FEFA uses TPE only on fold-specific
  inner embeddings. The legacy scripts retain their original TPE settings.
  Search spaces are inlined in the corresponding scripts.
- Neural network strategies (MTL / MAML / FEFA / diversity control) require
  PyTorch and a GPU is recommended (CPU works but is ~10× slower).
- Mordred descriptors take ~1–2 hours on a modern CPU; `mordred_datasets.pkl`
  is ~640 MB. The FP-only track does not need it.
- The `_checkpoint.py` helper provides atomic pickle saves so any script can
  be killed and resumed without corrupting partial results.

---

## Citation

If you use this code, please cite the paper (TBD upon acceptance).
