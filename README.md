# ITACI — Decision Boundary Incompatibility in Cross-Endpoint Toxicity Transfer

Reproducibility code for the manuscript *"Decision Boundary Incompatibility as
the Central Barrier to Cross-Endpoint Transfer in Computational Toxicology"*.

This repository contains only the **data-generation pipeline** — the scripts
that load the raw toxicity datasets, build scaffold-disjoint splits, train all
baseline and cross-endpoint transfer models, and compute every diagnostic
metric reported in the paper (cross-prediction AUC matrix, JSD, ITACI, SMD,
overlap statistics). Figure-rendering and manuscript scripts are not included.

Every script writes its outputs into a top-level `results/` directory as
pickled Python objects, and most checkpoint themselves so they can be
interrupted and resumed.

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

# 3. point each script at the data folder
#    (default paths in the scripts are absolute Windows paths used during
#     development; either run with the working directory set to the repo
#     root and edit the DATA_FILE / RESULTS_DIR constants near the top of
#     each script, or symlink `data/` and `results/` to your preferred
#     locations.)

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

## Pipeline order

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

- All `step_scaffold_*` scripts use **Bemis–Murcko scaffold-disjoint 80/20
  splits** as the primary protocol, repeated `N = 10` times with seeds 101–110.
  The earlier non-`scaffold_*` outputs (e.g. `baseline_results.pkl`) come from
  random splits and are only used as a within-scaffold contrast.
- Hyperparameter optimisation: LightGBM via TPE (50 evaluations, 5-fold CV).
  Search spaces are inlined in each script.
- Neural network strategies (MTL / MAML / FEFA / diversity control) require
  PyTorch and a GPU is recommended (CPU works but is ~10× slower).
- Mordred descriptors take ~1–2 hours on a modern CPU; `mordred_datasets.pkl`
  is ~640 MB. The FP-only track does not need it.
- The `_checkpoint.py` helper provides atomic pickle saves so any script can
  be killed and resumed without corrupting partial results.

---

## Citation

If you use this code, please cite the paper (TBD upon acceptance).
