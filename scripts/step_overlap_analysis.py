"""Compound overlap analyses for §2.7 / §S6 TBD numbers.

Computes (under scaffold-disjoint splits):
  (a) per-target overlap rate: fraction of target test compounds that appear in ANY
      other endpoint's scaffold-split training set, per rep, then averaged.
  (b) per-pair overlap rate (156 ordered pairs): fraction of target test compounds in
      pair (A->B) that appear in source B's scaffold-split training set.
  (c) Spearman correlations:
      - cross-pred AUC vs per-pair overlap rate (156 pairs)
      - per-target stacking ΔAUC vs mean overlap rate (13 endpoints)
      - lowest-overlap-quartile mean cross-pred AUC
  (d) DL-2 vs DL-4 paired tests; transfer ΔAUC = DL-2 mean minus DL-4 mean.

Output:
  results/scaffold_overlap_analysis.pkl
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.stats import spearmanr, wilcoxon

RESULTS_DIR = r'D:\quxintong\results'

ENDPOINTS = [
    'prenatal_development_C', 'TSHR_agonist_activity_C',
    'respiratory_toxicity_C', 'ocular_toxicity_C',
    'ames_mutagenicity_C', 'reproductive_toxicity_C',
    'skin_corrosion_C', 'neurotoxicity_C',
    'Estrogen_Receptor_α_C', 'Androgen_Receptor_C',
    'cytotoxicity_C', 'Carcinogenicity_C', 'Hepatotoxicity_C',
]


def fp_to_key(row):
    """Hash a 2048-bit row to a bytes key for fast set membership."""
    return bytes(row.astype(np.uint8))


def main():
    print('=' * 70, flush=True)
    print('Compound overlap & DL-2/DL-4 analyses', flush=True)
    print('=' * 70, flush=True)
    t0 = time.time()

    print('Loading datasets, splits, and predictions...', flush=True)
    datasets = pickle.load(open(os.path.join(RESULTS_DIR, 'datasets.pkl'), 'rb'))
    splits = pickle.load(open(os.path.join(RESULTS_DIR, 'scaffold_splits_record.pkl'), 'rb'))
    baseline = pickle.load(open(os.path.join(RESULTS_DIR, 'scaffold_baseline_results.pkl'), 'rb'))
    s1 = pickle.load(open(os.path.join(RESULTS_DIR, 'scaffold_strategy1.pkl'), 'rb'))
    dl2 = pickle.load(open(os.path.join(RESULTS_DIR, 'scaffold_dl_global_merge.pkl'), 'rb'))
    dl4 = pickle.load(open(os.path.join(RESULTS_DIR, 'scaffold_dl4_bootstrap.pkl'), 'rb'))
    root = pickle.load(open(os.path.join(RESULTS_DIR, 'scaffold_rootcause_results.pkl'), 'rb'))

    N_REPS = 10
    n_ep = len(ENDPOINTS)

    # === (a) per-target overlap rate (test in any-other train) ===
    print('\n--- (a) Per-target overlap rate (test in any-other train) ---', flush=True)
    overlap_any = np.zeros((n_ep, N_REPS))
    for i, ep in enumerate(ENDPOINTS):
        X = datasets[ep]['X']
        for rep in range(N_REPS):
            tr_idx, te_idx = splits[ep][rep]
            te_keys = set(fp_to_key(X[k]) for k in te_idx)
            other_train = set()
            for ep2 in ENDPOINTS:
                if ep2 == ep: continue
                X2 = datasets[ep2]['X']
                tr2, _ = splits[ep2][rep]
                for k in tr2:
                    other_train.add(fp_to_key(X2[k]))
            overlap_any[i, rep] = len(te_keys & other_train) / len(te_keys)
        m = overlap_any[i].mean()
        print(f'  {ep[:28]:<30} mean={m:.3f}  range=[{overlap_any[i].min():.3f}, {overlap_any[i].max():.3f}]', flush=True)
    per_target_overlap_mean = overlap_any.mean(axis=1)
    print(f'  Across 13 endpoints: mean={per_target_overlap_mean.mean():.3f}  '
          f'range=[{per_target_overlap_mean.min():.3f}, {per_target_overlap_mean.max():.3f}]', flush=True)

    # === (b) per-pair overlap (A's test ∩ B's train, normalized by A's test) ===
    print('\n--- (b) Per-pair overlap (target test ∩ source train) ---', flush=True)
    pair_overlap = np.zeros((n_ep, n_ep))
    for i, ep_A in enumerate(ENDPOINTS):
        X_A = datasets[ep_A]['X']
        for j, ep_B in enumerate(ENDPOINTS):
            if i == j: continue
            X_B = datasets[ep_B]['X']
            vals = []
            for rep in range(N_REPS):
                _, te_A = splits[ep_A][rep]
                tr_B, _ = splits[ep_B][rep]
                te_A_keys = set(fp_to_key(X_A[k]) for k in te_A)
                tr_B_keys = set(fp_to_key(X_B[k]) for k in tr_B)
                vals.append(len(te_A_keys & tr_B_keys) / max(len(te_A_keys), 1))
            pair_overlap[i, j] = np.mean(vals)
    print(f'  156 directed pairs: mean={pair_overlap[~np.eye(n_ep,dtype=bool)].mean():.3f}, '
          f'median={np.median(pair_overlap[~np.eye(n_ep,dtype=bool)]):.3f}', flush=True)

    # === (c) Spearman correlations ===
    print('\n--- (c) Spearman correlations ---', flush=True)
    cross_aucs_mat = root['cross_pred_mean_matrix']

    # cross-pred AUC vs pair overlap (156 directed pairs)
    cp_flat, ov_flat = [], []
    for i in range(n_ep):
        for j in range(n_ep):
            if i == j: continue
            cp_flat.append(cross_aucs_mat[i, j])
            ov_flat.append(pair_overlap[i, j])
    rho, pval = spearmanr(cp_flat, ov_flat)
    print(f'  Cross-pred AUC vs per-pair overlap (n=156): ρ={rho:.3f}, p={pval:.4g}', flush=True)

    # Lowest-overlap quartile cross-pred AUC
    cp_arr = np.array(cp_flat); ov_arr = np.array(ov_flat)
    q1 = np.quantile(ov_arr, 0.25)
    low_mask = ov_arr <= q1
    print(f'  Lowest-overlap quartile (overlap ≤ {q1:.3f}): n={low_mask.sum()}, '
          f'mean cross-pred AUC = {cp_arr[low_mask].mean():.4f} '
          f'(vs overall {cp_arr.mean():.4f})', flush=True)

    # Per-target stacking ΔAUC vs mean overlap rate (n=13)
    stack_dauc = []
    for ep in ENDPOINTS:
        b = np.array([r['AUC'] for r in baseline[ep]])
        s = np.array([r['AUC'] for r in s1[ep]])
        stack_dauc.append((s - b).mean())
    stack_dauc = np.array(stack_dauc)
    rho2, pval2 = spearmanr(stack_dauc, per_target_overlap_mean)
    print(f'  Stacking ΔAUC vs per-target overlap (n=13): ρ={rho2:.3f}, p={pval2:.4f}', flush=True)

    # === (d) DL-2 vs DL-4 ===
    print('\n--- (d) DL-2 vs DL-4 paired tests ---', flush=True)
    B, D2, D4 = [], [], []
    for ep in ENDPOINTS:
        b = np.array([r['AUC'] for r in baseline[ep]])
        d2 = np.array([r['AUC'] for r in dl2[ep]])
        d4 = np.array([r['AUC'] for r in dl4[ep]])
        B.append(b.mean()); D2.append(d2.mean()); D4.append(d4.mean())
    B = np.array(B); D2 = np.array(D2); D4 = np.array(D4)
    w_d2 = wilcoxon(D2, B); w_d4 = wilcoxon(D4, B); w_d2d4 = wilcoxon(D2, D4)
    print(f'  Baseline mean AUC      : {B.mean():.4f}', flush=True)
    print(f'  DL-2 merge mean AUC    : {D2.mean():.4f}  ΔAUC = {(D2-B).mean():+.4f}  Wilcoxon p={w_d2.pvalue:.4f}', flush=True)
    print(f'  DL-4 bootstrap mean AUC: {D4.mean():.4f}  ΔAUC = {(D4-B).mean():+.4f}  Wilcoxon p={w_d4.pvalue:.4f}', flush=True)
    print(f'  DL-2 minus DL-4 (cross-endpoint INFO alone) = {(D2-D4).mean():+.4f}  Wilcoxon p={w_d2d4.pvalue:.4f}', flush=True)

    out = {
        'per_target_overlap_mean': per_target_overlap_mean,
        'per_target_overlap_per_rep': overlap_any,
        'pair_overlap_matrix': pair_overlap,
        'cross_pred_vs_overlap_spearman': (rho, pval),
        'lowest_overlap_quartile_cross_pred_AUC_mean': float(cp_arr[low_mask].mean()),
        'lowest_overlap_quartile_threshold': float(q1),
        'stacking_dauc_vs_overlap_spearman': (rho2, pval2),
        'baseline_mean': B, 'dl2_mean': D2, 'dl4_mean': D4,
        'wilcoxon_d2_vs_baseline': (float(w_d2.statistic), float(w_d2.pvalue)),
        'wilcoxon_d4_vs_baseline': (float(w_d4.statistic), float(w_d4.pvalue)),
        'wilcoxon_d2_vs_d4':       (float(w_d2d4.statistic), float(w_d2d4.pvalue)),
        'dl2_minus_dl4_mean': float((D2-D4).mean()),
        'endpoints': ENDPOINTS,
    }
    out_path = os.path.join(RESULTS_DIR, 'scaffold_overlap_analysis.pkl')
    pickle.dump(out, open(out_path, 'wb'))
    print(f'\n✓ Saved: {out_path}', flush=True)
    print(f'Total: {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
