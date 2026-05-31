"""Root-cause diagnostics on MORDRED descriptors under scaffold splits.

Parallel to step_scaffold_rootcause.py but with two simplifications:
  1. SHAP comes from FIXED-hp LGBM (n_est=300, depth=7, ...) instead of TPE
     baselines — saves ~30h of recomputation; the fixed-hp SHAP is a valid
     mechanistic snapshot for SMD/JSD comparison purposes.
  2. Same fixed-hp models serve BOTH cross-prediction AND SHAP (one fit per
     (ep, rep), reused).

Computes:
  - Cross-prediction AUC matrix (N_REPS, 13, 13) on Mordred raw 1338-D
  - SHAP global importance via LGBM pred_contrib (1338-D)
  - JS divergence on L1-normalized mean-abs SHAP (78 pairs)
  - Top-K SHAP feature comparison (k=10)
  - t-SNE coords for representative pairs (z-scored 690-D for stable geometry)

Outputs: results/scaffold_diagnostics_mordred.pkl

X choice:
  - LGBM (cross-pred + SHAP):  X_mordred_raw   (1338-D, no scaling)
  - t-SNE geometry:            X_mordred_z     (690-D z-scored, preserves shape)
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from scipy.spatial.distance import jensenshannon
import lightgbm as lgb

sys.path.insert(0, r'D:\quxintong\scripts')
from _checkpoint import load_dict, save_dict_atomic

RESULTS_DIR = r'D:\quxintong\results'
N_REPS = 10
TOP_K = 10

ENDPOINTS = [
    'prenatal_development_C', 'TSHR_agonist_activity_C',
    'respiratory_toxicity_C', 'ocular_toxicity_C',
    'ames_mutagenicity_C', 'reproductive_toxicity_C',
    'skin_corrosion_C', 'neurotoxicity_C',
    'Estrogen_Receptor_α_C', 'Androgen_Receptor_C',
    'cytotoxicity_C', 'Carcinogenicity_C', 'Hepatotoxicity_C'
]

REPRESENTATIVE_PAIRS = [
    ('Androgen_Receptor_C', 'Estrogen_Receptor_α_C'),
    ('TSHR_agonist_activity_C', 'neurotoxicity_C'),
    ('skin_corrosion_C', 'Carcinogenicity_C'),
    ('ocular_toxicity_C', 'prenatal_development_C'),
]

XPRED_PARAMS = dict(
    n_estimators=300, max_depth=7, num_leaves=63, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8
)

X_KEY_RAW = 'X_mordred_raw'   # 1338-D for LGBM (cross-pred + SHAP)
X_KEY_Z   = 'X_mordred_z'     # 690-D z-scored for t-SNE


def train_xpred_lgb(X_tr, y_tr, seed):
    m = lgb.LGBMClassifier(**XPRED_PARAMS, random_state=seed, verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return m


def shap_global(model, X):
    sv = model.predict(X, pred_contrib=True)
    sv = sv[:, :-1]
    return sv.mean(axis=0), np.abs(sv).mean(axis=0)


def main():
    print('=' * 70, flush=True)
    print(f'MORDRED ROOT-CAUSE DIAGNOSTICS  (13 endpoints × N={N_REPS})', flush=True)
    print('=' * 70, flush=True)
    t0 = time.time()

    print('Loading mordred_datasets.pkl + scaffold splits...', flush=True)
    with open(os.path.join(RESULTS_DIR, 'mordred_datasets.pkl'), 'rb') as f:
        mordred = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_splits_record.pkl'), 'rb') as f:
        splits = pickle.load(f)

    xpred_path = os.path.join(RESULTS_DIR, 'scaffold_diagnostics_mordred_xpred_models.pkl')
    results_path = os.path.join(RESULTS_DIR, 'scaffold_diagnostics_mordred.pkl')

    xpred_models = load_dict(xpred_path)
    results = load_dict(results_path)

    # ===========================================================================
    # STAGE A: Train fixed-hp Mordred LGBM per (rep, ep) — reused for cross-pred + SHAP
    # ===========================================================================
    print('\n--- STAGE A: fixed-hp Mordred LGBM (n=300, depth=7) ---', flush=True)
    for rep in range(N_REPS):
        seed = 101 + rep
        xpred_models.setdefault(rep, {})
        for ep in ENDPOINTS:
            if ep in xpred_models[rep]:
                continue
            t_ep = time.time()
            tr_idx, _ = splits[ep][rep]
            X = mordred[ep][X_KEY_RAW]; y = mordred[ep]['y']
            m = train_xpred_lgb(X[tr_idx], y[tr_idx], seed)
            xpred_models[rep][ep] = m
            print(f'  rep{rep} {ep[:25]:<25} fit [{time.time()-t_ep:.0f}s]', flush=True)
        save_dict_atomic(xpred_models, xpred_path)
    print(f'  Stage A done [{(time.time()-t0)/60:.1f} min]', flush=True)

    # ===========================================================================
    # STAGE B: Cross-prediction AUC matrix
    # ===========================================================================
    print('\n--- STAGE B: cross-prediction AUC matrix ---', flush=True)
    cross_per_rep = results.get('cross_pred_per_rep')
    if cross_per_rep is None or np.asarray(cross_per_rep).shape != (N_REPS, 13, 13):
        cross_per_rep = np.full((N_REPS, 13, 13), np.nan, dtype=np.float64)
    cross_per_rep = np.asarray(cross_per_rep, dtype=np.float64)

    for rep in range(N_REPS):
        for i_src, ep_src in enumerate(ENDPOINTS):
            m = xpred_models[rep][ep_src]
            for i_tgt, ep_tgt in enumerate(ENDPOINTS):
                if not np.isnan(cross_per_rep[rep, i_src, i_tgt]):
                    continue
                _, te_idx_tgt = splits[ep_tgt][rep]
                X_te_tgt = mordred[ep_tgt][X_KEY_RAW][te_idx_tgt]
                y_te_tgt = mordred[ep_tgt]['y'][te_idx_tgt]
                if len(np.unique(y_te_tgt)) < 2:
                    cross_per_rep[rep, i_src, i_tgt] = 0.5
                    continue
                try:
                    yp = m.predict_proba(X_te_tgt)[:, 1]
                    cross_per_rep[rep, i_src, i_tgt] = float(roc_auc_score(y_te_tgt, yp))
                except Exception:
                    cross_per_rep[rep, i_src, i_tgt] = 0.5
        results['cross_pred_per_rep'] = cross_per_rep
        save_dict_atomic(results, results_path)
        print(f'  rep{rep} done', flush=True)

    cross_pred_mean = np.nanmean(cross_per_rep, axis=0)
    cross_pred_std  = np.nanstd(cross_per_rep, axis=0)
    mask_off = ~np.eye(13, dtype=bool)
    off_diag = cross_pred_mean[mask_off]
    results['cross_pred_aucs'] = cross_pred_mean
    results['cross_pred_mean_matrix'] = cross_pred_mean
    results['cross_pred_std_matrix']  = cross_pred_std
    results['cross_pred_mean'] = float(np.mean(off_diag))
    results['cross_pred_std']  = float(np.std(off_diag))
    results['endpoints'] = ENDPOINTS
    save_dict_atomic(results, results_path)
    print(f'  Stage B done. Off-diag mean = {results["cross_pred_mean"]:.4f} '
          f'+/- {results["cross_pred_std"]:.4f}', flush=True)

    # ===========================================================================
    # STAGE C: SHAP via pred_contrib on fixed-hp Mordred models
    # ===========================================================================
    print('\n--- STAGE C: SHAP global importance (fixed-hp Mordred) ---', flush=True)
    shap_data = results.get('shap_importance_per_rep', {rep: {} for rep in range(N_REPS)})
    for rep in range(N_REPS):
        shap_data.setdefault(rep, {})
        for ep in ENDPOINTS:
            if ep in shap_data[rep]:
                continue
            t_ep = time.time()
            m = xpred_models[rep][ep]
            tr_idx, _ = splits[ep][rep]
            X_tr = mordred[ep][X_KEY_RAW][tr_idx]
            mean_signed, mean_abs = shap_global(m, X_tr)
            shap_data[rep][ep] = dict(
                mean_signed=mean_signed.astype(np.float32),
                mean_abs=mean_abs.astype(np.float32))
            print(f'  rep{rep} {ep[:25]:<25} SHAP [{time.time()-t_ep:.0f}s]', flush=True)
        results['shap_importance_per_rep'] = shap_data
        save_dict_atomic(results, results_path)

    shap_global_avg = {}
    for ep in ENDPOINTS:
        arr = np.stack([shap_data[rep][ep]['mean_abs'] for rep in range(N_REPS)], axis=0)
        shap_global_avg[ep] = arr.mean(axis=0).astype(np.float32)
    results['shap_global_importance'] = shap_global_avg
    print(f'  Stage C done [{(time.time()-t0)/60:.1f} min]', flush=True)

    # ===========================================================================
    # STAGE D: JS divergence on Mordred SHAP mean-abs
    # ===========================================================================
    print('\n--- STAGE D: JS divergence ---', flush=True)
    js_per_rep = np.full((N_REPS, 13, 13), np.nan, dtype=np.float64)
    for rep in range(N_REPS):
        vecs = {}
        for ep in ENDPOINTS:
            fi = np.asarray(shap_data[rep][ep]['mean_abs'], dtype=np.float64)
            vecs[ep] = fi / (fi.sum() + 1e-12)
        for i, ep_a in enumerate(ENDPOINTS):
            for j, ep_b in enumerate(ENDPOINTS):
                js_per_rep[rep, i, j] = float(jensenshannon(vecs[ep_a], vecs[ep_b]))
    results['js_per_rep'] = js_per_rep
    results['js_mean_matrix'] = np.nanmean(js_per_rep, axis=0)
    results['js_std_matrix']  = np.nanstd(js_per_rep, axis=0)
    rep_avg_js = results['js_mean_matrix']
    js_upper = [float(rep_avg_js[i, j]) for i in range(13) for j in range(i+1, 13)]
    results['js_divergences'] = js_upper
    results['js_mean'] = float(np.mean(js_upper))
    results['js_std']  = float(np.std(js_upper))
    save_dict_atomic(results, results_path)
    print(f'  Stage D done. 78-pair JS: mean={results["js_mean"]:.4f} '
          f'+/- {results["js_std"]:.4f}', flush=True)

    # ===========================================================================
    # STAGE E: Top-K Mordred-feature comparison (parallel to FP top-K bits)
    # ===========================================================================
    print('\n--- STAGE E: Top-K Mordred SHAP features ---', flush=True)
    top_k_details = {rep: {} for rep in range(N_REPS)}
    top_k_idx = np.full((N_REPS, 13, TOP_K), -1, dtype=np.int32)
    top_k_shared = np.full((N_REPS, 13, 13), 0, dtype=np.int32)
    top_k_dir = np.full((N_REPS, 13, 13), np.nan, dtype=np.float64)

    for rep in range(N_REPS):
        for i, ep in enumerate(ENDPOINTS):
            abs_imp = shap_data[rep][ep]['mean_abs']
            mean_sv = shap_data[rep][ep]['mean_signed']
            order = np.argsort(abs_imp)[::-1][:TOP_K]
            top_k_idx[rep, i, :] = order
            top_k_details[rep][ep] = [
                dict(feature_idx=int(idx),
                     mean_abs_shap=float(abs_imp[idx]),
                     mean_shap=float(mean_sv[idx]),
                     direction='pro-toxic' if mean_sv[idx] > 0 else 'anti-toxic')
                for idx in order
            ]
        for i, ep_a in enumerate(ENDPOINTS):
            for j, ep_b in enumerate(ENDPOINTS):
                sa = set(top_k_idx[rep, i, :].tolist())
                sb = set(top_k_idx[rep, j, :].tolist())
                shared = sa & sb
                top_k_shared[rep, i, j] = len(shared)
                if len(shared) > 0:
                    sgn_a = shap_data[rep][ep_a]['mean_signed']
                    sgn_b = shap_data[rep][ep_b]['mean_signed']
                    concords = sum(int(np.sign(sgn_a[k]) == np.sign(sgn_b[k])) for k in shared)
                    top_k_dir[rep, i, j] = concords / len(shared)
    results['top_k_details_per_rep'] = top_k_details
    results['top_k_idx'] = top_k_idx
    results['top_k_shared'] = top_k_shared
    results['top_k_dir_concord'] = top_k_dir
    save_dict_atomic(results, results_path)
    print(f'  Stage E done', flush=True)

    # ===========================================================================
    # STAGE F: t-SNE for representative pairs (use z-scored 690-D)
    # ===========================================================================
    print('\n--- STAGE F: t-SNE for representative pairs (z-scored 690-D) ---', flush=True)
    tsne_data = results.get('tsne_data', {})
    for pair in REPRESENTATIVE_PAIRS:
        pair_key = '|'.join(pair)
        if pair_key in tsne_data:
            print(f'  {pair_key}: cached', flush=True); continue
        t_pair = time.time()
        ep_a, ep_b = pair
        smiles_a = mordred[ep_a].get('smiles', None)
        smiles_b = mordred[ep_b].get('smiles', None)
        if smiles_a is None or smiles_b is None:
            print(f'  {pair_key}: SKIP (smiles missing)', flush=True); continue
        set_a = {s: i for i, s in enumerate(smiles_a)}
        set_b = {s: i for i, s in enumerate(smiles_b)}
        common = sorted(set(set_a.keys()) & set(set_b.keys()))
        if len(common) < 30:
            print(f'  {pair_key}: only {len(common)} overlap; SKIP', flush=True); continue
        idx_a = np.array([set_a[s] for s in common])
        idx_b = np.array([set_b[s] for s in common])
        X_overlap = mordred[ep_a][X_KEY_Z][idx_a]
        y_a_overlap = mordred[ep_a]['y'][idx_a]
        y_b_overlap = mordred[ep_b]['y'][idx_b]
        perplexity = min(30, max(5, len(common) // 5))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca')
        coords = tsne.fit_transform(X_overlap.astype(np.float64))
        # Predict using each ep's fixed-hp Mordred LGBM (raw features for prediction)
        X_overlap_raw = mordred[ep_a][X_KEY_RAW][idx_a]
        preds_a = np.zeros((N_REPS, len(common)), dtype=np.float32)
        preds_b = np.zeros((N_REPS, len(common)), dtype=np.float32)
        for rep in range(N_REPS):
            preds_a[rep] = xpred_models[rep][ep_a].predict_proba(X_overlap_raw)[:, 1]
            preds_b[rep] = xpred_models[rep][ep_b].predict_proba(X_overlap_raw)[:, 1]
        tsne_data[pair_key] = dict(
            smiles=common, coords=coords.astype(np.float32),
            y_a=y_a_overlap.astype(np.int8), y_b=y_b_overlap.astype(np.int8),
            preds_a=preds_a, preds_b=preds_b,
            ep_a=ep_a, ep_b=ep_b,
        )
        results['tsne_data'] = tsne_data
        save_dict_atomic(results, results_path)
        print(f'  {pair_key}: n_overlap={len(common)} [{time.time()-t_pair:.0f}s]', flush=True)

    results['config'] = dict(
        n_reps=N_REPS, top_k=TOP_K,
        representative_pairs=REPRESENTATIVE_PAIRS,
        endpoints=ENDPOINTS,
        xpred_params=XPRED_PARAMS,
        notes=(
            "Mordred root-cause diagnostics. SHAP from fixed-hp LightGBM "
            "(n_est=300, depth=7, lr=0.05) — NOT TPE-tuned, to avoid re-running "
            "the ~30h baseline retraining. The same fixed-hp models serve both "
            "cross-prediction and SHAP/JSD/top-K. JSD = raw scipy.jensenshannon. "
            "t-SNE on z-scored 690-D for stable geometry; predictions for color "
            "use raw 1338-D (matches LGBM training distribution)."
        ),
    )
    save_dict_atomic(results, results_path)
    print(f'\n✓ Total: {(time.time()-t0)/60:.1f} min', flush=True)
    print(f'✓ Saved: {results_path}', flush=True)


if __name__ == '__main__':
    main()
