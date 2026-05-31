"""Root-cause diagnostic analyses for §5 (Why Data-Level Integration Fails)
under scaffold-disjoint primary protocol.

Four analyses mirror random-split protocol semantics (step5_new_strategies.py +
step2_strategies.py SHAP section), adapted to scaffold splits:

  (i)   Cross-endpoint prediction AUC matrix
        Source model = fixed-hyperparam LightGBM (n_est=300, depth=7, leaves=63,
        lr=0.05, subsample=0.8, colsample=0.8) — matches random-split rootcause
        protocol. Train on A's scaffold-split train, predict on B's scaffold-split
        test, single-class fallback = 0.5.
        Output shape: (N_REPS, 13, 13).

  (ii)  Feature importance divergence (Jensen–Shannon distance)
        SHAP from §2.1 baseline (full TPE 50×5fold) via LGBM pred_contrib
        (numerically equivalent to shap.TreeExplainer). Per (rep, endpoint),
        compute mean signed and mean abs SHAP. Pairwise JSD on L1-normalized
        mean-abs distributions: scipy.spatial.distance.jensenshannon (raw, no
        squaring — matches random-split step5 convention).
        Output: per_rep (N_REPS, 13, 13), upper-triangle 78-list (rep-averaged),
        mean/std of off-diagonal.

  (iii) Decision boundary visualization (t-SNE)
        For representative pairs, t-SNE on Morgan FPs of compounds appearing in
        BOTH endpoints' annotations. Coordinates are split-invariant; per-rep
        predictions from each endpoint's TPE baseline provide the coloring.
        Pairs: AR/ER (NHR family), TSHR/neuro (off-target), skin/Carcin
        (DL-3 negative pair), ocular/prenatal (DL-3 negative pair).

  (iv)  Top-10 SHAP substructure comparison
        Per (rep, endpoint), top-10 fingerprint bits by mean abs SHAP.
        Each top-10 entry is {fp_bit, mean_abs_shap, mean_shap, direction}.
        Pairwise: shared-bit count + directional concordance among shared bits.
        Output: per_rep top_k details + (N_REPS, 13, 13) shared / dir matrices.

Protocol note vs random-split:
  - Cross-pred model: identical fixed hyperparams (n=300, depth=7, ...) ✓
  - SHAP: TPE baseline (matches random-split semantic), but computed for ALL
    10 reps rather than rep-0-only (scaffold protocol upgrade) ✓
  - JSD: raw scipy.jensenshannon (= JS distance, not squared) ✓
  - Single-class fallback: 0.5 (matches random-split step5) ✓
  - Seed: 101+rep (consistent with other scaffold scripts; differs from random
    split's 42+rep; noted in SI) ✓
  - SHAP via LGBM pred_contrib instead of shap.TreeExplainer (numerically
    equivalent TreeSHAP algorithm; faster + zero extra deps) ✓

Outputs:
  results/scaffold_rootcause_tpe_baselines.pkl  -> dict[(rep, ep) -> {model, best_params,
                                                       oof_auc, test_auc}]
  results/scaffold_rootcause_xpred_models.pkl   -> dict[(rep, ep) -> fixed-param model]
  results/scaffold_rootcause_results.pkl        -> all 4 analyses
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
from scipy.spatial.distance import jensenshannon
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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

# Fixed cross-prediction hyperparams (matches random-split step5_new_strategies.py)
XPRED_PARAMS = dict(
    n_estimators=300, max_depth=7, num_leaves=63, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8
)


def train_tpe_baseline(X_tr, y_tr, seed, max_evals=50, n_folds=5):
    """Full §2.1 baseline: TPE 50×5fold. Returns (model, best_params, oof_auc)."""
    ne_c    = [100, 200, 300, 500, 700, 1000]
    md_c    = [3, 5, 7, 9, 11, -1]
    nl_c    = [15, 31, 63, 127, 255]
    lr_c    = [0.01, 0.03, 0.05, 0.1, 0.2]
    ch_c    = [5, 10, 20, 30, 50]
    sub_c   = [0.6, 0.7, 0.8, 0.9, 1.0]
    col_c   = [0.6, 0.7, 0.8, 0.9, 1.0]
    a_c     = [0, 0.01, 0.1, 1.0]
    l_c     = [0, 0.01, 0.1, 1.0]
    space = {
        'n_estimators':      hp.choice('n_estimators',      ne_c),
        'max_depth':         hp.choice('max_depth',         md_c),
        'num_leaves':        hp.choice('num_leaves',        nl_c),
        'learning_rate':     hp.choice('learning_rate',     lr_c),
        'min_child_samples': hp.choice('min_child_samples', ch_c),
        'subsample':         hp.choice('subsample',         sub_c),
        'colsample_bytree':  hp.choice('colsample_bytree',  col_c),
        'reg_alpha':         hp.choice('reg_alpha',         a_c),
        'reg_lambda':        hp.choice('reg_lambda',        l_c),
    }
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    def objective(params):
        scores = []
        for tr, va in skf.split(X_tr, y_tr):
            if len(np.unique(y_tr[va])) < 2:
                scores.append(0.5); continue
            m = lgb.LGBMClassifier(**params, random_state=seed, verbose=-1, n_jobs=-1)
            m.fit(X_tr[tr], y_tr[tr])
            scores.append(roc_auc_score(y_tr[va], m.predict_proba(X_tr[va])[:, 1]))
        return {'loss': -np.mean(scores), 'status': STATUS_OK}
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals,
                trials=trials, rstate=np.random.default_rng(seed), verbose=False)
    bp = {
        'n_estimators':      ne_c[best['n_estimators']],
        'max_depth':         md_c[best['max_depth']],
        'num_leaves':        nl_c[best['num_leaves']],
        'learning_rate':     lr_c[best['learning_rate']],
        'min_child_samples': ch_c[best['min_child_samples']],
        'subsample':         sub_c[best['subsample']],
        'colsample_bytree':  col_c[best['colsample_bytree']],
        'reg_alpha':         a_c[best['reg_alpha']],
        'reg_lambda':        l_c[best['reg_lambda']],
    }
    m = lgb.LGBMClassifier(**bp, random_state=seed, verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr)
    oof_auc = -min(t['result']['loss'] for t in trials.trials)
    return m, bp, float(oof_auc)


def train_xpred_lgb(X_tr, y_tr, seed):
    """Fixed-hyperparam diagnostic model — matches random-split step5_new_strategies."""
    m = lgb.LGBMClassifier(**XPRED_PARAMS, random_state=seed, verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return m


def shap_global_importance(model, X):
    """TreeSHAP via LGBM pred_contrib (numerically equivalent to shap.TreeExplainer).
    For binary classification returns (n, d+1) where the last column is the expected value;
    we drop it and return the per-feature SHAP matrix.
    Returns: mean_signed (d,), mean_abs (d,), raw SHAP matrix (n, d).
    """
    sv = model.predict(X, pred_contrib=True)
    sv = sv[:, :-1]  # drop expected_value column
    mean_signed = sv.mean(axis=0)
    mean_abs    = np.abs(sv).mean(axis=0)
    return mean_signed, mean_abs, sv


def main():
    print('=' * 70, flush=True)
    print(f'SCAFFOLD ROOT-CAUSE DIAGNOSTICS  (13 endpoints × N={N_REPS})', flush=True)
    print('=' * 70, flush=True)
    t0 = time.time()

    print('Loading datasets and scaffold splits...', flush=True)
    with open(os.path.join(RESULTS_DIR, 'datasets.pkl'), 'rb') as f:
        datasets = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_splits_record.pkl'), 'rb') as f:
        splits = pickle.load(f)

    tpe_models_path = os.path.join(RESULTS_DIR, 'scaffold_rootcause_tpe_baselines.pkl')
    xpred_models_path = os.path.join(RESULTS_DIR, 'scaffold_rootcause_xpred_models.pkl')
    results_path = os.path.join(RESULTS_DIR, 'scaffold_rootcause_results.pkl')

    tpe_models = load_dict(tpe_models_path)
    xpred_models = load_dict(xpred_models_path)
    results = load_dict(results_path)

    # ===========================================================================
    # STAGE A: Train TPE baselines (for SHAP-based analyses) — matches §2.1
    # ===========================================================================
    print('\n--- STAGE A: TPE baselines (50 evals × 5-fold) for SHAP analyses ---', flush=True)
    for rep in range(N_REPS):
        seed = 101 + rep
        tpe_models.setdefault(rep, {})
        for ep in ENDPOINTS:
            if ep in tpe_models[rep] and 'model' in tpe_models[rep][ep]:
                continue
            t_ep = time.time()
            tr_idx, te_idx = splits[ep][rep]
            X = datasets[ep]['X']; y = datasets[ep]['y']
            m, bp, oof_auc = train_tpe_baseline(X[tr_idx], y[tr_idx], seed)
            te_auc = float(roc_auc_score(y[te_idx], m.predict_proba(X[te_idx])[:, 1]))
            tpe_models[rep][ep] = dict(model=m, best_params=bp,
                                        oof_auc=oof_auc, test_auc=te_auc)
            save_dict_atomic(tpe_models, tpe_models_path)
            print(f'  rep{rep} {ep[:25]:<25} OOF={oof_auc:.4f} TEST={te_auc:.4f} '
                  f'[{time.time()-t_ep:.0f}s]', flush=True)
    print(f'  Stage A done [{(time.time()-t0)/60:.1f} min total]', flush=True)

    # ===========================================================================
    # STAGE B: Train fixed-param cross-pred models — matches random-split step5
    # ===========================================================================
    print('\n--- STAGE B: fixed-param diagnostic models (n=300, depth=7) for cross-pred ---', flush=True)
    for rep in range(N_REPS):
        seed = 101 + rep
        xpred_models.setdefault(rep, {})
        for ep in ENDPOINTS:
            if ep in xpred_models[rep]:
                continue
            t_ep = time.time()
            tr_idx, _ = splits[ep][rep]
            X = datasets[ep]['X']; y = datasets[ep]['y']
            m = train_xpred_lgb(X[tr_idx], y[tr_idx], seed)
            xpred_models[rep][ep] = m
            save_dict_atomic(xpred_models, xpred_models_path)
        print(f'  rep{rep} done [{time.time()-t_ep:.0f}s]', flush=True)
    print(f'  Stage B done [{(time.time()-t0)/60:.1f} min total]', flush=True)

    # ===========================================================================
    # STAGE C: Cross-prediction AUC matrix (N_REPS, 13, 13)
    #   Uses Stage B fixed-param models. Fallback = 0.5 (matches random-split).
    # ===========================================================================
    print('\n--- STAGE C: cross-prediction AUC matrix ---', flush=True)
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
                X_te_tgt = datasets[ep_tgt]['X'][te_idx_tgt]
                y_te_tgt = datasets[ep_tgt]['y'][te_idx_tgt]
                if len(np.unique(y_te_tgt)) < 2:
                    cross_per_rep[rep, i_src, i_tgt] = 0.5  # matches random-split
                    continue
                try:
                    yp = m.predict_proba(X_te_tgt)[:, 1]
                    cross_per_rep[rep, i_src, i_tgt] = float(roc_auc_score(y_te_tgt, yp))
                except Exception:
                    cross_per_rep[rep, i_src, i_tgt] = 0.5
        results['cross_pred_per_rep'] = cross_per_rep
        results['endpoints'] = ENDPOINTS
        save_dict_atomic(results, results_path)
        print(f'  rep{rep} done', flush=True)
    # Aggregates compatible with random-split rootcause_results.pkl schema
    cross_pred_mean_matrix = np.nanmean(cross_per_rep, axis=0)  # 13x13
    cross_pred_std_matrix  = np.nanstd(cross_per_rep, axis=0)
    mask_off = ~np.eye(13, dtype=bool)
    off_diag = cross_pred_mean_matrix[mask_off]
    results['cross_pred_aucs'] = cross_pred_mean_matrix  # 13x13 mean — matches random-split key
    results['cross_pred_mean'] = float(np.mean(off_diag))  # scalar off-diag mean
    results['cross_pred_std']  = float(np.std(off_diag))
    results['cross_pred_mean_matrix'] = cross_pred_mean_matrix
    results['cross_pred_std_matrix']  = cross_pred_std_matrix
    save_dict_atomic(results, results_path)
    print(f'  Stage C done. Off-diag mean = {results["cross_pred_mean"]:.4f} '
          f'+/- {results["cross_pred_std"]:.4f}', flush=True)

    # ===========================================================================
    # STAGE D: SHAP global importance (per rep × endpoint, from TPE baselines)
    # ===========================================================================
    print('\n--- STAGE D: SHAP global importance (TPE baselines) ---', flush=True)
    shap_data = results.get('shap_importance_per_rep')
    if shap_data is None:
        shap_data = {rep: {} for rep in range(N_REPS)}

    for rep in range(N_REPS):
        shap_data.setdefault(rep, {})
        for ep in ENDPOINTS:
            if ep in shap_data[rep]:
                continue
            t_ep = time.time()
            m = tpe_models[rep][ep]['model']
            tr_idx, _ = splits[ep][rep]
            X_tr = datasets[ep]['X'][tr_idx]
            mean_signed, mean_abs, _ = shap_global_importance(m, X_tr)
            shap_data[rep][ep] = dict(
                mean_signed=mean_signed.astype(np.float32),
                mean_abs=mean_abs.astype(np.float32))
            print(f'  rep{rep} {ep[:25]:<25} SHAP done [{time.time()-t_ep:.0f}s]', flush=True)
        results['shap_importance_per_rep'] = shap_data
        save_dict_atomic(results, results_path)

    # Rep-averaged shap_global_importance per endpoint (matches random-split schema)
    shap_global_avg = {}
    for ep in ENDPOINTS:
        arr = np.stack([shap_data[rep][ep]['mean_abs'] for rep in range(N_REPS)], axis=0)
        shap_global_avg[ep] = arr.mean(axis=0).astype(np.float32)
    results['shap_global_importance'] = shap_global_avg  # matches random-split key
    print(f'  Stage D done [{(time.time()-t0)/60:.1f} min total]', flush=True)

    # ===========================================================================
    # STAGE E: JS divergence (raw scipy.jensenshannon = JS distance)
    # ===========================================================================
    print('\n--- STAGE E: JS divergence ---', flush=True)
    js_per_rep = np.full((N_REPS, 13, 13), np.nan, dtype=np.float64)
    for rep in range(N_REPS):
        # L1-normalize each endpoint's mean-abs SHAP
        vecs = {}
        for ep in ENDPOINTS:
            fi = np.asarray(shap_data[rep][ep]['mean_abs'], dtype=np.float64)
            vecs[ep] = fi / (fi.sum() + 1e-12)
        for i, ep_a in enumerate(ENDPOINTS):
            for j, ep_b in enumerate(ENDPOINTS):
                # raw scipy.jensenshannon (returns JS distance, matches random-split convention)
                js_per_rep[rep, i, j] = float(jensenshannon(vecs[ep_a], vecs[ep_b]))
    results['js_per_rep'] = js_per_rep
    results['js_mean_matrix'] = np.nanmean(js_per_rep, axis=0)
    results['js_std_matrix']  = np.nanstd(js_per_rep, axis=0)

    # Upper-triangle 78-list aggregate (matches random-split schema)
    rep_avg_js = results['js_mean_matrix']
    js_upper_list = []
    for i in range(13):
        for j in range(i + 1, 13):
            js_upper_list.append(float(rep_avg_js[i, j]))
    results['js_divergences'] = js_upper_list  # 78-list, matches random-split key
    results['js_mean'] = float(np.mean(js_upper_list))
    results['js_std']  = float(np.std(js_upper_list))
    save_dict_atomic(results, results_path)
    print(f'  Stage E done. 78-pair upper-triangle: mean = {results["js_mean"]:.4f} '
          f'+/- {results["js_std"]:.4f}', flush=True)

    # ===========================================================================
    # STAGE F: Top-K SHAP substructures (dict list per endpoint + pairwise overlap)
    # ===========================================================================
    print('\n--- STAGE F: top-K SHAP substructures ---', flush=True)
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
            # dict list format matching random-split shap_results.pkl schema
            top_k_details[rep][ep] = [
                dict(
                    fp_bit=int(idx),
                    mean_abs_shap=float(abs_imp[idx]),
                    mean_shap=float(mean_sv[idx]),
                    direction='pro-toxic' if mean_sv[idx] > 0 else 'anti-toxic',
                )
                for idx in order
            ]
        # pairwise overlap and directional concordance
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
    # Rep-averaged shap_results-style dict (matches random-split shap_results.pkl using rep 0)
    results['shap_results_rep0'] = top_k_details[0]
    save_dict_atomic(results, results_path)
    print(f'  Stage F done', flush=True)

    # ===========================================================================
    # STAGE G: t-SNE for representative pairs
    # ===========================================================================
    print('\n--- STAGE G: t-SNE for representative pairs ---', flush=True)
    tsne_data = results.get('tsne_data', {})
    for pair in REPRESENTATIVE_PAIRS:
        pair_key = '|'.join(pair)
        if pair_key in tsne_data:
            print(f'  {pair_key}: cached', flush=True)
            continue
        t_pair = time.time()
        ep_a, ep_b = pair
        smiles_a = datasets[ep_a].get('smiles', None)
        smiles_b = datasets[ep_b].get('smiles', None)
        if smiles_a is None or smiles_b is None:
            print(f'  {pair_key}: SKIP (smiles missing)', flush=True)
            continue
        set_a = {s: i for i, s in enumerate(smiles_a)}
        set_b = {s: i for i, s in enumerate(smiles_b)}
        common = sorted(set(set_a.keys()) & set(set_b.keys()))
        if len(common) < 30:
            print(f'  {pair_key}: only {len(common)} overlap; SKIP', flush=True)
            continue
        idx_a = np.array([set_a[s] for s in common])
        idx_b = np.array([set_b[s] for s in common])
        X_overlap = datasets[ep_a]['X'][idx_a]
        y_a_overlap = datasets[ep_a]['y'][idx_a]
        y_b_overlap = datasets[ep_b]['y'][idx_b]
        perplexity = min(30, max(5, len(common) // 5))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca')
        coords = tsne.fit_transform(X_overlap.astype(np.float64))
        preds_a = np.zeros((N_REPS, len(common)), dtype=np.float32)
        preds_b = np.zeros((N_REPS, len(common)), dtype=np.float32)
        for rep in range(N_REPS):
            preds_a[rep] = tpe_models[rep][ep_a]['model'].predict_proba(X_overlap)[:, 1]
            preds_b[rep] = tpe_models[rep][ep_b]['model'].predict_proba(X_overlap)[:, 1]
        tsne_data[pair_key] = dict(
            smiles=common, coords=coords.astype(np.float32),
            y_a=y_a_overlap.astype(np.int8), y_b=y_b_overlap.astype(np.int8),
            preds_a=preds_a, preds_b=preds_b,
            ep_a=ep_a, ep_b=ep_b,
        )
        results['tsne_data'] = tsne_data
        save_dict_atomic(results, results_path)
        print(f'  {pair_key}: n_overlap={len(common)}, t-SNE done '
              f'[{time.time()-t_pair:.0f}s]', flush=True)

    # ===========================================================================
    # Final
    # ===========================================================================
    results['config'] = dict(
        n_reps=N_REPS, top_k=TOP_K,
        representative_pairs=REPRESENTATIVE_PAIRS,
        endpoints=ENDPOINTS,
        xpred_params=XPRED_PARAMS,
        notes=(
            "Scaffold-disjoint primary protocol root-cause analyses. "
            "Cross-pred fixed hyperparams match random-split step5_new_strategies.py. "
            "SHAP from TPE baselines (matches random-split step2 baseline semantics) "
            "for all 10 reps (vs random-split rep-0-only); JSD = raw scipy.jensenshannon "
            "(= JS distance, matches random-split). Single-class fallback = 0.5. "
            "Seed = 101+rep (consistent with other scaffold scripts)."
        ),
    )
    save_dict_atomic(results, results_path)
    print(f'\n✓ Total: {(time.time()-t0)/60:.1f} min', flush=True)
    print(f'✓ Saved: {results_path}', flush=True)
    print(f'✓ Saved: {tpe_models_path}', flush=True)
    print(f'✓ Saved: {xpred_models_path}', flush=True)


if __name__ == '__main__':
    main()
