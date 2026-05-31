"""DL-4 Bootstrap Control under SCAFFOLD-DISJOINT splits.

Purpose: isolate cross-endpoint information from training-set-size in DL-2 global
merge. For each target endpoint:
  1. Compute N_global = sum of train sizes across all 13 endpoints (per rep)
  2. Bootstrap-resample N_global compounds (with replacement) from the target's
     OWN scaffold-split training set
  3. Train LightGBM with same TPE 50 evals + 5-fold protocol as baseline
  4. Evaluate on the target's scaffold test set (paired w/ baseline & DL-2)

This holds training-set size constant w.r.t. DL-2 but removes all cross-endpoint
information. A null result here implies DL-2's deviations come from cross-endpoint
information (not the size bump).

Output: results/scaffold_dl4_bootstrap.pkl
  Schema: dict[endpoint -> list[N_REPS] of {AUC, Accuracy, Precision, Recall, F1,
                                            N_bootstrap, N_train_original}]
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                              recall_score, f1_score)
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

sys.path.insert(0, r'D:\quxintong\scripts')
from _checkpoint import load_dict, save_dict_atomic

RESULTS_DIR = r'D:\quxintong\results'
N_REPS = 10

ENDPOINTS = [
    'prenatal_development_C', 'TSHR_agonist_activity_C',
    'respiratory_toxicity_C', 'ocular_toxicity_C',
    'ames_mutagenicity_C', 'reproductive_toxicity_C',
    'skin_corrosion_C', 'neurotoxicity_C',
    'Estrogen_Receptor_α_C', 'Androgen_Receptor_C',
    'cytotoxicity_C', 'Carcinogenicity_C', 'Hepatotoxicity_C'
]


def evaluate_preds(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'AUC': float(roc_auc_score(y_true, y_prob)),
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_lgb(X_tr, y_tr, X_te, y_te, seed, max_evals=50, n_folds=5):
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
        p = dict(params)
        scores = []
        for tr, va in skf.split(X_tr, y_tr):
            if len(np.unique(y_tr[va])) < 2:
                scores.append(0.5); continue
            m = lgb.LGBMClassifier(**p, random_state=seed, verbose=-1, n_jobs=-1)
            m.fit(X_tr[tr], y_tr[tr])
            scores.append(roc_auc_score(y_tr[va], m.predict_proba(X_tr[va])[:, 1]))
        return {'loss': -np.mean(scores), 'status': STATUS_OK}
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals,
                trials=trials, rstate=np.random.default_rng(seed), verbose=False)
    bp = {
        'n_estimators': ne_c[best['n_estimators']],
        'max_depth':    md_c[best['max_depth']],
        'num_leaves':   nl_c[best['num_leaves']],
        'learning_rate': lr_c[best['learning_rate']],
        'min_child_samples': ch_c[best['min_child_samples']],
        'subsample':    sub_c[best['subsample']],
        'colsample_bytree': col_c[best['colsample_bytree']],
        'reg_alpha':    a_c[best['reg_alpha']],
        'reg_lambda':   l_c[best['reg_lambda']],
    }
    m = lgb.LGBMClassifier(**bp, random_state=seed, verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr)
    yp = m.predict_proba(X_te)[:, 1]
    return evaluate_preds(y_te, yp)


def compute_n_global(datasets, splits, rep):
    """N_global for this rep = sum of train sizes across 13 endpoints."""
    return sum(len(splits[ep][rep][0]) for ep in ENDPOINTS)


def main():
    print("=" * 70, flush=True)
    print(f"SCAFFOLD DL-4 BOOTSTRAP CONTROL  (13 endpoints × N={N_REPS})", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    print("Loading datasets and scaffold splits...", flush=True)
    with open(os.path.join(RESULTS_DIR, 'datasets.pkl'), 'rb') as f:
        datasets = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_splits_record.pkl'), 'rb') as f:
        splits = pickle.load(f)

    # Pre-compute N_global per rep (constant across endpoints within a rep)
    n_global = [compute_n_global(datasets, splits, rep) for rep in range(N_REPS)]
    print(f"N_global per rep: {n_global}", flush=True)

    out_path = os.path.join(RESULTS_DIR, 'scaffold_dl4_bootstrap.pkl')
    results = load_dict(out_path)

    for ep in ENDPOINTS:
        results.setdefault(ep, [])
        start = len(results[ep])
        if start >= N_REPS:
            print(f"  {ep[:30]:<30} already complete", flush=True)
            continue
        X = datasets[ep]['X']
        y = datasets[ep]['y']
        for rep in range(start, N_REPS):
            t_rep = time.time()
            seed = 101 + rep
            tr_idx, te_idx = splits[ep][rep]
            X_tr_full, y_tr_full = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]
            n_boot = n_global[rep]

            # Bootstrap-resample (with replacement) n_boot compounds from
            # target's OWN train set. Use a seeded RNG for reproducibility.
            rng = np.random.default_rng(seed * 1000 + ENDPOINTS.index(ep))
            boot_idx = rng.integers(0, len(X_tr_full), size=n_boot)
            X_tr_boot, y_tr_boot = X_tr_full[boot_idx], y_tr_full[boot_idx]

            # Sanity: both classes present (rare to fail given large n_boot)
            if len(np.unique(y_tr_boot)) < 2:
                print(f"  {ep[:30]:<30} rep{rep}: SKIP (bootstrap single-class)", flush=True)
                continue

            metrics = train_lgb(X_tr_boot, y_tr_boot, X_te, y_te, seed)
            metrics['N_bootstrap'] = int(n_boot)
            metrics['N_train_original'] = int(len(tr_idx))
            results[ep].append(metrics)
            save_dict_atomic(results, out_path)
            print(f"  {ep[:30]:<30} rep{rep}: AUC={metrics['AUC']:.4f}  "
                  f"(N_boot={n_boot})  [{time.time()-t_rep:.0f}s]", flush=True)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
