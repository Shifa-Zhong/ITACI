"""Clean-source stacking control under scaffold-disjoint splits — v2.

This v2 STRICTLY MATCHES the dirty S1 stacking protocol (step_scaffold_stacking.py)
in every aspect except the source-cleaning step. This makes the comparison
"dirty S1 vs clean-source S1" a clean apples-to-apples test of compound-overlap
leakage.

Protocol (mirrors step_scaffold_stacking.py line-by-line):
  - Source models (12 per target × 10 reps): FIXED hp n_est=500, depth=7,
    leaves=63, lr=0.05, subsample=0.8, colsample=0.8 — same as dirty S1 source.
  - Source training data: scaffold-split TRAIN of each source endpoint,
    BUT with target's scaffold-split TEST compounds REMOVED by SMILES.
  - Stacking model: train_lgb_hyperopt TPE 15 evals + 3-fold CV on the
    augmented (X_tr + 12 source preds) — same as dirty S1.
  - Baseline branch: yp_base loaded from scaffold_baseline_results.pkl
    [ep][rep]['y_pred'] — the unified §2.1 TPE 50+5fold baseline — same as dirty S1.
  - OOF λ-grid: {0.1, 0.2, ..., 0.8} via 3-fold CV — same as dirty S1.
  - Final ensemble: yp_ens = best_lam * yp_base + (1 - best_lam) * yp_stack.

Output: scaffold_clean_source_v2.pkl (the prior scaffold_clean_source.pkl is
retained as scaffold_clean_source.pkl.archive_v1).
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
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


def train_lgb_quick(X_tr, y_tr, X_te, seed):
    """Fixed-hp LGBM for OOF λ-grid CV (matches step_scaffold_stacking.py)."""
    m = lgb.LGBMClassifier(
        n_estimators=300, max_depth=7, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def train_lgb_hyperopt(X_tr, y_tr, X_te, seed, max_evals=15, n_folds=3):
    """TPE 15 evals + 3-fold CV for the stacking model on augmented features
    (mirrors step_scaffold_stacking.py:train_lgb_hyperopt)."""
    ne_choices    = [100, 200, 300, 500, 700, 1000]
    md_choices    = [3, 5, 7, 9, 11, -1]
    nl_choices    = [15, 31, 63, 127, 255]
    lr_choices    = [0.01, 0.03, 0.05, 0.1, 0.2]
    child_choices = [5, 10, 20, 30, 50]
    sub_choices   = [0.6, 0.7, 0.8, 0.9, 1.0]
    col_choices   = [0.6, 0.7, 0.8, 0.9, 1.0]
    a_choices     = [0, 0.01, 0.1, 1.0]
    l_choices     = [0, 0.01, 0.1, 1.0]
    space = {
        'n_estimators':      hp.choice('n_estimators',      ne_choices),
        'max_depth':         hp.choice('max_depth',         md_choices),
        'num_leaves':        hp.choice('num_leaves',        nl_choices),
        'learning_rate':     hp.choice('learning_rate',     lr_choices),
        'min_child_samples': hp.choice('min_child_samples', child_choices),
        'subsample':         hp.choice('subsample',         sub_choices),
        'colsample_bytree':  hp.choice('colsample_bytree',  col_choices),
        'reg_alpha':         hp.choice('reg_alpha',         a_choices),
        'reg_lambda':        hp.choice('reg_lambda',        l_choices),
    }
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    def objective(params):
        p = dict(params); scores = []
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
        'n_estimators':       ne_choices[best['n_estimators']],
        'max_depth':          md_choices[best['max_depth']],
        'num_leaves':         nl_choices[best['num_leaves']],
        'learning_rate':      lr_choices[best['learning_rate']],
        'min_child_samples':  child_choices[best['min_child_samples']],
        'subsample':          sub_choices[best['subsample']],
        'colsample_bytree':   col_choices[best['colsample_bytree']],
        'reg_alpha':          a_choices[best['reg_alpha']],
        'reg_lambda':         l_choices[best['reg_lambda']],
    }
    m = lgb.LGBMClassifier(**bp, random_state=seed, verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def main():
    print("=" * 70, flush=True)
    print(f"SCAFFOLD CLEAN-SOURCE STACKING (v2, MATCHED PROTOCOL)  13 × N={N_REPS}", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    print("Loading datasets, scaffold splits, §2.1 baseline preds...", flush=True)
    with open(os.path.join(RESULTS_DIR, 'datasets.pkl'), 'rb') as f:
        datasets = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_splits_record.pkl'), 'rb') as f:
        splits = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_baseline_results.pkl'), 'rb') as f:
        base_preds = pickle.load(f)
    assert all('y_pred' in base_preds[ep][0] for ep in ENDPOINTS), \
        "baseline pkl missing y_pred — required for §2.1-matched ensemble"

    out_path = os.path.join(RESULTS_DIR, 'scaffold_clean_source_v2.pkl')
    results = load_dict(out_path)

    for rep in range(N_REPS):
        seed = 101 + rep
        for ep_target in ENDPOINTS:
            results.setdefault(ep_target, [])
            if len(results[ep_target]) > rep:
                continue
            t_ep = time.time()
            X_t = datasets[ep_target]['X']
            y_t = datasets[ep_target]['y']
            smi_t = datasets[ep_target]['smiles']
            tr_idx, te_idx = splits[ep_target][rep]
            X_tr, X_te = X_t[tr_idx], X_t[te_idx]
            y_tr, y_te = y_t[tr_idx], y_t[te_idx]
            test_smi_set = set([smi_t[i] for i in te_idx])

            # Train 12 cleaned source models (FIXED hp, source per-rep scaffold train minus target test)
            src_preds_tr, src_preds_te = [], []
            for ep_src in ENDPOINTS:
                if ep_src == ep_target: continue
                tr_idx_src, _ = splits[ep_src][rep]
                X_src = datasets[ep_src]['X'][tr_idx_src]
                y_src = datasets[ep_src]['y'][tr_idx_src]
                smi_src = [datasets[ep_src]['smiles'][i] for i in tr_idx_src]
                mask = np.array([s not in test_smi_set for s in smi_src], dtype=bool)
                X_src_clean = X_src[mask]
                y_src_clean = y_src[mask]
                if len(np.unique(y_src_clean)) < 2 or len(X_src_clean) < 50:
                    src_preds_tr.append(np.full(len(X_tr), 0.5))
                    src_preds_te.append(np.full(len(X_te), 0.5))
                    continue
                m = lgb.LGBMClassifier(
                    n_estimators=500, max_depth=7, num_leaves=63,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    random_state=seed, verbose=-1, n_jobs=-1)
                m.fit(X_src_clean, y_src_clean)
                src_preds_tr.append(m.predict_proba(X_tr)[:, 1])
                src_preds_te.append(m.predict_proba(X_te)[:, 1])
            X_tr_aug = np.hstack([X_tr, np.column_stack(src_preds_tr)])
            X_te_aug = np.hstack([X_te, np.column_stack(src_preds_te)])

            # OOF λ-tuning (MATCHES DIRTY S1 EXACTLY: train_lgb_quick + grid {0.1..0.8})
            oof_base = np.zeros(len(y_tr))
            oof_stack = np.zeros(len(y_tr))
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            for inner_tr, inner_va in skf.split(X_tr, y_tr):
                oof_base[inner_va]  = train_lgb_quick(X_tr[inner_tr],     y_tr[inner_tr], X_tr[inner_va],     seed)
                oof_stack[inner_va] = train_lgb_quick(X_tr_aug[inner_tr], y_tr[inner_tr], X_tr_aug[inner_va], seed)

            best_lam, best_auc = 0.5, 0
            for lam in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                try:
                    auc = roc_auc_score(y_tr, lam * oof_base + (1 - lam) * oof_stack)
                    if auc > best_auc:
                        best_auc = auc; best_lam = lam
                except Exception:
                    pass

            # Final ensemble: yp_base from §2.1 TPE50+5fold baseline pkl + yp_stack via TPE15 on aug
            yp_base  = np.asarray(base_preds[ep_target][rep]['y_pred'], dtype=np.float32)
            yp_stack = train_lgb_hyperopt(X_tr_aug, y_tr, X_te_aug, seed)
            yp_ens   = best_lam * yp_base + (1 - best_lam) * yp_stack
            metrics  = evaluate_preds(y_te, yp_ens)
            metrics['lambda'] = float(best_lam)
            metrics['baseline_AUC'] = float(roc_auc_score(y_te, yp_base))
            results[ep_target].append(metrics)
            save_dict_atomic(results, out_path)
            print(f"  rep{rep} {ep_target[:25]:<25} CLEAN: B={metrics['baseline_AUC']:.4f} "
                  f"Ens(λ={best_lam:.1f})={metrics['AUC']:.4f} "
                  f"d={metrics['AUC']-metrics['baseline_AUC']:+.4f} "
                  f"[{time.time()-t_ep:.0f}s]", flush=True)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
