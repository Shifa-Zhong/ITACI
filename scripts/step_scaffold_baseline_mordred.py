"""Mordred-feature LightGBM baseline on scaffold-split (N=10) for all 13 endpoints.

Direct parallel to step_scaffold_baseline.py for the Mordred-descriptor extension
(Stage 0 of the cross-representation staged plan).

CRITICAL COMPARABILITY GUARANTEES — every aspect mirrors the FP baseline:
  - Splits      : LOADS scaffold_splits_record.pkl unchanged (same train/test
                   indices, same Bemis-Murcko scaffold groups, same per-rep seeds 101-110)
  - HPO         : TPE max_evals=50, 5-fold StratifiedKFold inner CV (seeded with `seed`)
  - HP space    : IDENTICAL 9 hp.choice ranges as step_scaffold_baseline.py
  - Random seed : np.random.default_rng(seed) for fmin, same as FP
  - Refit       : best HP on full train, predict test, return identical metrics dict

Only diff from FP baseline:
  - X input : mordred_datasets[ep]['X_mordred_raw']  shape (n, 1338)
              instead of datasets[ep]['X']           shape (n, 2048)
  - Output  : scaffold_baseline_mordred_results.pkl  (parallel to scaffold_baseline_results.pkl)
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')  # handle α in Estrogen_Receptor_α_C
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
        'AUC':       float(roc_auc_score(y_true, y_prob)),
        'Accuracy':  float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'F1':        float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_baseline_lgb(X_tr, y_tr, X_te, y_te, seed,
                        max_evals=50, n_folds=5):
    """IDENTICAL to step_scaffold_baseline.py:train_baseline_lgb. Do NOT diverge."""
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
    yp = m.predict_proba(X_te)[:, 1]
    metrics = evaluate_preds(y_te, yp)
    metrics['y_pred'] = yp.astype(np.float32)
    return metrics, yp


def main():
    print("=" * 70, flush=True)
    print(f"SCAFFOLD BASELINE — MORDRED  (13 endpoints × N={N_REPS})", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    print("Loading mordred_datasets.pkl and scaffold_splits_record.pkl...", flush=True)
    with open(os.path.join(RESULTS_DIR, 'mordred_datasets.pkl'), 'rb') as f:
        mordred = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_splits_record.pkl'), 'rb') as f:
        splits = pickle.load(f)

    # Sanity check: every endpoint has 10 splits and X_mordred_raw rows align with y
    for ep in ENDPOINTS:
        Xm = mordred[ep]['X_mordred_raw']
        ym = mordred[ep]['y']
        assert Xm.shape[0] == len(ym), f'{ep}: X/y row mismatch'
        assert len(splits[ep]) >= N_REPS, f'{ep}: need {N_REPS} splits, found {len(splits[ep])}'
        # Spot check: max idx in any split is in range
        for tr, te in splits[ep]:
            assert int(tr.max()) < Xm.shape[0] and int(te.max()) < Xm.shape[0], \
                f'{ep}: split index OOB for Mordred matrix'
    print(f"  comparability check passed: splits map cleanly to Mordred matrices", flush=True)

    out_path = os.path.join(RESULTS_DIR, 'scaffold_baseline_mordred_results.pkl')
    results = load_dict(out_path)

    for ep in ENDPOINTS:
        results.setdefault(ep, [])
        start = len(results[ep])
        if start >= N_REPS:
            print(f"  {ep[:30]:<30} already complete", flush=True)
            continue
        X = mordred[ep]['X_mordred_raw']    # (n, 1338) float64
        y = mordred[ep]['y']
        for rep in range(start, N_REPS):
            t_rep = time.time()
            seed = 101 + rep
            tr_idx, te_idx = splits[ep][rep]
            metrics, yp = train_baseline_lgb(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], seed)
            results[ep].append(metrics)
            save_dict_atomic(results, out_path)
            print(f"  {ep[:30]:<30} rep{rep}: AUC={metrics['AUC']:.4f}  "
                  f"[{time.time()-t_rep:.0f}s]", flush=True)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"Saved: {out_path}", flush=True)


if __name__ == '__main__':
    main()
