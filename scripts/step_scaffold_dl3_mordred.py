"""DL-3 SHAP-guided pairwise merge — MORDRED extension under scaffold splits.

Direct parallel to the run_shap_pairs() function in step_scaffold_data_level.py.

DESIGN NOTE — pair selection:
  Reuses the FP-SHAP-derived mech_pairs.pkl (7 selected pairs at shared_count>=3
  AND directional_concordance>=2/3). The pair list is endpoint-level (which two
  endpoints look mechanistically related), and reusing it keeps the Mordred DL-3
  paired with FP DL-3 on the SAME pairs. A separate "Mordred-SHAP-selected
  pairs" variant is deferred — that would test SHAP-selection-in-Mordred,
  whereas this script tests "do FP-SHAP-mechanism pairs transfer in Mordred."

COMPARABILITY GUARANTEES — every aspect mirrors FP DL-3:
  - Splits     : LOADS scaffold_splits_record.pkl unchanged
  - HPO        : TPE 50 evals + 5-fold StratifiedKFold (seeded with `seed`)
  - HP space   : IDENTICAL 9 hp.choice ranges
  - Task ID    : 2-d one-hot uint8 [[1,0]/[0,1]]
  - RNG        : np.random.default_rng(seed)
  - Test eval  : SAME scaffold test sets via splits[ep][rep]
  - Pairs      : SAME 7 pairs from mech_pairs.pkl

Only diff:
  - X input : mordred[ep]['X_mordred_raw']  shape (n, 1338)  dense float64
              instead of datasets[ep]['X']  shape (n, 2048)  dense uint8

OUTPUT: results/scaffold_dl_shap_pairs_mordred.pkl
  Schema: dict[f"merge:{epA}+{epB}->test:{ep_target}" -> list of metrics]
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
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

X_KEY = 'X_mordred_raw'   # 1338-D cleaned Mordred descriptors


def evaluate_preds(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'AUC':       float(roc_auc_score(y_true, y_prob)),
        'Accuracy':  float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'F1':        float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main():
    print("=" * 70, flush=True)
    print(" DL-3: SHAP-GUIDED PAIRWISE MERGE — MORDRED (scaffold)", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    print("Loading mordred_datasets.pkl, scaffold_splits_record.pkl, mech_pairs.pkl...", flush=True)
    with open(os.path.join(RESULTS_DIR, 'mordred_datasets.pkl'), 'rb') as f:
        mordred = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_splits_record.pkl'), 'rb') as f:
        splits = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'mech_pairs.pkl'), 'rb') as f:
        mech_pairs = pickle.load(f)

    selected = [p for p in mech_pairs
                if p['shared_count'] >= 3
                and p['directional_concordance'] >= 2.0/3.0]
    print(f"  Using {len(selected)} pre-selected pairs from FP mech_pairs.pkl", flush=True)
    for p in selected:
        print(f"    {p['ep_a']} <-> {p['ep_b']} "
              f"(shared={p['shared_count']}, dir={p['directional_concordance']:.0%}, ovl={p['n_overlap']})", flush=True)

    out_file = os.path.join(RESULTS_DIR, 'scaffold_dl_shap_pairs_mordred.pkl')
    results = load_dict(out_file)

    # IDENTICAL hp.choice space as FP DL-3 (and baseline)
    ne_c = [100, 200, 300, 500, 700, 1000]
    md_c = [3, 5, 7, 9, 11, -1]
    nl_c = [15, 31, 63, 127, 255]
    lr_c = [0.01, 0.03, 0.05, 0.1, 0.2]
    ch_c = [5, 10, 20, 30, 50]
    sub_c = [0.6, 0.7, 0.8, 0.9, 1.0]
    col_c = [0.6, 0.7, 0.8, 0.9, 1.0]
    a_c  = [0, 0.01, 0.1, 1.0]
    l_c  = [0, 0.01, 0.1, 1.0]
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

    for rep in range(N_REPS):
        seed = 101 + rep
        for pair in selected:
            ep_a, ep_b = pair['ep_a'], pair['ep_b']
            pair_key = f"{ep_a}+{ep_b}"
            exp_a = f"merge:{pair_key}->test:{ep_a}"
            exp_b = f"merge:{pair_key}->test:{ep_b}"
            results.setdefault(exp_a, [])
            results.setdefault(exp_b, [])
            if len(results[exp_a]) > rep and len(results[exp_b]) > rep:
                continue
            t_pair = time.time()

            # Build merged training data with 2-d task id (uint8)
            tr_idx_a, _ = splits[ep_a][rep]
            tr_idx_b, _ = splits[ep_b][rep]
            X_a = mordred[ep_a][X_KEY][tr_idx_a].astype(np.float64)
            y_a = mordred[ep_a]['y'][tr_idx_a]
            X_b = mordred[ep_b][X_KEY][tr_idx_b].astype(np.float64)
            y_b = mordred[ep_b]['y'][tr_idx_b]
            tid_a = np.tile([1, 0], (len(X_a), 1)).astype(np.uint8)
            tid_b = np.tile([0, 1], (len(X_b), 1)).astype(np.uint8)
            X_merge = np.vstack([np.hstack([X_a, tid_a]),
                                  np.hstack([X_b, tid_b])])
            y_merge = np.concatenate([y_a, y_b])

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            def objective(params):
                p = dict(params)
                scores = []
                for tr, va in skf.split(X_merge, y_merge):
                    if len(np.unique(y_merge[va])) < 2:
                        scores.append(0.5); continue
                    m = lgb.LGBMClassifier(**p, random_state=seed, verbose=-1, n_jobs=-1)
                    m.fit(X_merge[tr], y_merge[tr])
                    scores.append(roc_auc_score(y_merge[va], m.predict_proba(X_merge[va])[:, 1]))
                return {'loss': -np.mean(scores), 'status': STATUS_OK}
            trials = Trials()
            best = fmin(objective, space, algo=tpe.suggest, max_evals=50,
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
            model = lgb.LGBMClassifier(**bp, random_state=seed, verbose=-1, n_jobs=-1)
            model.fit(X_merge, y_merge)

            for tgt, exp_key in [(ep_a, exp_a), (ep_b, exp_b)]:
                if len(results[exp_key]) > rep: continue
                _, te_idx = splits[tgt][rep]
                X_te = mordred[tgt][X_KEY][te_idx].astype(np.float64)
                y_te = mordred[tgt]['y'][te_idx]
                tid = np.array([1, 0], dtype=np.uint8) if tgt == ep_a else np.array([0, 1], dtype=np.uint8)
                X_te_full = np.hstack([X_te, np.tile(tid, (len(X_te), 1))])
                yp = model.predict_proba(X_te_full)[:, 1]
                results[exp_key].append(evaluate_preds(y_te, yp))
            save_dict_atomic(results, out_file)
            print(f"  rep{rep} {pair_key[:42]}  [{time.time()-t_pair:.0f}s]", flush=True)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"Saved: {out_file}", flush=True)


if __name__ == '__main__':
    main()
