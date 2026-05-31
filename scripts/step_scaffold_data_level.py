"""Data-level integration strategies under SCAFFOLD-DISJOINT splits.

Three strategies (mirror step2_strategies.py with random→scaffold splits):

  DL-1. Correlation-based augmentation
        For each (source, target) endpoint pair, identify compounds in source's
        scaffold-split train set that meet the 90% conditional-concordance
        threshold (computed over the overlap of both endpoints' compounds).
        Augment target's scaffold-split train set with these source compounds
        (labeled by source). Train LGB. Evaluate on target's scaffold test.

  DL-2. Global merge with 13-d one-hot task identifier
        Pool all 13 endpoints' scaffold-split train data into a single matrix
        of (2048 + 13) = 2061-d. Train one global LGB. Evaluate per endpoint
        on its scaffold test (with that endpoint's one-hot).

  DL-3. SHAP-guided pairwise merge
        Reuse mech_pairs.pkl from random-split SHAP analysis (the selection
        criteria — ≥3 shared top-10 SHAP bits, directional concordance ≥2/3 —
        are properties of the dataset/representation, not the split).
        For each qualifying pair, merge their scaffold-split train data with
        2-d task ID, train LGB, evaluate on both endpoints' scaffold test.

Outputs:
  results/scaffold_dl_corr_aug.pkl
  results/scaffold_dl_global_merge.pkl
  results/scaffold_dl_shap_pairs.pkl
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings('ignore')
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


def train_baseline_lgb(X_tr, y_tr, X_te, seed,
                        max_evals=50, n_folds=5):
    """Mirrors step1_baseline / step2_strategies TPE: 50 evals + 5-fold CV."""
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
    return m.predict_proba(X_te)[:, 1]


# ============================================================================
# DL-1: Correlation-based augmentation
# ============================================================================
def run_corr_aug(datasets, splits):
    print("\n" + "=" * 70, flush=True)
    print(" DL-1: CORRELATION-BASED AUGMENTATION (scaffold)", flush=True)
    print("=" * 70, flush=True)
    out_file = os.path.join(RESULTS_DIR, 'scaffold_dl_corr_aug.pkl')
    results = load_dict(out_file)

    # Build corr_matrix and augmentation_rules ONCE using FULL datasets
    # (matches step2_strategies.py protocol exactly — concordance is a property
    # of the label data, not of any particular train/test split).
    print("  Computing corr_matrix on FULL datasets (matching random-split protocol)...", flush=True)
    corr_matrix = {}
    for i, ep_a in enumerate(ENDPOINTS):
        smi_a = {s: l for s, l in zip(datasets[ep_a]['smiles'], datasets[ep_a]['y'])}
        for j, ep_b in enumerate(ENDPOINTS):
            if i == j: continue
            smi_b = {s: l for s, l in zip(datasets[ep_b]['smiles'], datasets[ep_b]['y'])}
            overlap = set(smi_a) & set(smi_b)
            if len(overlap) < 10: continue
            results_pair = {}
            for cond_label in (0, 1):
                subset = [(smi_a[s], smi_b[s]) for s in overlap if smi_a[s] == cond_label]
                if len(subset) == 0: continue
                b_labels = [x[1] for x in subset]
                c0, c1 = b_labels.count(0), b_labels.count(1)
                total = len(b_labels)
                results_pair[f'A={cond_label}'] = {
                    'n': total, 'dominant_label': 0 if c0 >= c1 else 1,
                    'dominant_pct': max(c0, c1) / total * 100,
                }
            corr_matrix[(ep_a, ep_b)] = {'overlap': len(overlap),
                                          'conditionals': results_pair}

    augmentation_rules = []
    for (ep_a, ep_b), info in corr_matrix.items():
        for cond_key, ci in info['conditionals'].items():
            if ci['dominant_pct'] >= 90 and ci['n'] >= 10:
                augmentation_rules.append({
                    'source': ep_a, 'target': ep_b,
                    'cond_label_source': int(cond_key.split('=')[1]),
                    'inferred_label_target': ci['dominant_label'],
                })
    target_endpoints = sorted(set(r['target'] for r in augmentation_rules))
    print(f"  Endpoints eligible for augmentation: {len(target_endpoints)}: {target_endpoints}",
          flush=True)

    # Per-rep: apply rules under scaffold split (exclude target's scaffold
    # train/test SMILES, identical to random-split protocol but using scaffold splits)
    for rep in range(N_REPS):
        seed = 101 + rep
        for ep_target in ENDPOINTS:
            results.setdefault(ep_target, [])
            if len(results[ep_target]) > rep: continue
            t_ep = time.time()

            tr_idx, te_idx = splits[ep_target][rep]
            X_tr = datasets[ep_target]['X'][tr_idx].copy()
            y_tr = datasets[ep_target]['y'][tr_idx].copy()
            X_te = datasets[ep_target]['X'][te_idx]
            y_te = datasets[ep_target]['y'][te_idx]
            smi_target_set = set(datasets[ep_target]['smiles'])  # full target SMILES (matches random)
            test_smi_set   = set(datasets[ep_target]['smiles'][te_idx])

            if ep_target not in target_endpoints:
                # No augmentation rule targeting this endpoint
                yp = train_baseline_lgb(X_tr, y_tr, X_te, seed)
                m = evaluate_preds(y_te, yp)
                m['augmented'] = False; m['n_added'] = 0
                results[ep_target].append(m)
                save_dict_atomic(results, out_file)
                print(f"  rep{rep} {ep_target[:25]:<25} (no rule) AUC={m['AUC']:.4f} "
                      f"[{time.time()-t_ep:.0f}s]", flush=True)
                continue

            # Collect augmented compounds — protocol mirrors step2_strategies.py:
            # source compounds whose SMILES is NOT in target_full or test set.
            rules = [r for r in augmentation_rules if r['target'] == ep_target]
            aug_X = []; aug_y = []
            for rule in rules:
                src = rule['source']
                smi_s_all = datasets[src]['smiles']
                y_s_all   = datasets[src]['y']
                X_s_all   = datasets[src]['X']
                for k, smi in enumerate(smi_s_all):
                    if smi in smi_target_set or smi in test_smi_set:
                        continue
                    if y_s_all[k] == rule['cond_label_source']:
                        aug_X.append(X_s_all[k])
                        aug_y.append(rule['inferred_label_target'])
            if aug_X:
                X_aug = np.vstack([X_tr, np.array(aug_X)])
                y_aug = np.concatenate([y_tr, np.array(aug_y, dtype=y_tr.dtype)])
            else:
                X_aug, y_aug = X_tr, y_tr

            yp = train_baseline_lgb(X_aug, y_aug, X_te, seed)
            m = evaluate_preds(y_te, yp)
            m['augmented'] = True; m['n_added'] = len(aug_X)
            results[ep_target].append(m)
            save_dict_atomic(results, out_file)
            print(f"  rep{rep} {ep_target[:25]:<25} (aug +{len(aug_X)}) AUC={m['AUC']:.4f} "
                  f"[{time.time()-t_ep:.0f}s]", flush=True)


# ============================================================================
# DL-2: Global merge
# ============================================================================
def run_global_merge(datasets, splits):
    print("\n" + "=" * 70, flush=True)
    print(" DL-2: GLOBAL DATASET MERGE (scaffold)", flush=True)
    print("=" * 70, flush=True)
    out_file = os.path.join(RESULTS_DIR, 'scaffold_dl_global_merge.pkl')
    results = load_dict(out_file)

    for rep in range(N_REPS):
        seed = 101 + rep
        # Check if this rep is done across all endpoints
        if all(len(results.get(ep, [])) > rep for ep in ENDPOINTS):
            print(f"  Rep {rep+1}/{N_REPS}: already complete", flush=True)
            continue
        t_rep = time.time()
        print(f"\n  --- Rep {rep+1}/{N_REPS} (seed={seed}) ---", flush=True)

        # Build pooled training data: each (ep, compound) row gets 13-d one-hot
        global_X_train = []; global_y_train = []
        test_sets = {}
        for ei, ep in enumerate(ENDPOINTS):
            tr_idx, te_idx = splits[ep][rep]
            X = datasets[ep]['X']; y = datasets[ep]['y']
            tid = np.zeros(len(ENDPOINTS), dtype=np.float32); tid[ei] = 1.0
            global_X_train.append(np.hstack([X[tr_idx], np.tile(tid, (len(tr_idx), 1))]))
            global_y_train.append(y[tr_idx])
            test_sets[ep] = (np.hstack([X[te_idx], np.tile(tid, (len(te_idx), 1))]),
                              y[te_idx])
        gX = np.vstack(global_X_train); gy = np.concatenate(global_y_train)
        print(f"    Training global model on {len(gy)} samples...", flush=True)

        # Hyperopt on a dummy single-row test, then use trained model
        # Trick: pass test_sets via closure. Use full TPE on the global pool
        # with CV inside, then predict per-endpoint test.
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
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        def objective(params):
            p = dict(params)
            scores = []
            for tr, va in skf.split(gX, gy):
                if len(np.unique(gy[va])) < 2: scores.append(0.5); continue
                m = lgb.LGBMClassifier(**p, random_state=seed, verbose=-1, n_jobs=-1)
                m.fit(gX[tr], gy[tr])
                scores.append(roc_auc_score(gy[va], m.predict_proba(gX[va])[:, 1]))
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
        m = lgb.LGBMClassifier(**bp, random_state=seed, verbose=-1, n_jobs=-1)
        m.fit(gX, gy)
        print(f"    Global model trained [{time.time()-t_rep:.0f}s]; predicting per endpoint...", flush=True)
        for ep in ENDPOINTS:
            results.setdefault(ep, [])
            if len(results[ep]) > rep: continue
            X_te, y_te = test_sets[ep]
            yp = m.predict_proba(X_te)[:, 1]
            results[ep].append(evaluate_preds(y_te, yp))
        save_dict_atomic(results, out_file)
        print(f"    Rep {rep+1} done [{(time.time()-t_rep)/60:.1f} min]", flush=True)


# ============================================================================
# DL-3: SHAP-guided pairwise merge
# ============================================================================
def run_shap_pairs(datasets, splits):
    print("\n" + "=" * 70, flush=True)
    print(" DL-3: SHAP-GUIDED PAIRWISE MERGE (scaffold)", flush=True)
    print("=" * 70, flush=True)
    # Reuse pair selection from random-split SHAP analysis
    with open(os.path.join(RESULTS_DIR, 'mech_pairs.pkl'), 'rb') as f:
        mech_pairs = pickle.load(f)
    selected = [p for p in mech_pairs
                if p['shared_count'] >= 3
                and p['directional_concordance'] >= 2.0/3.0]
    print(f"  Using {len(selected)} pre-selected pairs from mech_pairs.pkl", flush=True)
    for p in selected:
        print(f"    {p['ep_a']} <-> {p['ep_b']} "
              f"(shared={p['shared_count']}, dir={p['directional_concordance']:.0%})", flush=True)

    out_file = os.path.join(RESULTS_DIR, 'scaffold_dl_shap_pairs.pkl')
    results = load_dict(out_file)

    # Inline TPE + multi-target predict — train ONCE per (pair, rep).
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
            results.setdefault(exp_a, []); results.setdefault(exp_b, [])
            if len(results[exp_a]) > rep and len(results[exp_b]) > rep:
                continue
            t_pair = time.time()

            # Build merged training data with 2-d task id
            tr_idx_a, _ = splits[ep_a][rep]
            tr_idx_b, _ = splits[ep_b][rep]
            X_a = datasets[ep_a]['X'][tr_idx_a]; y_a = datasets[ep_a]['y'][tr_idx_a]
            X_b = datasets[ep_b]['X'][tr_idx_b]; y_b = datasets[ep_b]['y'][tr_idx_b]
            # Use uint8 task IDs (matches step2_strategies.py exactly)
            tid_a = np.tile([1, 0], (len(X_a), 1)).astype(np.uint8)
            tid_b = np.tile([0, 1], (len(X_b), 1)).astype(np.uint8)
            X_merge = np.vstack([np.hstack([X_a, tid_a]),
                                  np.hstack([X_b, tid_b])])
            y_merge = np.concatenate([y_a, y_b])

            # SINGLE TPE search on merged data (50 evals + 5-fold CV, matching baseline)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            def objective(params):
                p = dict(params)
                scores = []
                for tr, va in skf.split(X_merge, y_merge):
                    if len(np.unique(y_merge[va])) < 2: scores.append(0.5); continue
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

            # Predict on BOTH endpoints' scaffold test sets
            for tgt, exp_key in [(ep_a, exp_a), (ep_b, exp_b)]:
                if len(results[exp_key]) > rep: continue
                _, te_idx = splits[tgt][rep]
                X_te = datasets[tgt]['X'][te_idx]
                y_te = datasets[tgt]['y'][te_idx]
                tid = np.array([1, 0], dtype=np.uint8) if tgt == ep_a else np.array([0, 1], dtype=np.uint8)
                X_te_full = np.hstack([X_te, np.tile(tid, (len(X_te), 1))])
                yp_te = model.predict_proba(X_te_full)[:, 1]
                results[exp_key].append(evaluate_preds(y_te, yp_te))
            save_dict_atomic(results, out_file)
            print(f"  rep{rep} {pair_key[:40]}  [{time.time()-t_pair:.0f}s]", flush=True)


def main():
    t0 = time.time()
    print("Loading datasets and scaffold splits...", flush=True)
    with open(os.path.join(RESULTS_DIR, 'datasets.pkl'), 'rb') as f:
        datasets = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_splits_record.pkl'), 'rb') as f:
        splits = pickle.load(f)

    # Run in order: corr_aug (cheapest), global (expensive), shap (medium)
    run_corr_aug(datasets, splits)
    run_global_merge(datasets, splits)
    run_shap_pairs(datasets, splits)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
