"""Mordred-feature data-level integration under SCAFFOLD-DISJOINT splits.

Direct parallel to step_scaffold_data_level.py for the Mordred-descriptor
extension. Covers DL-1 (correlation-based augmentation) + DL-2 (global merge
with 13-d one-hot task identifier). DL-3 (SHAP-guided pairwise merge) requires
Mordred-derived mech_pairs and is handled separately in step_scaffold_dl3_mordred.py.

CRITICAL COMPARABILITY GUARANTEES (mirror FP exactly):
  - Splits      : reuse scaffold_splits_record.pkl
  - Per-rep seeds: 101..110
  - HPO         : TPE 50 evals, 5-fold inner CV — IDENTICAL hp.choice ranges
  - Correlation/augmentation rules (DL-1): computed on FULL per-endpoint labels —
                    matches FP protocol exactly since concordance is a label
                    property, independent of the feature representation
  - Global merge (DL-2): same 13-d one-hot task identifier; only the feature
                    width differs (1338 Mordred + 13 = 1351-d vs 2048 FP + 13 = 2061-d)

OUTPUTS:
  results/scaffold_dl_corr_aug_mordred.pkl
  results/scaffold_dl_global_merge_mordred.pkl
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')  # handle α in Estrogen_Receptor_α_C
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

# Mordred uses 'X_mordred_raw' (1338-D, cleaned + median-imputed, no scaling).
# Trees don't need scaling; FP used binary X directly with the same LGBM HP space.
X_KEY = 'X_mordred_raw'


def evaluate_preds(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'AUC':       float(roc_auc_score(y_true, y_prob)),
        'Accuracy':  float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'F1':        float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_baseline_lgb(X_tr, y_tr, X_te, seed,
                        max_evals=50, n_folds=5):
    """IDENTICAL HP space + TPE config as step_scaffold_data_level.py."""
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
# Rules are computed on labels only -> identical to FP. Only X differs.
# ============================================================================
def run_corr_aug(mordred, datasets_for_smiles, splits):
    print("\n" + "=" * 70, flush=True)
    print(" DL-1: CORRELATION-BASED AUGMENTATION (Mordred / scaffold)", flush=True)
    print("=" * 70, flush=True)
    out_file = os.path.join(RESULTS_DIR, 'scaffold_dl_corr_aug_mordred.pkl')
    results = load_dict(out_file)

    # Build corr_matrix on FULL datasets - this is label-based ONLY, so it should
    # produce the SAME augmentation rules as FP. We compute fresh here for clarity.
    print("  Computing corr_matrix on FULL datasets (labels only)...", flush=True)
    corr_matrix = {}
    for i, ep_a in enumerate(ENDPOINTS):
        smi_a = {s: l for s, l in zip(datasets_for_smiles[ep_a]['smiles'],
                                       datasets_for_smiles[ep_a]['y'])}
        for j, ep_b in enumerate(ENDPOINTS):
            if i == j: continue
            smi_b = {s: l for s, l in zip(datasets_for_smiles[ep_b]['smiles'],
                                           datasets_for_smiles[ep_b]['y'])}
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
    print(f"  Total augmentation rules: {len(augmentation_rules)}", flush=True)

    for rep in range(N_REPS):
        seed = 101 + rep
        for ep_target in ENDPOINTS:
            results.setdefault(ep_target, [])
            if len(results[ep_target]) > rep: continue
            t_ep = time.time()

            tr_idx, te_idx = splits[ep_target][rep]
            X_tr = mordred[ep_target][X_KEY][tr_idx].copy()
            y_tr = mordred[ep_target]['y'][tr_idx].copy()
            X_te = mordred[ep_target][X_KEY][te_idx]
            y_te = mordred[ep_target]['y'][te_idx]
            smi_target_set = set(mordred[ep_target]['smiles'])
            test_smi_set   = set(np.asarray(mordred[ep_target]['smiles'])[te_idx])

            if ep_target not in target_endpoints:
                yp = train_baseline_lgb(X_tr, y_tr, X_te, seed)
                m = evaluate_preds(y_te, yp)
                m['augmented'] = False; m['n_added'] = 0
                results[ep_target].append(m)
                save_dict_atomic(results, out_file)
                print(f"  rep{rep} {ep_target[:25]:<25} (no rule) AUC={m['AUC']:.4f} "
                      f"[{time.time()-t_ep:.0f}s]", flush=True)
                continue

            # Collect augmented compounds from sources (Mordred features)
            rules = [r for r in augmentation_rules if r['target'] == ep_target]
            aug_X = []; aug_y = []
            for rule in rules:
                src = rule['source']
                smi_s_all = mordred[src]['smiles']
                y_s_all   = mordred[src]['y']
                X_s_all   = mordred[src][X_KEY]
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
# DL-2: Global merge with 13-d one-hot task identifier
# Mordred parallel: feature width 1338 + 13 = 1351-d instead of 2048+13=2061-d
# ============================================================================
def run_global_merge(mordred, splits):
    print("\n" + "=" * 70, flush=True)
    print(" DL-2: GLOBAL DATASET MERGE (Mordred / scaffold)", flush=True)
    print("=" * 70, flush=True)
    out_file = os.path.join(RESULTS_DIR, 'scaffold_dl_global_merge_mordred.pkl')
    results = load_dict(out_file)

    for rep in range(N_REPS):
        seed = 101 + rep
        if all(len(results.get(ep, [])) > rep for ep in ENDPOINTS):
            print(f"  Rep {rep+1}/{N_REPS}: already complete", flush=True)
            continue
        t_rep = time.time()
        print(f"\n  --- Rep {rep+1}/{N_REPS} (seed={seed}) ---", flush=True)

        global_X_train = []; global_y_train = []
        test_sets = {}
        for ei, ep in enumerate(ENDPOINTS):
            tr_idx, te_idx = splits[ep][rep]
            X = mordred[ep][X_KEY]; y = mordred[ep]['y']
            tid = np.zeros(len(ENDPOINTS), dtype=np.float32); tid[ei] = 1.0
            global_X_train.append(np.hstack([X[tr_idx], np.tile(tid, (len(tr_idx), 1))]))
            global_y_train.append(y[tr_idx])
            test_sets[ep] = (np.hstack([X[te_idx], np.tile(tid, (len(te_idx), 1))]),
                              y[te_idx])
        gX = np.vstack(global_X_train); gy = np.concatenate(global_y_train)
        print(f"    Global pool: {gX.shape}, training...", flush=True)

        # TPE on the global pool (50 evals, 5-fold CV) - IDENTICAL to FP DL-2
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
        print(f"    Global model trained [{time.time()-t_rep:.0f}s]; predicting per endpoint...",
              flush=True)
        for ep in ENDPOINTS:
            results.setdefault(ep, [])
            if len(results[ep]) > rep: continue
            X_te, y_te = test_sets[ep]
            yp = m.predict_proba(X_te)[:, 1]
            results[ep].append(evaluate_preds(y_te, yp))
        save_dict_atomic(results, out_file)
        print(f"    Rep {rep+1} done [{(time.time()-t_rep)/60:.1f} min]", flush=True)


def main():
    t0 = time.time()
    print("Loading mordred_datasets.pkl, datasets.pkl (for SMILES/labels), and scaffold splits...",
          flush=True)
    with open(os.path.join(RESULTS_DIR, 'mordred_datasets.pkl'), 'rb') as f:
        mordred = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'datasets.pkl'), 'rb') as f:
        # Used for SMILES+labels in corr-aug rule construction (identical to FP)
        # mordred_datasets.pkl already has these but datasets.pkl is the canonical source
        datasets_for_smiles = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_splits_record.pkl'), 'rb') as f:
        splits = pickle.load(f)

    # Sanity check
    for ep in ENDPOINTS:
        Xm = mordred[ep][X_KEY]
        ym = mordred[ep]['y']
        assert Xm.shape[0] == len(ym), f'{ep}: row count mismatch'
        for tr, te in splits[ep]:
            assert int(tr.max()) < Xm.shape[0] and int(te.max()) < Xm.shape[0], \
                f'{ep}: split index OOB for Mordred matrix'
    print(f"  comparability check passed", flush=True)

    # Run DL-1 first (cheaper per-rep, more reps), then DL-2 (one global model per rep)
    run_corr_aug(mordred, datasets_for_smiles, splits)
    run_global_merge(mordred, splits)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
