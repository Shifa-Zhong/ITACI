"""Step 1: Load data + Train baseline models for all 13 endpoints (N=10 reps)."""
import os, sys, pickle, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

RANDOM_SEEDS = list(range(42, 52))
N_HYPEROPT_EVALS = 50
N_CV_FOLDS = 5
FP_RADIUS = 2
FP_NBITS = 2048
DATA_FILE = r'D:\quxintong\data\toxnew_right.xlsx'
RESULTS_DIR = r'D:\quxintong\results'
os.makedirs(RESULTS_DIR, exist_ok=True)

ENDPOINTS = [
    'prenatal_development_C', 'TSHR_agonist_activity_C',
    'respiratory_toxicity_C', 'ocular_toxicity_C',
    'ames_mutagenicity_C', 'reproductive_toxicity_C',
    'skin_corrosion_C', 'neurotoxicity_C',
    'Estrogen_Receptor_α_C', 'Androgen_Receptor_C',
    'cytotoxicity_C', 'Carcinogenicity_C', 'Hepatotoxicity_C'
]
ENDPOINT_NAMES = [
    'Prenatal Dev. Tox.', 'TSHR Agonist', 'Respiratory Tox.',
    'Ocular Tox.', 'Ames Mutagenicity', 'Reproductive Tox.',
    'Skin Corrosion', 'Neurotoxicity', 'Estrogen Receptor a',
    'Androgen Receptor', 'Cytotoxicity', 'Carcinogenicity', 'Hepatotoxicity'
]

fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return list(fp_gen.GetFingerprintAsNumPy(mol))

def train_evaluate_lgb(X_train, y_train, X_test, y_test, seed=42,
                       max_evals=N_HYPEROPT_EVALS, n_folds=N_CV_FOLDS):
    """LightGBM with full TPE search per Text S2.1 (Table S13)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n_est_c   = [100, 200, 300, 500, 700, 1000]
    depth_c   = [3, 5, 7, 9, 11, -1]
    leaves_c  = [15, 31, 63, 127, 255]
    lr_c      = [0.01, 0.03, 0.05, 0.1, 0.2]
    child_c   = [5, 10, 20, 30, 50]
    sub_c     = [0.6, 0.7, 0.8, 0.9, 1.0]
    col_c     = [0.6, 0.7, 0.8, 0.9, 1.0]
    a_c       = [0, 0.01, 0.1, 1.0]
    l_c       = [0, 0.01, 0.1, 1.0]
    space = {
        'n_estimators':      hp.choice('n_estimators',      n_est_c),
        'max_depth':         hp.choice('max_depth',         depth_c),
        'num_leaves':        hp.choice('num_leaves',        leaves_c),
        'learning_rate':     hp.choice('learning_rate',     lr_c),
        'min_child_samples': hp.choice('min_child_samples', child_c),
        'subsample':         hp.choice('subsample',         sub_c),
        'colsample_bytree':  hp.choice('colsample_bytree',  col_c),
        'reg_alpha':         hp.choice('reg_alpha',         a_c),
        'reg_lambda':        hp.choice('reg_lambda',        l_c),
    }

    def objective(params):
        p = {**params, 'verbose': -1, 'random_state': seed, 'n_jobs': -1}
        p['n_estimators'] = int(p['n_estimators'])
        p['max_depth'] = int(p['max_depth'])
        p['num_leaves'] = int(p['num_leaves'])
        p['min_child_samples'] = int(p['min_child_samples'])
        aucs = []
        for tr, val in skf.split(X_train, y_train):
            m = lgb.LGBMClassifier(**p)
            m.fit(X_train[tr], y_train[tr])
            aucs.append(roc_auc_score(y_train[val], m.predict_proba(X_train[val])[:, 1]))
        return {'loss': -np.mean(aucs), 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=max_evals, trials=trials,
                rstate=np.random.default_rng(seed), verbose=False)

    best_params = {
        'n_estimators':      n_est_c[best['n_estimators']],
        'max_depth':         depth_c[best['max_depth']],
        'num_leaves':        leaves_c[best['num_leaves']],
        'learning_rate':     lr_c[best['learning_rate']],
        'min_child_samples': child_c[best['min_child_samples']],
        'subsample':         sub_c[best['subsample']],
        'colsample_bytree':  col_c[best['colsample_bytree']],
        'reg_alpha':         a_c[best['reg_alpha']],
        'reg_lambda':        l_c[best['reg_lambda']],
        'verbose': -1, 'random_state': seed, 'n_jobs': -1
    }
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    metrics = {
        'AUC': roc_auc_score(y_test, y_prob),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0)
    }
    return model, metrics, best_params

# ---- LOAD DATA ----
datasets_file = os.path.join(RESULTS_DIR, 'datasets.pkl')
if os.path.exists(datasets_file):
    print("Loading cached datasets...")
    with open(datasets_file, 'rb') as f:
        datasets = pickle.load(f)
else:
    print("Computing fingerprints...")
    datasets = {}
    for ep in ENDPOINTS:
        df = pd.read_excel(DATA_FILE, sheet_name=ep)
        fps, valid_idx = [], []
        for i, smi in enumerate(df['new_s']):
            fp = smiles_to_fp(smi)
            if fp is not None:
                fps.append(fp)
                valid_idx.append(i)
        datasets[ep] = {
            'X': np.array(fps),
            'y': df['label'].values[valid_idx],
            'smiles': df['new_s'].values[valid_idx]
        }
    with open(datasets_file, 'wb') as f:
        pickle.dump(datasets, f)

# ---- TABLE 1 ----
print("\n=== TABLE 1: Dataset Summary ===")
print(f"{'Endpoint':<25} {'N':>6} {'Active':>7} {'Inactive':>8} {'Active%':>8}")
total = 0
for i, ep in enumerate(ENDPOINTS):
    n = len(datasets[ep]['y'])
    na = int(datasets[ep]['y'].sum())
    ni = n - na
    total += n
    print(f"{ENDPOINT_NAMES[i]:<25} {n:>6} {na:>7} {ni:>8} {na/n:>7.1%}")
print(f"{'TOTAL':<25} {total:>6}")

# ---- BASELINE (with per-rep checkpointing) ----
from _checkpoint import load_dict, save_dict_atomic
baseline_file = os.path.join(RESULTS_DIR, 'baseline_results.pkl')
splits_file = os.path.join(RESULTS_DIR, 'splits_record.pkl')

baseline_results = load_dict(baseline_file)
splits_record    = load_dict(splits_file)
already = {ep: len(baseline_results.get(ep, [])) for ep in ENDPOINTS}
if any(already.values()):
    print("\n[Resume] existing per-endpoint progress (reps done):")
    for ep in ENDPOINTS:
        if already[ep]:
            print(f"  {ENDPOINT_NAMES[ENDPOINTS.index(ep)]:<25} {already[ep]}/10")

for ei, ep in enumerate(ENDPOINTS):
    print(f"\n[Baseline {ei+1}/13] {ENDPOINT_NAMES[ei]} ({len(datasets[ep]['y'])} compounds)...")
    X, y = datasets[ep]['X'], datasets[ep]['y']
    baseline_results.setdefault(ep, [])
    splits_record.setdefault(ep, [])
    start_rep = len(baseline_results[ep])
    if start_rep >= len(RANDOM_SEEDS):
        print(f"  already complete, skipping")
        continue
    if start_rep > 0:
        print(f"  resuming from rep {start_rep+1}/10")
    for rep in range(start_rep, len(RANDOM_SEEDS)):
        seed = RANDOM_SEEDS[rep]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(sss.split(X, y))
        # ensure splits_record[ep] has this rep slot
        while len(splits_record[ep]) <= rep:
            splits_record[ep].append(None)
        splits_record[ep][rep] = (train_idx, test_idx)
        sys.stdout.write(f"  Rep {rep+1}/10...")
        sys.stdout.flush()
        model, metrics, params = train_evaluate_lgb(
            X[train_idx], y[train_idx], X[test_idx], y[test_idx], seed)
        baseline_results[ep].append(metrics)
        # Save after every rep so a crash loses at most one rep.
        save_dict_atomic(baseline_results, baseline_file)
        save_dict_atomic(splits_record,    splits_file)
        print(f" AUC={metrics['AUC']:.4f}")

# ---- PRINT RESULTS ----
print("\n\n=== TABLE 2: Baseline Performance (mean +/- SD) ===")
for ei, ep in enumerate(ENDPOINTS):
    r = baseline_results[ep]
    vals = {m: [x[m] for x in r] for m in ['AUC','Accuracy','Precision','Recall','F1']}
    line = f"{ENDPOINT_NAMES[ei]:<25}"
    for m in ['AUC','Accuracy','Precision','Recall','F1']:
        line += f" {np.mean(vals[m]):.4f}+/-{np.std(vals[m]):.4f}"
    print(line)

print("\nBaseline complete! Results saved.")
