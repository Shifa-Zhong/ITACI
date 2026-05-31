"""Strategy 3 (MAML) on scaffold-split.

MAML meta-training across 13 endpoints using scaffold-split training data,
then per-endpoint fine-tuning + fixed 0.6/0.4 LightGBM ensemble.
Output: scaffold_strategy3.pkl
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, r'D:\quxintong\scripts')
from _checkpoint import load_dict, save_dict_atomic

RESULTS_DIR = r'D:\quxintong\results'
N_REPS = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[scaffold_maml] using device: {device}", flush=True)

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


class MAMLModel(nn.Module):
    """3-layer MLP, weights managed manually for inner-loop SGD.
    Init scale `* 0.01` matches step9_refined.py (random-split MAML protocol).
    """
    def __init__(self, in_dim=2048, h1=256, h2=128):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(h1, in_dim) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(h1))
        self.w2 = nn.Parameter(torch.randn(h2, h1) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(h2))
        self.w3 = nn.Parameter(torch.randn(1, h2) * 0.01)
        self.b3 = nn.Parameter(torch.zeros(1))

    def fast_forward(self, x, params=None):
        if params is None:
            w1, b1, w2, b2, w3, b3 = self.w1, self.b1, self.w2, self.b2, self.w3, self.b3
        else:
            w1, b1, w2, b2, w3, b3 = params
        h = torch.relu(torch.nn.functional.linear(x, w1, b1))
        h = torch.relu(torch.nn.functional.linear(h, w2, b2))
        return torch.nn.functional.linear(h, w3, b3).squeeze(-1)


def train_lgb_hyperopt(X_tr, y_tr, X_te, seed, max_evals=15, n_folds=3):
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


def main():
    print("=" * 70, flush=True)
    print(f"SCAFFOLD MAML  (S3)  13 endpoints × N={N_REPS}", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    with open(os.path.join(RESULTS_DIR, 'datasets.pkl'), 'rb') as f:
        datasets = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_splits_record.pkl'), 'rb') as f:
        splits = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_baseline_results.pkl'), 'rb') as f:
        base_preds = pickle.load(f)
    assert all('y_pred' in base_preds[ep][0] for ep in ENDPOINTS), \
        "baseline pkl is missing y_pred — rerun scaffold_baseline first"

    out_file = os.path.join(RESULTS_DIR, 'scaffold_strategy3.pkl')
    s3 = load_dict(out_file)
    done = min((len(s3.get(ep, [])) for ep in ENDPOINTS), default=0)
    print(f"Resume from rep {done}/{N_REPS}", flush=True)

    INNER_STEPS = 5
    INNER_LR = 0.01
    OUTER_LR = 1e-3
    META_EPOCHS = 500
    N_TASKS_PER_EP = 5
    SUP_SIZE = 64
    FT_EPOCHS = 100

    for rep in range(done, N_REPS):
        seed = 101 + rep
        torch.manual_seed(seed); np.random.seed(seed)
        t_rep = time.time()
        print(f"\n--- S3 Rep {rep+1}/{N_REPS} (seed={seed}) ---", flush=True)

        # Build per-endpoint training pools using scaffold-split TRAIN indices
        train_pools = {}  # ei -> (X_tensor, y_tensor)
        for ei, ep in enumerate(ENDPOINTS):
            tr_idx, _ = splits[ep][rep]
            X = datasets[ep]['X'][tr_idx].astype(np.float32)
            y = datasets[ep]['y'][tr_idx].astype(np.float32)
            train_pools[ei] = (torch.FloatTensor(X).to(device),
                               torch.FloatTensor(y).to(device))

        model = MAMLModel(2048, 256, 128).to(device)
        outer_opt = optim.Adam(model.parameters(), lr=OUTER_LR)

        for ep_meta in range(META_EPOCHS):
            # Sample N_TASKS_PER_EP endpoints
            task_ids = np.random.choice(len(ENDPOINTS), N_TASKS_PER_EP, replace=False)
            outer_opt.zero_grad()
            total_loss = 0.0
            for tid in task_ids:
                Xp, yp = train_pools[tid]
                if len(Xp) < SUP_SIZE * 2: continue
                perm = torch.randperm(len(Xp))[:SUP_SIZE * 2]
                X_sup, y_sup = Xp[perm[:SUP_SIZE]], yp[perm[:SUP_SIZE]]
                X_qry, y_qry = Xp[perm[SUP_SIZE:]], yp[perm[SUP_SIZE:]]
                # Inner loop: clone params and SGD on support
                params = [model.w1, model.b1, model.w2, model.b2, model.w3, model.b3]
                for _ in range(INNER_STEPS):
                    logits = model.fast_forward(X_sup, params)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_sup)
                    grads = torch.autograd.grad(loss, params, create_graph=True)
                    params = [p - INNER_LR * g for p, g in zip(params, grads)]
                # Query loss with adapted params
                logits_q = model.fast_forward(X_qry, params)
                qloss = torch.nn.functional.binary_cross_entropy_with_logits(logits_q, y_qry)
                total_loss = total_loss + qloss
            if isinstance(total_loss, torch.Tensor):
                total_loss = total_loss / N_TASKS_PER_EP
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                outer_opt.step()
            if (ep_meta + 1) % 100 == 0:
                tl = total_loss.item() if isinstance(total_loss, torch.Tensor) else 0
                print(f"    meta-ep {ep_meta+1}: outer loss={tl:.4f}", flush=True)

        # Per-endpoint fine-tune (100 epochs) and 0.6/0.4 ensemble with LGB baseline
        import copy
        for ei, ep_target in enumerate(ENDPOINTS):
            tr_idx, te_idx = splits[ep_target][rep]
            X_tr = datasets[ep_target]['X'][tr_idx].astype(np.float32)
            X_te = datasets[ep_target]['X'][te_idx].astype(np.float32)
            y_tr = datasets[ep_target]['y'][tr_idx]
            y_te = datasets[ep_target]['y'][te_idx]

            ft_model = copy.deepcopy(model)
            ft_opt = optim.Adam(ft_model.parameters(), lr=1e-3)
            X_tr_t = torch.FloatTensor(X_tr).to(device)
            y_tr_t = torch.FloatTensor(y_tr.astype(np.float32)).to(device)
            ft_model.train()
            for _ in range(FT_EPOCHS):
                perm = torch.randperm(len(X_tr_t))
                for i in range(0, len(X_tr_t), 256):
                    bb = perm[i:i+256]
                    logits = ft_model.fast_forward(X_tr_t[bb])
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tr_t[bb])
                    ft_opt.zero_grad(); loss.backward(); ft_opt.step()
            ft_model.eval()
            with torch.no_grad():
                yp_maml = torch.sigmoid(ft_model.fast_forward(torch.FloatTensor(X_te).to(device))).cpu().numpy()

            # Baseline predictions loaded from full TPE50+5fold baseline pkl.
            # Ensemble weight kept FIXED 0.6/0.4 — matches random-split MAML protocol.
            # (Unlike S1/S2/FEFA which use OOF λ-tuning over {0.1..0.8}.)
            yp_lgb = np.asarray(base_preds[ep_target][rep]['y_pred'], dtype=np.float32)
            yp_ens = 0.6 * yp_lgb + 0.4 * yp_maml
            metrics = evaluate_preds(y_te, yp_ens)
            metrics['baseline_AUC'] = float(roc_auc_score(y_te, yp_lgb))
            s3.setdefault(ep_target, []).append(metrics)
            save_dict_atomic(s3, out_file)
            print(f"  {ep_target[:25]:<25} B={metrics['baseline_AUC']:.4f} "
                  f"Ens(0.6/0.4)={metrics['AUC']:.4f} "
                  f"d={metrics['AUC']-metrics['baseline_AUC']:+.4f}", flush=True)

        print(f"  Rep done [{(time.time()-t_rep)/60:.1f} min]", flush=True)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
