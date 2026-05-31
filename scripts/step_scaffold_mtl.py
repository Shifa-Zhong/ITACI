"""Strategy 2 (Multi-Task Learning) on scaffold-split.

Joint training across 13 endpoints using scaffold-split-derived training data,
then per-endpoint fine-tuning + LightGBM ensemble.  Each rep uses scaffold
splits from scaffold_splits_record.pkl.  Output: scaffold_strategy2.pkl.
"""
import os, sys, pickle, time, copy, warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

sys.path.insert(0, r'D:\quxintong\scripts')
from _checkpoint import load_dict, save_dict_atomic

RESULTS_DIR = r'D:\quxintong\results'
N_REPS = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[scaffold_mtl] using device: {device}", flush=True)

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


class MultiTaskNetV3(nn.Module):
    def __init__(self, in_dim=2048, hid=256, n_eps=13):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, hid),    nn.ReLU(), nn.BatchNorm1d(hid),  nn.Dropout(0.2),
        )
        self.heads = nn.ModuleList([nn.Linear(hid, 1) for _ in range(n_eps)])

    def forward(self, x):
        h = self.encoder(x)
        return h, [head(h).squeeze(-1) for head in self.heads]


def train_lgb_hyperopt(X_tr, y_tr, X_te, seed, max_evals=15, n_folds=3):
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
        'n_estimators': ne_choices[best['n_estimators']],
        'max_depth':    md_choices[best['max_depth']],
        'num_leaves':   nl_choices[best['num_leaves']],
        'learning_rate': lr_choices[best['learning_rate']],
        'min_child_samples': child_choices[best['min_child_samples']],
        'subsample':    sub_choices[best['subsample']],
        'colsample_bytree': col_choices[best['colsample_bytree']],
        'reg_alpha':    a_choices[best['reg_alpha']],
        'reg_lambda':   l_choices[best['reg_lambda']],
    }
    m = lgb.LGBMClassifier(**bp, random_state=seed, verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def train_lgb_quick(X_tr, y_tr, X_te, seed):
    m = lgb.LGBMClassifier(
        n_estimators=300, max_depth=7, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def main():
    print("=" * 70, flush=True)
    print(f"SCAFFOLD MTL  (S2)  13 endpoints × N={N_REPS}", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    print("Loading datasets / scaffold splits / baseline preds...", flush=True)
    with open(os.path.join(RESULTS_DIR, 'datasets.pkl'), 'rb') as f:
        datasets = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_splits_record.pkl'), 'rb') as f:
        splits = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_baseline_results.pkl'), 'rb') as f:
        base_preds = pickle.load(f)
    assert all('y_pred' in base_preds[ep][0] for ep in ENDPOINTS), \
        "baseline pkl is missing y_pred — rerun scaffold_baseline first"

    out_file = os.path.join(RESULTS_DIR, 'scaffold_strategy2.pkl')
    s2 = load_dict(out_file)
    s2_done_reps = min((len(s2.get(ep, [])) for ep in ENDPOINTS), default=0)
    print(f"Resume from rep {s2_done_reps}/{N_REPS}", flush=True)

    for rep in range(s2_done_reps, N_REPS):
        seed = 101 + rep
        torch.manual_seed(seed); np.random.seed(seed)
        t_rep = time.time()
        print(f"\n--- S2 Rep {rep+1}/{N_REPS} (seed={seed}) ---", flush=True)

        # Build unified training set using scaffold-split training indices
        smiles_data = {}
        for ei, ep in enumerate(ENDPOINTS):
            tr_idx, _ = splits[ep][rep]
            smis = datasets[ep]['smiles']
            for idx in tr_idx:
                s = smis[idx]
                if s not in smiles_data:
                    smiles_data[s] = {'fp': datasets[ep]['X'][idx], 'labels': {}}
                smiles_data[s]['labels'][ei] = datasets[ep]['y'][idx]
        all_fps = np.array([d['fp'] for d in smiles_data.values()], dtype=np.float32)
        all_labels = np.array([
            [d['labels'].get(i, float('nan')) for i in range(13)]
            for d in smiles_data.values()
        ], dtype=np.float32)
        print(f"  Unified train: {len(all_fps)} molecules", flush=True)

        model = MultiTaskNetV3(2048, 256, 13).to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        crit = nn.BCEWithLogitsLoss(reduction='none')

        X_t = torch.FloatTensor(all_fps).to(device)
        y_t = torch.FloatTensor(all_labels).to(device)
        mask_t = ~torch.isnan(y_t)
        y_t = torch.nan_to_num(y_t, 0.0)

        n_total = len(all_fps)
        perm = np.random.permutation(n_total)
        n_val = max(1, int(0.15 * n_total))
        val_idx = perm[:n_val]
        train_idx_mtl = perm[n_val:]
        Xtr = X_t[train_idx_mtl]; ytr = y_t[train_idx_mtl]; mtr = mask_t[train_idx_mtl]
        Xva = X_t[val_idx];        yva = y_t[val_idx];        mva = mask_t[val_idx]
        loader = DataLoader(
            TensorDataset(Xtr, ytr, mtr.float()),
            batch_size=256, shuffle=True, drop_last=len(train_idx_mtl) > 256)

        best_vl = float('inf'); best_state = None; patience = 15; no_imp = 0
        for epoch in range(200):
            model.train()
            for Xb, yb, mb in loader:
                _, outs = model(Xb)
                loss = torch.tensor(0.0, device=device); nt = 0
                for i in range(13):
                    mi = mb[:, i].bool()
                    if mi.sum() > 0:
                        loss = loss + crit(outs[i][mi], yb[:, i][mi]).mean(); nt += 1
                if nt > 0: loss /= nt
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                _, vout = model(Xva)
                vl, nv = 0.0, 0
                for i in range(13):
                    mi = mva[:, i]
                    if mi.sum() > 0:
                        vl += crit(vout[i][mi], yva[:, i][mi]).mean().item(); nv += 1
                if nv > 0: vl /= nv
            if vl < best_vl:
                best_vl = vl; best_state = copy.deepcopy(model.state_dict()); no_imp = 0
            else:
                no_imp += 1
            if no_imp >= patience:
                print(f"    Early stop at epoch {epoch+1}", flush=True); break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Per-endpoint fine-tune + LGB ensemble
        for ei, ep_target in enumerate(ENDPOINTS):
            tr_idx, te_idx = splits[ep_target][rep]
            X_tr_fp = datasets[ep_target]['X'][tr_idx]
            X_te_fp = datasets[ep_target]['X'][te_idx]
            y_tr = datasets[ep_target]['y'][tr_idx]
            y_te = datasets[ep_target]['y'][te_idx]

            # Fine-tune (30 epochs, lr=5e-4)
            ft = copy.deepcopy(model); ft.train()
            ft_opt = optim.Adam(ft.parameters(), lr=5e-4, weight_decay=1e-4)
            ft_crit = nn.BCEWithLogitsLoss()
            X_tr_t = torch.FloatTensor(X_tr_fp).to(device)
            y_tr_t = torch.FloatTensor(y_tr).to(device)
            for ep_ft in range(30):
                p_ft = torch.randperm(len(X_tr_t))
                for i in range(0, len(X_tr_t), 256):
                    bb = p_ft[i:i+256]
                    _, oo = ft(X_tr_t[bb])
                    l = ft_crit(oo[ei], y_tr_t[bb])
                    ft_opt.zero_grad(); l.backward(); ft_opt.step()
            ft.eval()
            with torch.no_grad():
                _, te_out = ft(torch.FloatTensor(X_te_fp).to(device))
                yp_mtl = torch.sigmoid(te_out[ei]).cpu().numpy()

            # LGB baseline: load full TPE50+5fold predictions from baseline pkl
            yp_lgb = np.asarray(base_preds[ep_target][rep]['y_pred'], dtype=np.float32)

            # OOF λ selection
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            oof_lgb = np.zeros(len(y_tr)); oof_mtl = np.zeros(len(y_tr))
            for tr_in, va_in in skf.split(X_tr_fp, y_tr):
                oof_lgb[va_in] = train_lgb_quick(X_tr_fp[tr_in], y_tr[tr_in], X_tr_fp[va_in], seed)
                ft_in = copy.deepcopy(model); ft_in.train()
                ft_in_opt = optim.Adam(ft_in.parameters(), lr=5e-4, weight_decay=1e-4)
                Xi = torch.FloatTensor(X_tr_fp[tr_in]).to(device)
                yi = torch.FloatTensor(y_tr[tr_in]).to(device)
                for _ in range(30):
                    p_in = torch.randperm(len(Xi))
                    for j in range(0, len(Xi), 256):
                        bb = p_in[j:j+256]
                        _, oo = ft_in(Xi[bb])
                        l = ft_crit(oo[ei], yi[bb])
                        ft_in_opt.zero_grad(); l.backward(); ft_in_opt.step()
                ft_in.eval()
                with torch.no_grad():
                    _, vo = ft_in(torch.FloatTensor(X_tr_fp[va_in]).to(device))
                    oof_mtl[va_in] = torch.sigmoid(vo[ei]).cpu().numpy()
            best_lam, best_auc = 0.5, 0
            for lam in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                try:
                    a = roc_auc_score(y_tr, lam * oof_lgb + (1 - lam) * oof_mtl)
                    if a > best_auc: best_auc = a; best_lam = lam
                except Exception:
                    pass
            yp_ens = best_lam * yp_lgb + (1 - best_lam) * yp_mtl
            metrics = evaluate_preds(y_te, yp_ens)
            metrics['lambda'] = float(best_lam)
            metrics['baseline_AUC'] = float(roc_auc_score(y_te, yp_lgb))
            s2.setdefault(ep_target, []).append(metrics)
            save_dict_atomic(s2, out_file)
            print(f"  {ep_target[:25]:<25} B={metrics['baseline_AUC']:.4f} "
                  f"Ens(λ={best_lam:.1f})={metrics['AUC']:.4f} "
                  f"d={metrics['AUC']-metrics['baseline_AUC']:+.4f}", flush=True)

        print(f"  Rep done [{(time.time()-t_rep)/60:.1f} min]", flush=True)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
