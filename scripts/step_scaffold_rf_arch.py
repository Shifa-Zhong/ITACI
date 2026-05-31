"""RF cross-algorithm + MLP architecture sensitivity on scaffold-split.

Protocol matched 1:1 with random-split counterparts:
  - step_rf_stacking.py        → scaffold RF baseline + stacking
  - step_mlp_arch_sensitivity.py → scaffold MTL & MAML arch sweep

Outputs:
  scaffold_rf_baseline.pkl
  scaffold_rf_stacking.pkl
  scaffold_arch_sensitivity.pkl   (includes BOTH mtl and maml configs)
"""
import os, sys, pickle, time, warnings, copy
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, r'D:\quxintong\scripts')
from _checkpoint import load_dict, save_dict_atomic

RESULTS_DIR = r'D:\quxintong\results'
N_REPS = 10
N_REPS_ARCH = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def train_rf_grid(X_tr, y_tr, X_te, seed):
    """RF with 3-fold CV grid search over max_depth (mirrors random-split RF)."""
    DEPTHS = [5, 10, 20, None]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_d, best_auc = None, -1
    for d in DEPTHS:
        scores = []
        for tr, va in skf.split(X_tr, y_tr):
            if len(np.unique(y_tr[va])) < 2: scores.append(0.5); continue
            m = RandomForestClassifier(n_estimators=500, max_depth=d,
                                        random_state=seed, n_jobs=-1)
            m.fit(X_tr[tr], y_tr[tr])
            scores.append(roc_auc_score(y_tr[va], m.predict_proba(X_tr[va])[:, 1]))
        if np.mean(scores) > best_auc:
            best_auc = np.mean(scores); best_d = d
    m = RandomForestClassifier(n_estimators=500, max_depth=best_d,
                                random_state=seed, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def rf_scaffold():
    print("\n--- SCAFFOLD RF baseline + stacking ---", flush=True)
    with open(os.path.join(RESULTS_DIR, 'datasets.pkl'), 'rb') as f:
        datasets = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_splits_record.pkl'), 'rb') as f:
        splits = pickle.load(f)

    bf = os.path.join(RESULTS_DIR, 'scaffold_rf_baseline.pkl')
    sf = os.path.join(RESULTS_DIR, 'scaffold_rf_stacking.pkl')
    rb = load_dict(bf); rs = load_dict(sf)

    # Sources: per-rep scaffold-split train (mirrors random-split RF protocol).
    # Each source RF is grid-searched on max_depth (mirrors random protocol).
    src_cache = {}  # (rep, ep_src) -> trained model
    def get_src(ep_src, rep, seed):
        key = (rep, ep_src)
        if key in src_cache: return src_cache[key]
        tr_idx, _ = splits[ep_src][rep]
        X = datasets[ep_src]['X'][tr_idx]
        y = datasets[ep_src]['y'][tr_idx]
        # Grid search depth on this source's training fold
        DEPTHS = [5, 10, 20, None]
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        best_d, best_auc = None, -1
        for d in DEPTHS:
            scores = []
            for tr, va in skf.split(X, y):
                if len(np.unique(y[va])) < 2: scores.append(0.5); continue
                m = RandomForestClassifier(n_estimators=500, max_depth=d,
                                            random_state=seed, n_jobs=-1)
                m.fit(X[tr], y[tr])
                scores.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
            if np.mean(scores) > best_auc:
                best_auc = np.mean(scores); best_d = d
        m = RandomForestClassifier(n_estimators=500, max_depth=best_d,
                                    random_state=seed, n_jobs=-1)
        m.fit(X, y)
        src_cache[key] = m
        return m

    for rep in range(N_REPS):
        seed = 101 + rep
        for ep_target in ENDPOINTS:
            rb.setdefault(ep_target, []); rs.setdefault(ep_target, [])
            if len(rb[ep_target]) > rep and len(rs[ep_target]) > rep:
                continue
            X = datasets[ep_target]['X']; y = datasets[ep_target]['y']
            tr_idx, te_idx = splits[ep_target][rep]
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            t_ep = time.time()
            if len(rb[ep_target]) <= rep:
                yp_b = train_rf_grid(X_tr, y_tr, X_te, seed)
                rb[ep_target].append(evaluate_preds(y_te, yp_b))
                save_dict_atomic(rb, bf)
            if len(rs[ep_target]) <= rep:
                src_preds_tr, src_preds_te = [], []
                for ep_src in ENDPOINTS:
                    if ep_src == ep_target: continue
                    m = get_src(ep_src, rep, seed)
                    src_preds_tr.append(m.predict_proba(X_tr)[:, 1])
                    src_preds_te.append(m.predict_proba(X_te)[:, 1])
                X_tr_aug = np.hstack([X_tr, np.column_stack(src_preds_tr)])
                X_te_aug = np.hstack([X_te, np.column_stack(src_preds_te)])
                yp_s = train_rf_grid(X_tr_aug, y_tr, X_te_aug, seed)
                rs[ep_target].append(evaluate_preds(y_te, yp_s))
                save_dict_atomic(rs, sf)
            print(f"  rep{rep} {ep_target[:25]:<25} RF [{time.time()-t_ep:.0f}s]", flush=True)
        src_cache.clear()


# ============================================================================
# MLP arch sensitivity — protocol mirrors step_mlp_arch_sensitivity.py
# ============================================================================
class MTLArchNet(nn.Module):
    """Mirrors random-split arch sensitivity layout: Linear → BN → ReLU → Dropout."""
    def __init__(self, in_dim, h1, h2, n_eps=13):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h1, h2),     nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(0.2),
        )
        self.heads = nn.ModuleList([nn.Linear(h2, 1) for _ in range(n_eps)])
    def forward(self, x):
        h = self.encoder(x)
        return [head(h).squeeze(-1) for head in self.heads]


class MAMLArchNet(nn.Module):
    """Plain MLP, weight init `* 0.01` (matches step9 / random arch sensitivity)."""
    def __init__(self, in_dim, h1, h2):
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


def arch_scaffold():
    print("\n--- SCAFFOLD ARCH SENSITIVITY (MTL + MAML) ---", flush=True)
    with open(os.path.join(RESULTS_DIR, 'datasets.pkl'), 'rb') as f:
        datasets = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_splits_record.pkl'), 'rb') as f:
        splits = pickle.load(f)

    out_file = os.path.join(RESULTS_DIR, 'scaffold_arch_sensitivity.pkl')
    arch = load_dict(out_file)
    arch.setdefault('mtl', {}); arch.setdefault('maml', {})

    MTL_CONFIGS = {'small': (256, 128), 'default': (512, 256), 'large': (1024, 512)}
    MAML_CONFIGS = {'small': (128, 64), 'default': (256, 128), 'large': (512, 256)}

    # ===== MTL sweep =====
    for sz_name, (h1, h2) in MTL_CONFIGS.items():
        arch['mtl'].setdefault(sz_name, {'arch': (h1, h2), 'results': {}})
        for ep_target in ENDPOINTS:
            if ep_target in arch['mtl'][sz_name]['results']: continue
            aucs = []
            for rep in range(N_REPS_ARCH):
                seed = 101 + rep
                torch.manual_seed(seed); np.random.seed(seed)
                # Build pooled training data with 15% val holdout (mirrors random)
                smi_data = {}
                for ei, ep_pool in enumerate(ENDPOINTS):
                    tr_idx, _ = splits[ep_pool][rep]
                    smis = datasets[ep_pool]['smiles']
                    for idx in tr_idx:
                        s = smis[idx]
                        if s not in smi_data:
                            smi_data[s] = {'fp': datasets[ep_pool]['X'][idx], 'labels': {}}
                        smi_data[s]['labels'][ei] = datasets[ep_pool]['y'][idx]
                all_fps = np.array([d['fp'] for d in smi_data.values()], dtype=np.float32)
                all_labels = np.array([
                    [d['labels'].get(i, float('nan')) for i in range(13)]
                    for d in smi_data.values()], dtype=np.float32)
                # 15% val holdout (random-split arch script does this but doesn't use ES)
                n_total = len(all_fps)
                perm = np.random.permutation(n_total)
                n_val = max(1, int(0.15 * n_total))
                train_pool_idx = perm[n_val:]
                X_pool = all_fps[train_pool_idx]; y_pool = all_labels[train_pool_idx]

                model = MTLArchNet(2048, h1, h2).to(device)
                opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
                crit = nn.BCEWithLogitsLoss(reduction='none')
                X_t = torch.FloatTensor(X_pool).to(device)
                y_t = torch.FloatTensor(y_pool).to(device)
                mask_t = ~torch.isnan(y_t); y_t = torch.nan_to_num(y_t, 0.0)
                ds = TensorDataset(X_t, y_t, mask_t.float())
                ld = DataLoader(ds, batch_size=256, shuffle=True, drop_last=len(X_t) > 256)
                model.train()
                for _ in range(60):  # 60 epochs, no early stop (mirrors random)
                    for Xb, yb, mb in ld:
                        outs = model(Xb)
                        loss = torch.tensor(0.0, device=device); nt = 0
                        for i in range(13):
                            mi = mb[:, i].bool()
                            if mi.sum() > 0:
                                loss = loss + crit(outs[i][mi], yb[:, i][mi]).mean(); nt += 1
                        if nt > 0: loss /= nt
                        opt.zero_grad(); loss.backward(); opt.step()
                    sched.step()
                # Fine-tune 30 epochs on target endpoint
                ei = ENDPOINTS.index(ep_target)
                tr_idx, te_idx = splits[ep_target][rep]
                X_tr = torch.FloatTensor(datasets[ep_target]['X'][tr_idx]).to(device)
                y_tr = torch.FloatTensor(datasets[ep_target]['y'][tr_idx].astype(np.float32)).to(device)
                X_te = torch.FloatTensor(datasets[ep_target]['X'][te_idx]).to(device)
                y_te = datasets[ep_target]['y'][te_idx]
                ft_opt = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
                ft_crit = nn.BCEWithLogitsLoss()
                model.train()
                for _ in range(30):
                    perm = torch.randperm(len(X_tr))
                    for i in range(0, len(X_tr), 256):
                        bb = perm[i:i+256]
                        out = model(X_tr[bb])[ei]
                        loss = ft_crit(out, y_tr[bb])
                        ft_opt.zero_grad(); loss.backward(); ft_opt.step()
                model.eval()
                with torch.no_grad():
                    yp = torch.sigmoid(model(X_te)[ei]).cpu().numpy()
                try: aucs.append(float(roc_auc_score(y_te, yp)))
                except: aucs.append(0.5)
            arch['mtl'][sz_name]['results'][ep_target] = aucs
            save_dict_atomic(arch, out_file)
            print(f"  mtl/{sz_name} {ep_target[:25]:<25} aucs={[f'{a:.3f}' for a in aucs]}",
                  flush=True)

    # ===== MAML sweep =====
    INNER_STEPS = 5; INNER_LR = 0.01; OUTER_LR = 1e-3
    META_EPOCHS = 200; N_TASKS_PER_EP = 5
    SUP_SIZE = 64; FT_EPOCHS = 30

    for sz_name, (h1, h2) in MAML_CONFIGS.items():
        arch['maml'].setdefault(sz_name, {'arch': (h1, h2), 'results': {}})
        for ep_target in ENDPOINTS:
            if ep_target in arch['maml'][sz_name]['results']: continue
            aucs = []
            for rep in range(N_REPS_ARCH):
                seed = 101 + rep
                torch.manual_seed(seed); np.random.seed(seed)
                # Build per-endpoint training pools (scaffold-split train data)
                train_pools = {}
                for ei, ep_pool in enumerate(ENDPOINTS):
                    tr_idx, _ = splits[ep_pool][rep]
                    X = datasets[ep_pool]['X'][tr_idx].astype(np.float32)
                    y = datasets[ep_pool]['y'][tr_idx].astype(np.float32)
                    train_pools[ei] = (torch.FloatTensor(X).to(device),
                                       torch.FloatTensor(y).to(device))

                model = MAMLArchNet(2048, h1, h2).to(device)
                outer_opt = optim.Adam(model.parameters(), lr=OUTER_LR)
                for ep_meta in range(META_EPOCHS):
                    task_ids = np.random.choice(len(ENDPOINTS), N_TASKS_PER_EP, replace=False)
                    outer_opt.zero_grad(); total_loss = 0.0
                    for tid in task_ids:
                        Xp, yp = train_pools[tid]
                        if len(Xp) < SUP_SIZE * 2: continue
                        perm = torch.randperm(len(Xp))[:SUP_SIZE * 2]
                        X_sup, y_sup = Xp[perm[:SUP_SIZE]], yp[perm[:SUP_SIZE]]
                        X_qry, y_qry = Xp[perm[SUP_SIZE:]], yp[perm[SUP_SIZE:]]
                        params = [model.w1, model.b1, model.w2, model.b2, model.w3, model.b3]
                        for _ in range(INNER_STEPS):
                            logits = model.fast_forward(X_sup, params)
                            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_sup)
                            grads = torch.autograd.grad(loss, params, create_graph=True)
                            params = [p - INNER_LR * g for p, g in zip(params, grads)]
                        logits_q = model.fast_forward(X_qry, params)
                        qloss = torch.nn.functional.binary_cross_entropy_with_logits(logits_q, y_qry)
                        total_loss = total_loss + qloss
                    if isinstance(total_loss, torch.Tensor):
                        total_loss = total_loss / N_TASKS_PER_EP
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        outer_opt.step()
                # Fine-tune 30 epochs
                tr_idx, te_idx = splits[ep_target][rep]
                X_tr = torch.FloatTensor(datasets[ep_target]['X'][tr_idx].astype(np.float32)).to(device)
                y_tr = torch.FloatTensor(datasets[ep_target]['y'][tr_idx].astype(np.float32)).to(device)
                X_te = torch.FloatTensor(datasets[ep_target]['X'][te_idx].astype(np.float32)).to(device)
                y_te = datasets[ep_target]['y'][te_idx]
                ft_opt = optim.Adam(model.parameters(), lr=1e-3)
                for _ in range(FT_EPOCHS):
                    perm = torch.randperm(len(X_tr))
                    for i in range(0, len(X_tr), 256):
                        bb = perm[i:i+256]
                        logits = model.fast_forward(X_tr[bb])
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tr[bb])
                        ft_opt.zero_grad(); loss.backward(); ft_opt.step()
                model.eval()
                with torch.no_grad():
                    yp = torch.sigmoid(model.fast_forward(X_te)).cpu().numpy()
                try: aucs.append(float(roc_auc_score(y_te, yp)))
                except: aucs.append(0.5)
            arch['maml'][sz_name]['results'][ep_target] = aucs
            save_dict_atomic(arch, out_file)
            print(f"  maml/{sz_name} {ep_target[:25]:<25} aucs={[f'{a:.3f}' for a in aucs]}",
                  flush=True)


def main():
    t0 = time.time()
    rf_scaffold()
    arch_scaffold()
    print(f"\nTotal RF+Arch: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
