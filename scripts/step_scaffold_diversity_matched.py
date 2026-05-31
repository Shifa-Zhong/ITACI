"""Diversity + Matched controls on scaffold-split.

Produces TWO output files:
  scaffold_diversity_control.pkl  — random MLP (no early stopping) + LGB
  scaffold_matched_control.pkl    — random MLP (cosine + ES + matched FT) + LGB

Both use OOF λ-grid {0.5..0.9} ensemble selection, identical to random-split
counterparts.
"""
import os, sys, pickle, time, warnings, copy
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb

sys.path.insert(0, r'D:\quxintong\scripts')
from _checkpoint import load_dict, save_dict_atomic

RESULTS_DIR = r'D:\quxintong\results'
N_REPS = 10
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


def make_mlp(in_dim=2048):
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 1),
    ).to(device)


def train_mlp_diversity(X_tr, y_tr, X_te, seed, epochs=100):
    """Original diversity control: 100 epochs, no early stopping."""
    torch.manual_seed(seed)
    model = make_mlp(X_tr.shape[1])
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()
    X_t = torch.FloatTensor(X_tr).to(device)
    y_t = torch.FloatTensor(y_tr.astype(np.float32)).to(device)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 256):
            bb = perm[i:i+256]
            logits = model(X_t[bb]).squeeze(-1)
            loss = crit(logits, y_t[bb])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        yp = torch.sigmoid(model(torch.FloatTensor(X_te).to(device)).squeeze(-1)).cpu().numpy()
    return yp


def train_mlp_matched(X_tr, y_tr, X_te, seed, max_epochs=200, patience=15, ft_epochs=30):
    """Matched control: cosine + early stopping + 30-ep fine-tune at lr=5e-4."""
    torch.manual_seed(seed)
    # 15% val split
    X_in, X_va, y_in, y_va = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=seed,
        stratify=y_tr if len(np.unique(y_tr)) > 1 else None)
    model = make_mlp(X_in.shape[1])
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    crit = nn.BCEWithLogitsLoss()

    X_in_t = torch.FloatTensor(X_in).to(device)
    y_in_t = torch.FloatTensor(y_in.astype(np.float32)).to(device)
    X_va_t = torch.FloatTensor(X_va).to(device)
    y_va_t = torch.FloatTensor(y_va.astype(np.float32)).to(device)

    best_va_auc, best_state, no_imp = -1, None, 0
    model.train()
    for epoch in range(max_epochs):
        perm = torch.randperm(len(X_in_t))
        for i in range(0, len(X_in_t), 256):
            bb = perm[i:i+256]
            logits = model(X_in_t[bb]).squeeze(-1)
            loss = crit(logits, y_in_t[bb])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        # val AUC
        model.eval()
        with torch.no_grad():
            yp_va = torch.sigmoid(model(X_va_t).squeeze(-1)).cpu().numpy()
            try:
                va_auc = roc_auc_score(y_va, yp_va)
            except Exception:
                va_auc = 0.5
        model.train()
        if va_auc > best_va_auc:
            best_va_auc = va_auc; best_state = copy.deepcopy(model.state_dict()); no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience: break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Fine-tune on full train (in+va) for 30 epochs at lr=5e-4
    X_all = np.concatenate([X_in, X_va])
    y_all = np.concatenate([y_in, y_va])
    X_all_t = torch.FloatTensor(X_all).to(device)
    y_all_t = torch.FloatTensor(y_all.astype(np.float32)).to(device)
    ft_opt = optim.Adam(model.parameters(), lr=5e-4)
    model.train()
    for _ in range(ft_epochs):
        perm = torch.randperm(len(X_all_t))
        for i in range(0, len(X_all_t), 256):
            bb = perm[i:i+256]
            logits = model(X_all_t[bb]).squeeze(-1)
            loss = crit(logits, y_all_t[bb])
            ft_opt.zero_grad(); loss.backward(); ft_opt.step()
    model.eval()
    with torch.no_grad():
        yp = torch.sigmoid(model(torch.FloatTensor(X_te).to(device)).squeeze(-1)).cpu().numpy()
    return yp


def train_lgb_quick(X_tr, y_tr, X_te, seed):
    m = lgb.LGBMClassifier(
        n_estimators=300, max_depth=7, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def run_one(X_tr, y_tr, X_te, y_te, seed, mlp_trainer):
    """Train MLP via specified protocol, do OOF λ-grid, combine with LGB."""
    # OOF λ
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    oof_lgb = np.zeros(len(y_tr)); oof_mlp = np.zeros(len(y_tr))
    for tr_in, va_in in skf.split(X_tr, y_tr):
        oof_lgb[va_in] = train_lgb_quick(X_tr[tr_in], y_tr[tr_in], X_tr[va_in], seed)
        oof_mlp[va_in] = mlp_trainer(X_tr[tr_in], y_tr[tr_in], X_tr[va_in], seed)
    best_lam, best_auc = 0.7, 0
    for lam in [0.5, 0.6, 0.7, 0.8, 0.9]:
        try:
            a = roc_auc_score(y_tr, lam * oof_lgb + (1 - lam) * oof_mlp)
            if a > best_auc: best_auc = a; best_lam = lam
        except Exception:
            pass
    # Final
    yp_lgb = train_lgb_quick(X_tr, y_tr, X_te, seed)
    yp_mlp = mlp_trainer(X_tr, y_tr, X_te, seed)
    yp_ens = best_lam * yp_lgb + (1 - best_lam) * yp_mlp
    metrics = evaluate_preds(y_te, yp_ens)
    metrics['lambda'] = float(best_lam)
    metrics['baseline_AUC'] = float(roc_auc_score(y_te, yp_lgb))
    return metrics


def main():
    print("=" * 70, flush=True)
    print(f"SCAFFOLD DIVERSITY + MATCHED CONTROLS  13 endpoints × N={N_REPS}", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    with open(os.path.join(RESULTS_DIR, 'datasets.pkl'), 'rb') as f:
        datasets = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_splits_record.pkl'), 'rb') as f:
        splits = pickle.load(f)

    div_file = os.path.join(RESULTS_DIR, 'scaffold_diversity_control.pkl')
    mat_file = os.path.join(RESULTS_DIR, 'scaffold_matched_control.pkl')
    div = load_dict(div_file); mat = load_dict(mat_file)

    for rep in range(N_REPS):
        seed = 101 + rep
        for ep_target in ENDPOINTS:
            div.setdefault(ep_target, []); mat.setdefault(ep_target, [])
            if len(div[ep_target]) > rep and len(mat[ep_target]) > rep:
                continue
            X = datasets[ep_target]['X']; y = datasets[ep_target]['y']
            tr_idx, te_idx = splits[ep_target][rep]
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            t_ep = time.time()
            if len(div[ep_target]) <= rep:
                m_div = run_one(X_tr, y_tr, X_te, y_te, seed, train_mlp_diversity)
                div[ep_target].append(m_div)
                save_dict_atomic(div, div_file)
            else:
                m_div = div[ep_target][rep]
            if len(mat[ep_target]) <= rep:
                m_mat = run_one(X_tr, y_tr, X_te, y_te, seed, train_mlp_matched)
                mat[ep_target].append(m_mat)
                save_dict_atomic(mat, mat_file)
            else:
                m_mat = mat[ep_target][rep]
            print(f"  rep{rep} {ep_target[:25]:<25} div={m_div['AUC']:.4f} "
                  f"mat={m_mat['AUC']:.4f}  [{time.time()-t_ep:.0f}s]", flush=True)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
