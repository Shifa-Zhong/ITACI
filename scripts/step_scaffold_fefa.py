"""Strategy 3 REPLACEMENT: Frozen Encoder Feature Augmentation (FEFA).

Why this replaces MAML
---------------------
Under scaffold-disjoint splits, MAML's aggressive fine-tuning (100 epochs at
lr=1e-3 from a meta-learned init) over-adapts to the training-scaffold
subspace and damages out-of-scaffold generalization (mean ΔAUC = -0.014,
3/13 wins, 7 Bonferroni-significantly degraded endpoints).

FEFA eliminates ALL target-side encoder adaptation:
  1. Train MultiTaskNetV3 jointly on all 13 endpoints' scaffold-split TRAIN
     pool (identical protocol to Strategy 2 / scaffold_strategy2.pkl).
  2. FREEZE the encoder.
  3. Extract its 256-d bottleneck output for the target endpoint's train/test
     compounds.
  4. Concatenate the encoder output with the 2,048-bit Morgan fingerprint
     → 2,304-d augmented features.
  5. Train LightGBM (TPE 15 evals + 3-fold CV) on the augmented features.
  6. Ensemble with a baseline LGB via OOF λ-grid {0.2..0.8} (identical to
     Strategy 1 stacking ensemble).

Output: results/scaffold_strategy3_fefa.pkl
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
print(f"[scaffold_fefa] using device: {device}", flush=True)

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
    """Identical to scaffold_strategy2 (MTL) — encoder + 13 task heads."""
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


def train_lgb_quick(X_tr, y_tr, X_te, seed):
    m = lgb.LGBMClassifier(
        n_estimators=300, max_depth=7, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def train_lgb_hyperopt(X_tr, y_tr, X_te, seed, max_evals=15, n_folds=3):
    """TPE search matching Strategy 1's target-stacking LGB."""
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


def pretrain_encoder(datasets, splits, rep, seed):
    """Pre-train MultiTaskNetV3 on all 13 endpoints' scaffold-split TRAIN data.

    Identical to scaffold_strategy2 (MTL) pre-training protocol:
      Adam lr=1e-3, wd=1e-4, cosine T_max=100, batch=256,
      15% val holdout for early stopping (patience=15), max 200 epochs.
    """
    torch.manual_seed(seed); np.random.seed(seed)
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
            print(f"    [rep{rep}] encoder ES at epoch {epoch+1}", flush=True); break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()  # Freeze for embedding extraction
    return model


def extract_embeddings(model, X_fp):
    """Run X_fp through the encoder ONLY (frozen, eval mode)."""
    with torch.no_grad():
        X_t = torch.FloatTensor(X_fp).to(device)
        # Get only the encoder output (256-d)
        # BatchNorm with single sample fails; batch through DataLoader-style chunks
        chunk = 512
        out_chunks = []
        for i in range(0, len(X_t), chunk):
            out_chunks.append(model.encoder(X_t[i:i+chunk]).cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def main():
    print("=" * 70, flush=True)
    print(f"SCAFFOLD FEFA (S3 replacement) — 13 endpoints × N={N_REPS}", flush=True)
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

    out_file = os.path.join(RESULTS_DIR, 'scaffold_strategy3_fefa.pkl')
    fefa = load_dict(out_file)
    done = min((len(fefa.get(ep, [])) for ep in ENDPOINTS), default=0)
    print(f"Resume from rep {done}/{N_REPS}", flush=True)

    for rep in range(done, N_REPS):
        seed = 101 + rep
        torch.manual_seed(seed); np.random.seed(seed)
        t_rep = time.time()
        print(f"\n--- FEFA Rep {rep+1}/{N_REPS} (seed={seed}) ---", flush=True)

        # 1. Pre-train shared MTL encoder on rep's scaffold-split train pool
        encoder_model = pretrain_encoder(datasets, splits, rep, seed)

        # 2. Per endpoint: extract frozen embeddings, augment FP, train LGB
        for ep_target in ENDPOINTS:
            t_ep = time.time()
            tr_idx, te_idx = splits[ep_target][rep]
            X_tr_fp = datasets[ep_target]['X'][tr_idx].astype(np.float32)
            X_te_fp = datasets[ep_target]['X'][te_idx].astype(np.float32)
            y_tr    = datasets[ep_target]['y'][tr_idx]
            y_te    = datasets[ep_target]['y'][te_idx]

            # Frozen encoder embeddings
            emb_tr = extract_embeddings(encoder_model, X_tr_fp)  # 256-d
            emb_te = extract_embeddings(encoder_model, X_te_fp)

            # Augmented features: 2048 + 256 = 2304
            X_tr_aug = np.hstack([X_tr_fp, emb_tr])
            X_te_aug = np.hstack([X_te_fp, emb_te])

            # 3. OOF λ-grid (matches Strategy 1 stacking ensemble)
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            oof_base = np.zeros(len(y_tr))
            oof_fefa = np.zeros(len(y_tr))
            for tr_in, va_in in skf.split(X_tr_fp, y_tr):
                oof_base[va_in] = train_lgb_quick(X_tr_fp[tr_in],  y_tr[tr_in], X_tr_fp[va_in],  seed)
                oof_fefa[va_in] = train_lgb_quick(X_tr_aug[tr_in], y_tr[tr_in], X_tr_aug[va_in], seed)
            best_lam, best_auc = 0.5, 0
            for lam in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                try:
                    a = roc_auc_score(y_tr, lam * oof_base + (1 - lam) * oof_fefa)
                    if a > best_auc: best_auc = a; best_lam = lam
                except Exception:
                    pass

            # 4. Final models: yp_base loaded from full TPE50+5fold baseline;
            #    yp_fefa still trained here with TPE15+3fold on augmented features.
            yp_base = np.asarray(base_preds[ep_target][rep]['y_pred'], dtype=np.float32)
            yp_fefa = train_lgb_hyperopt(X_tr_aug, y_tr, X_te_aug, seed)
            yp_ens  = best_lam * yp_base + (1 - best_lam) * yp_fefa
            metrics = evaluate_preds(y_te, yp_ens)
            metrics['lambda'] = float(best_lam)
            metrics['baseline_AUC'] = float(roc_auc_score(y_te, yp_base))
            fefa.setdefault(ep_target, []).append(metrics)
            save_dict_atomic(fefa, out_file)
            print(f"  {ep_target[:25]:<25} B={metrics['baseline_AUC']:.4f} "
                  f"FEFA(λ={best_lam:.1f})={metrics['AUC']:.4f} "
                  f"d={metrics['AUC']-metrics['baseline_AUC']:+.4f} "
                  f"[{time.time()-t_ep:.0f}s]", flush=True)

        # Free encoder before next rep
        del encoder_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f"  Rep done [{(time.time()-t_rep)/60:.1f} min]", flush=True)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
