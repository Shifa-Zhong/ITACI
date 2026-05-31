"""Strategy 3 (MAML) — MORDRED extension under scaffold splits.

Direct parallel to step_scaffold_maml.py.

COMPARABILITY GUARANTEES (mirror FP S3):
  - Splits          : scaffold_splits_record.pkl unchanged
  - MAML architecture: 3-layer MLP, weights init * 0.01 (matches FP)
  - Hyperparams     : INNER_STEPS=5, INNER_LR=0.01, OUTER_LR=1e-3,
                       META_EPOCHS=500, N_TASKS_PER_EP=5, SUP_SIZE=64,
                       FT_EPOCHS=100  (IDENTICAL to FP)
  - Ensemble        : FIXED 0.6 LGB + 0.4 MAML (no λ tuning, matches FP)
  - Baseline preds  : loaded from scaffold_baseline_mordred_results.pkl

DIFFS from FP S3:
  - Neural-net X    : mordred[ep]['X_mordred_z']  shape (n, 690)  z-scored
                       (raw Mordred descriptors span 1e-5–1e+5; NN training
                       requires normalization. Adopting z-scored 690-D is the
                       minimal change that lets MAML actually converge.)
  - NN input dim    : in_dim=690 (was 2048)
  - Output          : scaffold_strategy3_mordred.pkl
"""
import os, sys, pickle, time, warnings, copy
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, r'D:\quxintong\scripts')
from _checkpoint import load_dict, save_dict_atomic

RESULTS_DIR = r'D:\quxintong\results'
N_REPS = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[scaffold_maml_mordred] using device: {device}", flush=True)

ENDPOINTS = [
    'prenatal_development_C', 'TSHR_agonist_activity_C',
    'respiratory_toxicity_C', 'ocular_toxicity_C',
    'ames_mutagenicity_C', 'reproductive_toxicity_C',
    'skin_corrosion_C', 'neurotoxicity_C',
    'Estrogen_Receptor_α_C', 'Androgen_Receptor_C',
    'cytotoxicity_C', 'Carcinogenicity_C', 'Hepatotoxicity_C'
]

X_KEY = 'X_mordred_z'   # 690-D z-scored
NN_IN_DIM = 690


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
    def __init__(self, in_dim=NN_IN_DIM, h1=256, h2=128):
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


def main():
    print("=" * 70, flush=True)
    print(f"SCAFFOLD MAML (S3) — MORDRED  13 endpoints × N={N_REPS}", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    with open(os.path.join(RESULTS_DIR, 'mordred_datasets.pkl'), 'rb') as f:
        mordred = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_splits_record.pkl'), 'rb') as f:
        splits = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaffold_baseline_mordred_results.pkl'), 'rb') as f:
        base_preds = pickle.load(f)
    assert all('y_pred' in base_preds[ep][0] for ep in ENDPOINTS), \
        "mordred baseline pkl is missing y_pred"

    out_file = os.path.join(RESULTS_DIR, 'scaffold_strategy3_mordred.pkl')
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

        train_pools = {}
        for ei, ep in enumerate(ENDPOINTS):
            tr_idx, _ = splits[ep][rep]
            X = mordred[ep][X_KEY][tr_idx].astype(np.float32)
            y = mordred[ep]['y'][tr_idx].astype(np.float32)
            train_pools[ei] = (torch.FloatTensor(X).to(device),
                               torch.FloatTensor(y).to(device))

        model = MAMLModel(NN_IN_DIM, 256, 128).to(device)
        outer_opt = optim.Adam(model.parameters(), lr=OUTER_LR)

        for ep_meta in range(META_EPOCHS):
            task_ids = np.random.choice(len(ENDPOINTS), N_TASKS_PER_EP, replace=False)
            outer_opt.zero_grad()
            total_loss = 0.0
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
            if (ep_meta + 1) % 100 == 0:
                tl = total_loss.item() if isinstance(total_loss, torch.Tensor) else 0
                print(f"    meta-ep {ep_meta+1}: outer loss={tl:.4f}", flush=True)

        for ei, ep_target in enumerate(ENDPOINTS):
            tr_idx, te_idx = splits[ep_target][rep]
            X_tr = mordred[ep_target][X_KEY][tr_idx].astype(np.float32)
            X_te = mordred[ep_target][X_KEY][te_idx].astype(np.float32)
            y_tr = mordred[ep_target]['y'][tr_idx]
            y_te = mordred[ep_target]['y'][te_idx]

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
