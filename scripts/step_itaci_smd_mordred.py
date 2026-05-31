"""ITACI + SMD on MORDRED descriptors (rep-specific variant).

Parallel to step_itaci_smd.py but with two changes:
  - ITACI similarity: COSINE on z-scored Mordred 690-D (instead of Tanimoto on
    binary Morgan FP). No hard τ_min floor (cosine on z-space lacks a canonical
    activity-cliff threshold); we take the unconditional k=5 nearest neighbours.
  - SMD source: SHAP from fixed-hp Mordred LGBM (loaded from
    scaffold_diagnostics_mordred.pkl). Signed cosine on 1338-D Mordred SHAP
    importance vectors instead of 2048-bit Morgan SHAP.

Outputs: results/scaffold_itaci_smd_mordred.pkl
  Schema mirrors scaffold_itaci_smd.pkl: itaci_matrix, itaci_sym, itaci_pairs (78),
  smd_signed_matrix, smd_magnitude_matrix, smd_direction_matrix, smd_pairs, config.
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

RESULTS_DIR = r'D:\quxintong\results'

ENDPOINTS = [
    'prenatal_development_C', 'TSHR_agonist_activity_C',
    'respiratory_toxicity_C', 'ocular_toxicity_C',
    'ames_mutagenicity_C', 'reproductive_toxicity_C',
    'skin_corrosion_C', 'neurotoxicity_C',
    'Estrogen_Receptor_α_C', 'Androgen_Receptor_C',
    'cytotoxicity_C', 'Carcinogenicity_C', 'Hepatotoxicity_C',
]

K_NN = 5
X_KEY_Z = 'X_mordred_z'  # z-scored 690-D for cosine similarity


def cosine_matrix(A, B):
    """Cosine similarity matrix A: (m, d) × B: (n, d) -> (m, n).
    Assumes A, B are float; normalizes rows by L2."""
    A = A.astype(np.float32); B = B.astype(np.float32)
    A_unit = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    B_unit = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    return A_unit @ B_unit.T


def itaci_directed_cosine(X_A, y_A, X_B, y_B, k=K_NN, batch=512):
    """Mordred-ITACI(A->B): for each c in A, k-NN in B by cosine, fraction of label disagreement.
    No τ_min floor — every compound contributes its k=5 nearest neighbours regardless of
    geometric proximity. This matches the spirit of "always-take-NN" on continuous features."""
    n = X_A.shape[0]
    cliff_frac = []
    for i in range(0, n, batch):
        sim = cosine_matrix(X_A[i:i+batch], X_B)
        for j in range(sim.shape[0]):
            row = sim[j]
            order = np.argsort(-row)[:k]
            disagree = (y_B[order] != y_A[i+j]).astype(np.float32).mean()
            cliff_frac.append(disagree)
    if not cliff_frac:
        return float('nan'), 0
    return float(np.mean(cliff_frac)), int(len(cliff_frac))


def smd_signed_cosine(s_A, s_B):
    na = np.linalg.norm(s_A); nb = np.linalg.norm(s_B)
    if na == 0 or nb == 0: return float('nan')
    return float(1 - (s_A @ s_B) / (na * nb))


def smd_magnitude(s_A, s_B):
    a = np.abs(s_A); b = np.abs(s_B)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return float('nan')
    return float(1 - (a @ b) / (na * nb))


def smd_direction(s_A, s_B):
    a, b = np.abs(s_A), np.abs(s_B)
    w = np.minimum(a, b)
    if w.sum() == 0: return float('nan')
    disagree = (np.sign(s_A) != np.sign(s_B)).astype(np.float32)
    return float((disagree * w).sum() / w.sum())


def build_signed_importance(shap_per_rep, ep):
    """Average mean_abs & mean_signed across reps -> signed importance vector (1338,)."""
    abs_acc = None; mean_acc = None; nreps = 0
    for rk in sorted(shap_per_rep.keys()):
        d = shap_per_rep[rk]
        if ep not in d: continue
        ent = d[ep]
        ma = np.asarray(ent['mean_abs'], dtype=np.float64)
        ms = np.asarray(ent['mean_signed'], dtype=np.float64)
        if abs_acc is None:
            abs_acc = np.zeros_like(ma); mean_acc = np.zeros_like(ms)
        abs_acc += ma; mean_acc += ms; nreps += 1
    if nreps == 0: return None
    return (abs_acc / nreps) * np.sign(mean_acc / nreps)


def main():
    print('=' * 70, flush=True)
    print(f'ITACI + SMD on MORDRED (cosine k-NN, K={K_NN}; SHAP from fixed-hp)', flush=True)
    print('=' * 70, flush=True)
    t0 = time.time()

    print('Loading mordred_datasets.pkl + Mordred diagnostics SHAP...', flush=True)
    mordred = pickle.load(open(os.path.join(RESULTS_DIR, 'mordred_datasets.pkl'), 'rb'))
    diag = pickle.load(open(os.path.join(RESULTS_DIR, 'scaffold_diagnostics_mordred.pkl'), 'rb'))
    shap_per_rep = diag['shap_importance_per_rep']

    # ===== ITACI (cosine k-NN on z-scored 690-D) =====
    print('\n--- ITACI (Mordred cosine k-NN) ---', flush=True)
    itaci_mat = np.full((13, 13), np.nan)
    n_qual_mat = np.zeros((13, 13), dtype=int)
    for i, ep_A in enumerate(ENDPOINTS):
        X_A = mordred[ep_A][X_KEY_Z]; y_A = mordred[ep_A]['y']
        for j, ep_B in enumerate(ENDPOINTS):
            if i == j: continue
            X_B = mordred[ep_B][X_KEY_Z]; y_B = mordred[ep_B]['y']
            t1 = time.time()
            val, n_q = itaci_directed_cosine(X_A, y_A, X_B, y_B)
            itaci_mat[i, j] = val
            n_qual_mat[i, j] = n_q
            print(f'  ITACI({ep_A[:18]:<18} -> {ep_B[:18]:<18}) = {val:.4f}  (n={n_q}) [{time.time()-t1:.0f}s]', flush=True)

    itaci_sym = np.full((13, 13), np.nan)
    for i in range(13):
        for j in range(i+1, 13):
            itaci_sym[i, j] = itaci_sym[j, i] = 0.5 * (itaci_mat[i, j] + itaci_mat[j, i])
    itaci_pairs = [float(itaci_sym[i, j]) for i in range(13) for j in range(i+1, 13)]
    print(f'\nITACI 78 pairs: mean={np.mean(itaci_pairs):.4f}, '
          f'median={np.median(itaci_pairs):.4f}, '
          f'min={min(itaci_pairs):.4f}, max={max(itaci_pairs):.4f}', flush=True)

    # ===== SMD on Mordred SHAP =====
    print('\n--- SMD (Mordred SHAP) ---', flush=True)
    signed = {}
    for ep in ENDPOINTS:
        signed[ep] = build_signed_importance(shap_per_rep, ep)
    smd_s = np.full((13, 13), np.nan)
    smd_m = np.full((13, 13), np.nan)
    smd_d = np.full((13, 13), np.nan)
    for i, ep_A in enumerate(ENDPOINTS):
        for j, ep_B in enumerate(ENDPOINTS):
            if i == j: continue
            sa, sb = signed[ep_A], signed[ep_B]
            if sa is None or sb is None: continue
            smd_s[i, j] = smd_signed_cosine(sa, sb)
            smd_m[i, j] = smd_magnitude(sa, sb)
            smd_d[i, j] = smd_direction(sa, sb)

    smd_pairs_s = [float(smd_s[i, j]) for i in range(13) for j in range(i+1, 13)]
    smd_pairs_m = [float(smd_m[i, j]) for i in range(13) for j in range(i+1, 13)]
    smd_pairs_d = [float(smd_d[i, j]) for i in range(13) for j in range(i+1, 13)]
    print(f'SMD signed 78 pairs: mean={np.nanmean(smd_pairs_s):.4f}, '
          f'median={np.nanmedian(smd_pairs_s):.4f}', flush=True)
    print(f'SMD magnitude 78 pairs: mean={np.nanmean(smd_pairs_m):.4f}', flush=True)
    print(f'SMD direction 78 pairs: mean={np.nanmean(smd_pairs_d):.4f}', flush=True)

    out = dict(
        endpoints=ENDPOINTS,
        itaci_matrix=itaci_mat,
        itaci_sym=itaci_sym,
        itaci_pairs=itaci_pairs,
        itaci_n_pairs_per_target=n_qual_mat,
        smd_signed_matrix=smd_s,
        smd_magnitude_matrix=smd_m,
        smd_direction_matrix=smd_d,
        smd_signed_pairs=smd_pairs_s,
        smd_magnitude_pairs=smd_pairs_m,
        smd_direction_pairs=smd_pairs_d,
        config=dict(
            k_nn=K_NN,
            similarity='cosine on z-scored Mordred 690-D',
            tau_min=None,
            shap_source='fixed-hp LightGBM (n=300, depth=7) — NOT TPE',
            notes=(
                "Mordred-specific ITACI uses cosine k-NN without τ_min floor "
                "(Tanimoto's 0.30 floor is FP-specific). SMD signed cosine on "
                "Mordred SHAP importance (1338-D) from fixed-hp LightGBM."
            ),
        ),
    )
    out_path = os.path.join(RESULTS_DIR, 'scaffold_itaci_smd_mordred.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\n✓ Total: {(time.time()-t0)/60:.1f} min', flush=True)
    print(f'✓ Saved: {out_path}', flush=True)


if __name__ == '__main__':
    main()
