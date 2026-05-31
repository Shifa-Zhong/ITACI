"""ITACI + SMD: Two incompatibility metrics over 78 endpoint pairs.

ITACI — Inter-Task Activity Cliff Index (label-space):
  For each compound c in endpoint A's full dataset, find its k=5 Tanimoto nearest
  neighbours in endpoint B's full dataset; for neighbours above the τ_min similarity
  floor, an "inter-task activity cliff" occurs when y_c (in A) ≠ y_n (in B).
  ITACI(A,B) = E_c [ fraction of qualifying neighbours that disagree in label ].
  Symmetrized: ITACI(A↔B) = mean(ITACI(A→B), ITACI(B→A)).

SMD — SHAP Mechanism Divergence (feature-space):
  Build a signed importance vector  s_E[i] = mean_abs_SHAP_E[i] * sign(mean_SHAP_E[i])
  over 2048 Morgan bits, using the per-endpoint SHAP from the scaffold rootcause pkl.
  SMD(A,B) = 1 - cos(s_A, s_B).  Range ≈ [0, 2]; > 1 means signed-direction disagreement.

Decomposition (reported alongside primary signed-cosine SMD):
  M  = magnitude divergence  = 1 - cos(|s_A|, |s_B|)
  D  = direction divergence  = fraction of bits where sign(s_A[i]) ≠ sign(s_B[i])
        (weighted by min(|s_A[i]|, |s_B[i]|))

Output:
  results/scaffold_itaci_smd.pkl
  Schema:
    endpoints: list[13]
    itaci_matrix: ndarray (13,13), symmetric, NaN on diagonal
    itaci_pairs: list[78] (upper-triangle, ordered by endpoints)
    itaci_n_pairs_per_target: ndarray (13,13) — number of qualifying compounds
    smd_signed_matrix: ndarray (13,13)
    smd_magnitude_matrix: ndarray (13,13)
    smd_direction_matrix: ndarray (13,13)
    smd_pairs: list[78]
    config: dict
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings('ignore')
import numpy as np

sys.path.insert(0, r'D:\quxintong\scripts')

RESULTS_DIR = r'D:\quxintong\results'

ENDPOINTS = [
    'prenatal_development_C', 'TSHR_agonist_activity_C',
    'respiratory_toxicity_C', 'ocular_toxicity_C',
    'ames_mutagenicity_C', 'reproductive_toxicity_C',
    'skin_corrosion_C', 'neurotoxicity_C',
    'Estrogen_Receptor_α_C', 'Androgen_Receptor_C',
    'cytotoxicity_C', 'Carcinogenicity_C', 'Hepatotoxicity_C',
]

# Frozen hyperparameters
K_NN = 5
TAU_MIN = 0.30   # Tanimoto similarity floor (typical activity-cliff threshold)


def tanimoto_matrix(A, B):
    """Tanimoto similarity for binary fingerprints. A: (m,d), B: (n,d) -> (m,n)."""
    A = A.astype(np.float32); B = B.astype(np.float32)
    inter = A @ B.T
    a = A.sum(axis=1, keepdims=True)
    b = B.sum(axis=1, keepdims=True).T
    denom = a + b - inter
    sim = np.where(denom > 0, inter / np.maximum(denom, 1e-9), 0.0)
    return sim


def itaci_directed(X_A, y_A, X_B, y_B, k=K_NN, tau_min=TAU_MIN, batch=512):
    """ITACI(A->B): for each c in A, k-NN in B (sim > tau_min), fraction of label disagreements."""
    n = X_A.shape[0]
    cliff_frac = []
    n_qual_compounds = 0
    for i in range(0, n, batch):
        sim = tanimoto_matrix(X_A[i:i+batch], X_B)  # (b, |B|)
        for j in range(sim.shape[0]):
            row = sim[j]
            order = np.argsort(-row)[:k]
            qual = order[row[order] >= tau_min]
            if len(qual) == 0:
                continue
            n_qual_compounds += 1
            disagree = (y_B[qual] != y_A[i+j]).astype(np.float32).mean()
            cliff_frac.append(disagree)
    if not cliff_frac:
        return float('nan'), 0
    return float(np.mean(cliff_frac)), int(n_qual_compounds)


def smd_signed_cosine(s_A, s_B):
    na = np.linalg.norm(s_A); nb = np.linalg.norm(s_B)
    if na == 0 or nb == 0:
        return float('nan')
    return float(1 - (s_A @ s_B) / (na * nb))


def smd_magnitude(s_A, s_B):
    a = np.abs(s_A); b = np.abs(s_B)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float('nan')
    return float(1 - (a @ b) / (na * nb))


def smd_direction(s_A, s_B):
    """Weighted fraction of bits where signs disagree, weighted by min(|a|,|b|)."""
    a, b = np.abs(s_A), np.abs(s_B)
    w = np.minimum(a, b)
    if w.sum() == 0:
        return float('nan')
    disagree = (np.sign(s_A) != np.sign(s_B)).astype(np.float32)
    # Bits with zero in either side: define as non-disagreement (mask out via weight)
    return float((disagree * w).sum() / w.sum())


def build_signed_importance(shap_global_per_rep, ep, endpoints):
    """Average mean_abs_shap and mean_shap across reps, return signed importance (2048,)."""
    rep_keys = sorted(shap_global_per_rep.keys())
    abs_acc = None; mean_acc = None; nreps = 0
    for rk in rep_keys:
        d = shap_global_per_rep[rk]
        if ep not in d:
            continue
        ent = d[ep]
        # ent: {'mean_abs': ndarray(2048), 'mean_signed': ndarray(2048)} from Stage D
        ma = ent['mean_abs']; ms = ent['mean_signed']
        if abs_acc is None:
            abs_acc = np.zeros_like(ma, dtype=np.float64)
            mean_acc = np.zeros_like(ms, dtype=np.float64)
        abs_acc += ma; mean_acc += ms; nreps += 1
    if nreps == 0:
        return None
    abs_avg = abs_acc / nreps
    mean_avg = mean_acc / nreps
    signed = abs_avg * np.sign(mean_avg)
    return signed.astype(np.float64)


def main():
    print('=' * 70, flush=True)
    print('ITACI + SMD: incompatibility metrics over 78 endpoint pairs', flush=True)
    print(f'K_NN={K_NN}, TAU_MIN={TAU_MIN}', flush=True)
    print('=' * 70, flush=True)
    t0 = time.time()

    print('Loading datasets and SHAP results...', flush=True)
    datasets = pickle.load(open(os.path.join(RESULTS_DIR, 'datasets.pkl'), 'rb'))
    root = pickle.load(open(os.path.join(RESULTS_DIR, 'scaffold_rootcause_results.pkl'), 'rb'))
    shap_per_rep = root['shap_importance_per_rep']

    # === ITACI ===
    print('\n--- ITACI ---', flush=True)
    itaci_mat = np.full((13, 13), np.nan)
    n_qual_mat = np.zeros((13, 13), dtype=int)

    for i, ep_A in enumerate(ENDPOINTS):
        X_A = datasets[ep_A]['X']; y_A = datasets[ep_A]['y']
        for j, ep_B in enumerate(ENDPOINTS):
            if i == j: continue
            X_B = datasets[ep_B]['X']; y_B = datasets[ep_B]['y']
            t1 = time.time()
            val, n_q = itaci_directed(X_A, y_A, X_B, y_B)
            itaci_mat[i, j] = val
            n_qual_mat[i, j] = n_q
            print(f'  ITACI({ep_A[:18]:<18} -> {ep_B[:18]:<18}) = {val:.4f}  (n_qual={n_q})  [{time.time()-t1:.0f}s]', flush=True)

    # Symmetrize
    itaci_sym = np.full((13, 13), np.nan)
    for i in range(13):
        for j in range(i+1, 13):
            itaci_sym[i, j] = itaci_sym[j, i] = 0.5 * (itaci_mat[i, j] + itaci_mat[j, i])

    itaci_pairs = [itaci_sym[i, j] for i in range(13) for j in range(i+1, 13)]
    print(f'\nITACI 78 pairs: mean={np.mean(itaci_pairs):.4f}, '
          f'median={np.median(itaci_pairs):.4f}, '
          f'min={min(itaci_pairs):.4f}, max={max(itaci_pairs):.4f}', flush=True)

    # === SMD ===
    print('\n--- SMD ---', flush=True)
    signed = {}
    for ep in ENDPOINTS:
        s = build_signed_importance(shap_per_rep, ep, ENDPOINTS)
        if s is None:
            print(f'  WARNING: no SHAP for {ep}', flush=True)
        signed[ep] = s

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

    smd_pairs_signed = [smd_s[i, j] for i in range(13) for j in range(i+1, 13)]
    print(f'SMD signed (78 pairs): mean={np.nanmean(smd_pairs_signed):.4f}, '
          f'median={np.nanmedian(smd_pairs_signed):.4f}, '
          f'min={np.nanmin(smd_pairs_signed):.4f}, max={np.nanmax(smd_pairs_signed):.4f}', flush=True)

    # Save
    out = {
        'endpoints': ENDPOINTS,
        'itaci_directed_matrix': itaci_mat,
        'itaci_symmetric_matrix': itaci_sym,
        'itaci_pairs': itaci_pairs,
        'itaci_n_qual_per_pair': n_qual_mat,
        'smd_signed_matrix': smd_s,
        'smd_magnitude_matrix': smd_m,
        'smd_direction_matrix': smd_d,
        'smd_signed_pairs': smd_pairs_signed,
        'config': {'k_nn': K_NN, 'tau_min': TAU_MIN, 'fp': 'Morgan_2048bit_r2',
                   'sim': 'Tanimoto', 'shap_source': 'scaffold_rootcause_results.pkl'},
    }
    out_path = os.path.join(RESULTS_DIR, 'scaffold_itaci_smd.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f'\n✓ Saved: {out_path}', flush=True)
    print(f'Total: {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
