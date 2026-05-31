"""Stage 1: Compute Mordred 2D descriptors for all unique SMILES across the 13
toxicity endpoints, then apply Phase A (NaN/Inf cleaning) + Phase B (variance +
correlation pruning).  Save raw + filtered + z-scored matrices.

INPUT  : D:/quxintong/results/datasets.pkl
         dict[endpoint] -> {'X': MorganFP (n, 2048), 'y': (n,), 'smiles': list[n]}

OUTPUT : D:/quxintong/results/mordred_raw.pkl
           dict with keys:
             smiles_list      : sorted list of N unique SMILES
             desc_names_raw   : list[1613] - column names from Mordred
             mat_raw          : (N, 1613) float64 - raw computed values, NaN for failures
             rdkit_failed     : list of SMILES that RDKit could not parse
           This file is the EXPENSIVE artifact - re-filtering only re-runs Stage 2.

         D:/quxintong/results/mordred_filtered.pkl
           dict with keys:
             smiles_list, desc_names_kept, mat_raw (post-A), mat_zscore (post-B),
             desc_names_zscore, dropped_log (per-phase report).

         D:/quxintong/results/mordred_datasets.pkl
           dict[endpoint] -> {'X_mordred_raw': (n, n_feat_A), 'X_mordred_z': (n, n_feat_B),
                              'y': labels, 'smiles': same as in datasets.pkl}
           Direct drop-in parallel to datasets.pkl for downstream pipeline reuse.
"""
import sys, os, pickle, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from mordred import Calculator, descriptors

DATASETS_IN = r'D:\quxintong\results\datasets.pkl'
OUT_RAW     = r'D:\quxintong\results\mordred_raw.pkl'
OUT_FILT    = r'D:\quxintong\results\mordred_filtered.pkl'
OUT_DSET    = r'D:\quxintong\results\mordred_datasets.pkl'

NPROC          = 16          # 24 cores available; leave headroom
NAN_FRAC_MAX   = 0.05        # Phase A: drop columns with > 5% NaN
VAR_MIN        = 1e-6        # Phase B: drop columns with var below this (post-zscore)
CORR_THRESH    = 0.95        # Phase B: drop one of each pair with |r| > 0.95


# ============================================================
# STAGE 1: compute raw Mordred (skip if checkpoint exists)
# ============================================================

def compute_raw():
    if os.path.exists(OUT_RAW):
        print(f'[STAGE 1] Checkpoint exists: {OUT_RAW}')
        with open(OUT_RAW, 'rb') as f:
            return pickle.load(f)

    print(f'[STAGE 1] Computing Mordred from scratch')
    with open(DATASETS_IN, 'rb') as f:
        ds = pickle.load(f)
    smiles_set = set()
    for v in ds.values():
        smiles_set.update(v['smiles'])
    smiles_list = sorted(smiles_set)
    print(f'  unique SMILES: {len(smiles_list)}')

    print(f'  parsing RDKit mols...')
    mols, failed = [], []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            failed.append(s)
            mols.append(None)
        else:
            mols.append(m)
    n_ok = sum(1 for m in mols if m is not None)
    print(f'    parsed OK: {n_ok}/{len(smiles_list)}  (failed: {len(failed)})')

    print(f'  running Mordred (nproc={NPROC})...')
    calc = Calculator(descriptors, ignore_3D=True)
    t0 = time.time()
    # calc.pandas accepts list with None - it returns NaN row for None mols
    valid_mols = [m for m in mols if m is not None]
    df = calc.pandas(valid_mols, nproc=NPROC, quiet=False)
    dt = time.time() - t0
    print(f'    done: {len(valid_mols)} mols in {dt:.1f}s  ({len(valid_mols)/dt:.1f} mol/sec)')
    print(f'    df shape: {df.shape}')

    # Coerce Missing/error objects to NaN -> float64
    print(f'  coercing to float64...')
    df = df.apply(pd.to_numeric, errors='coerce').astype('float64')

    # Insert NaN rows for the failed SMILES so the matrix aligns with smiles_list
    desc_names = list(df.columns)
    mat = np.full((len(smiles_list), len(desc_names)), np.nan, dtype=np.float64)
    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    mat[valid_idx, :] = df.values
    print(f'  raw matrix: {mat.shape}, NaN fraction overall: {np.isnan(mat).sum()/mat.size:.4f}')

    raw = {
        'smiles_list':    smiles_list,
        'desc_names_raw': desc_names,
        'mat_raw':        mat,
        'rdkit_failed':   failed,
    }
    with open(OUT_RAW, 'wb') as f:
        pickle.dump(raw, f)
    print(f'  saved: {OUT_RAW}')
    return raw


# ============================================================
# STAGE 2: filtering (Phase A + B)
# ============================================================

def filter_features(raw):
    print(f'\n[STAGE 2] Filtering (Phase A + B)')
    smiles_list = raw['smiles_list']
    names = list(raw['desc_names_raw'])
    mat = raw['mat_raw'].copy()
    N, P0 = mat.shape
    print(f'  starting: {N} mols, {P0} descriptors')

    dropped_log = {'phaseA_inf': [], 'phaseA_nan': [], 'phaseB_var': [], 'phaseB_corr': []}

    # ---- Phase A1: drop columns with any Inf ----
    has_inf = np.any(np.isinf(mat), axis=0)
    cols_inf = [names[i] for i in range(P0) if has_inf[i]]
    dropped_log['phaseA_inf'] = cols_inf
    keep = ~has_inf
    mat = mat[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    print(f'  Phase A1 (drop any-Inf cols): dropped {len(cols_inf)},  remaining {mat.shape[1]}')

    # ---- Phase A2: drop columns with > NAN_FRAC_MAX NaN ----
    nan_rate = np.isnan(mat).mean(axis=0)
    high_nan_cols = [n for n, r in zip(names, nan_rate) if r > NAN_FRAC_MAX]
    dropped_log['phaseA_nan'] = [(n, float(nan_rate[i])) for i, n in enumerate(names) if nan_rate[i] > NAN_FRAC_MAX]
    keep = nan_rate <= NAN_FRAC_MAX
    mat = mat[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    print(f'  Phase A2 (drop >{NAN_FRAC_MAX*100:.0f}% NaN): dropped {len(high_nan_cols)},  remaining {mat.shape[1]}')

    # ---- Phase A3: median-impute remaining NaN ----
    n_missing = np.isnan(mat).sum()
    if n_missing > 0:
        medians = np.nanmedian(mat, axis=0)
        nan_mask = np.isnan(mat)
        col_idx = np.where(nan_mask)
        mat[col_idx] = medians[col_idx[1]]
        print(f'  Phase A3 (median-impute): filled {n_missing} cells across {len(np.unique(col_idx[1]))} cols')
    else:
        print(f'  Phase A3 (median-impute): no remaining NaN')

    # ---- save post-Phase-A as the 'raw cleaned' matrix for tree models ----
    mat_raw_clean = mat.copy()
    names_raw_clean = list(names)
    P_A = mat.shape[1]

    # ---- Phase B1: drop near-zero variance (z-score var < VAR_MIN) ----
    sd = mat.std(axis=0)
    near_zero = sd < 1e-12
    keep = ~near_zero
    dropped_log['phaseB_var'] = [n for n, k in zip(names, keep) if not k]
    mat = mat[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    print(f'  Phase B1 (zero-variance): dropped {(~keep).sum()},  remaining {mat.shape[1]}')

    # ---- z-score standardize ----
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0)
    mat_z = (mat - mu) / sd

    # ---- Phase B2: high-correlation pruning ----
    print(f'  Phase B2 (corr pruning, |r| > {CORR_THRESH})...')
    t0 = time.time()
    # Compute correlation in chunks to avoid memory blowup
    P = mat_z.shape[1]
    keep = np.ones(P, dtype=bool)
    # Use a greedy strategy: iterate columns in order; if any earlier kept col has |r|>thresh, drop current
    # Vectorize via batched matmul on z-scored data (already z-scored => cov = X^T X / N = corr)
    X = mat_z
    N = X.shape[0]
    corr_full = (X.T @ X) / N
    # corr matrix should be symmetric with 1.0 diag (within float precision)
    # Greedy upper-triangle traversal
    for j in range(P):
        if not keep[j]:
            continue
        # find any earlier i with |corr[i,j]| > thresh and keep[i]
        # actually we go in order; drop j if any kept i<j has high corr with j
        for i in range(j):
            if keep[i] and abs(corr_full[i, j]) > CORR_THRESH:
                keep[j] = False
                dropped_log['phaseB_corr'].append((names[j], names[i], float(corr_full[i, j])))
                break
    dt = time.time() - t0
    mat_z = mat_z[:, keep]
    names_z = [n for n, k in zip(names, keep) if k]
    print(f'    dropped {(~keep).sum()} in {dt:.1f}s,  remaining {mat_z.shape[1]}')

    P_B = mat_z.shape[1]
    print(f'\n  FINAL SHAPE: raw-cleaned (Phase A) = {N} x {P_A};  zscore (Phase A+B) = {N} x {P_B}')

    filt = {
        'smiles_list':       smiles_list,
        'desc_names_raw':    names_raw_clean,    # 1613 -> P_A
        'mat_raw':           mat_raw_clean,       # for tree models, no scaling needed
        'desc_names_zscore': names_z,             # P_A -> P_B
        'mat_zscore':        mat_z,               # for JSD / cosine / distance
        'dropped_log':       dropped_log,
    }
    with open(OUT_FILT, 'wb') as f:
        pickle.dump(filt, f)
    print(f'  saved: {OUT_FILT}')
    return filt


# ============================================================
# STAGE 3: build per-endpoint datasets parallel to datasets.pkl
# ============================================================

def build_per_endpoint(filt):
    print(f'\n[STAGE 3] Building per-endpoint Mordred dataset')
    with open(DATASETS_IN, 'rb') as f:
        ds = pickle.load(f)
    smiles_list = filt['smiles_list']
    smi2idx = {s: i for i, s in enumerate(smiles_list)}

    out = {}
    for endpoint, v in ds.items():
        ep_smiles = v['smiles']
        idx = np.array([smi2idx[s] for s in ep_smiles], dtype=np.int64)
        # Use raw cleaned (for tree models)
        Xr = filt['mat_raw'][idx, :]
        Xz = filt['mat_zscore'][idx, :]
        out[endpoint] = {
            'X_mordred_raw': Xr,
            'X_mordred_z':   Xz,
            'y':             v['y'],
            'smiles':        ep_smiles,
        }
        print(f'  {endpoint:<35}  n={len(ep_smiles):>5}  X_raw={Xr.shape}  X_z={Xz.shape}')

    with open(OUT_DSET, 'wb') as f:
        pickle.dump(out, f)
    print(f'  saved: {OUT_DSET}')
    return out


# ============================================================
# main
# ============================================================
if __name__ == '__main__':
    t0 = time.time()
    raw = compute_raw()
    filt = filter_features(raw)
    _   = build_per_endpoint(filt)
    print(f'\n[ALL DONE in {(time.time()-t0)/60:.1f} min]')
