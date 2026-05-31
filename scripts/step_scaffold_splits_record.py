"""Generate scaffold splits for all 13 endpoints × N=10 random seeds,
analogous to splits_record.pkl for random splits.

Output: results/scaffold_splits_record.pkl
  Schema: dict[endpoint -> list[N_REPS] of (train_idx, test_idx)]

All downstream scaffold-split experiments (S1/S2/S3, controls, clean-source,
RF, arch sensitivity) reuse these splits for consistency.
"""
import os, sys, pickle, time
from pathlib import Path
import numpy as np
from collections import defaultdict
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.insert(0, r'D:\quxintong\scripts')
from _checkpoint import load_dict, save_dict_atomic

RESULTS_DIR = r'D:\quxintong\results'
N_REPS = 10
RANDOM_SEEDS = list(range(101, 101 + N_REPS))  # 101..110

ENDPOINTS = [
    'prenatal_development_C', 'TSHR_agonist_activity_C',
    'respiratory_toxicity_C', 'ocular_toxicity_C',
    'ames_mutagenicity_C', 'reproductive_toxicity_C',
    'skin_corrosion_C', 'neurotoxicity_C',
    'Estrogen_Receptor_α_C', 'Androgen_Receptor_C',
    'cytotoxicity_C', 'Carcinogenicity_C', 'Hepatotoxicity_C'
]


def scaffold_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ''
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ''


def make_scaffold_split(smiles_arr, y_arr, test_frac=0.2, seed=42):
    """Bemis-Murcko scaffold-disjoint split.

    Group compounds by Murcko scaffold; sort groups largest-first with a
    seeded random tie-break; greedily assign whole groups to test until
    1.2 × target test size; leftover → train.  Guarantees no scaffold
    appears in both train and test.
    """
    scaffolds = defaultdict(list)
    for i, smi in enumerate(smiles_arr):
        scaffolds[scaffold_smiles(smi)].append(i)
    rng = np.random.RandomState(seed)
    groups = sorted(scaffolds.values(), key=lambda g: (-len(g), rng.random()))
    n_total = len(smiles_arr)
    n_test_target = int(round(n_total * test_frac))
    test_idx, train_idx = [], []
    for g in groups:
        if len(test_idx) + len(g) <= n_test_target * 1.2:
            test_idx.extend(g)
        else:
            train_idx.extend(g)
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def main():
    print("=" * 70, flush=True)
    print(f"GENERATE scaffold_splits_record.pkl  ({len(ENDPOINTS)} endpoints × N={N_REPS} reps)", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    out_path = os.path.join(RESULTS_DIR, 'scaffold_splits_record.pkl')
    print(f"Loading datasets...", flush=True)
    with open(os.path.join(RESULTS_DIR, 'datasets.pkl'), 'rb') as f:
        datasets = pickle.load(f)

    splits = load_dict(out_path)

    for ep in ENDPOINTS:
        splits.setdefault(ep, [])
        if len(splits[ep]) >= N_REPS:
            print(f"  {ep[:30]:<30} already complete", flush=True)
            continue
        y = datasets[ep]['y']
        smi = datasets[ep]['smiles']
        for rep in range(len(splits[ep]), N_REPS):
            seed = RANDOM_SEEDS[rep]
            tr, te = make_scaffold_split(smi, y, test_frac=0.2, seed=seed)
            # Check class balance in test
            if len(np.unique(y[te])) < 2:
                print(f"  WARN: {ep} rep{rep} single-class test; using fallback",
                      flush=True)
                # Fallback: swap some train→test until both classes appear
                missing_cls = (set([0, 1]) - set(y[te].tolist())).pop()
                cand = [i for i in tr if y[i] == missing_cls]
                if cand:
                    # Move one compound of missing class to test
                    swap = cand[0]
                    tr = np.array([i for i in tr if i != swap])
                    te = np.append(te, swap)
            splits[ep].append((tr, te))
            print(f"  {ep[:30]:<30} rep{rep}: ntr={len(tr)}, nte={len(te)}, "
                  f"pos_te={float(y[te].mean()):.3f}", flush=True)
        save_dict_atomic(splits, out_path)

    print(f"\nDone. Total time: {time.time()-t0:.0f}s", flush=True)
    print(f"Saved: {out_path}", flush=True)


if __name__ == '__main__':
    main()
