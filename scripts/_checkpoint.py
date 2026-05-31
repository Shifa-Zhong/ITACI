"""Lightweight checkpoint helpers for crash-resilient runs.

Every per-(endpoint, rep) script saves after each completed rep, and on
restart loads whatever was finished and skips it.  No extra dependencies.

Typical usage in a step script:

    from _checkpoint import load_dict, save_dict_atomic

    results = load_dict(out_path)            # {} or partial dict
    for ep in ENDPOINTS:
        results.setdefault(ep, [])
        for rep in range(len(results[ep]), N_REPS):
            metrics = expensive_compute(ep, rep)
            results[ep].append(metrics)
            save_dict_atomic(results, out_path)   # flush after every rep

The atomic save uses a tmp file + os.replace so a crash mid-write never
leaves a corrupt pickle on disk.
"""
import os
import pickle
from pathlib import Path


def load_dict(path):
    """Return the unpickled object at `path`, or an empty dict if missing."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    try:
        with open(p, 'rb') as f:
            obj = pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        # Corrupt (e.g. mid-write crash); start over.
        return {}
    return obj if isinstance(obj, dict) else {}


def save_dict_atomic(obj, path):
    """Write `obj` to `path` atomically: tmp + os.replace.
    Ensures readers never see a half-written file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)


def reps_done(results, endpoint, expected=10):
    """How many reps are already finished for an endpoint."""
    return len(results.get(endpoint, []))


def fully_done(results, endpoints, n_reps):
    """True if results contain n_reps for every endpoint."""
    return all(len(results.get(ep, [])) >= n_reps for ep in endpoints)
