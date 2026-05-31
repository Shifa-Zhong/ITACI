"""Final scaffold-primary analysis: paired Wilcoxon + Bonferroni, bootstrap CIs,
Cohen's d_z, lambda distribution, DL-3 SHAP pair table.

Inputs (all under D:/quxintong/results/):
  scaffold_baseline_results.pkl     -> baseline {EP: [10 rep dicts with AUC, y_pred, ...]}
  scaffold_strategy1.pkl            -> S1 cross-model stacking
  scaffold_strategy2.pkl            -> S2 multi-task learning (MTL)
  scaffold_strategy3.pkl            -> S3 MAML
  scaffold_strategy3_fefa.pkl       -> FEFA (frozen-encoder feature augmentation)
  scaffold_dl_shap_pairs.pkl        -> Data-level strategy 1.3 (SHAP pairs)

Each strategy pkl is {EP: [10 rep dicts with AUC, baseline_AUC, ...]}.
ΔAUC per (strategy, EP, rep) = AUC - baseline_AUC (paired). The baseline_AUC field
matches scaffold_baseline_results.pkl AUC (sanity-checked below).

Outputs:
  results/scaffold_final_summary.pkl    -> all stats (Wilcoxon, CI, d_z, lambdas)
  results/scaffold_final_summary.xlsx   -> human-readable workbook
  logs/scaffold_final_summary.log       -> stdout transcript
"""
import os, sys, pickle, json, math
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

RES = r'D:/quxintong/results'
OUT_PKL = os.path.join(RES, 'scaffold_final_summary.pkl')
OUT_XLSX = os.path.join(RES, 'scaffold_final_summary.xlsx')
RNG_SEED = 20260517
N_BOOT = 10_000

STRATEGIES = [
    ('S1_stacking',  'scaffold_strategy1.pkl',       True),   # has lambda
    ('S2_MTL',       'scaffold_strategy2.pkl',       True),
    ('S3_MAML',      'scaffold_strategy3.pkl',       False),  # fixed 0.6/0.4
    ('FEFA',         'scaffold_strategy3_fefa.pkl',  True),
]


def load(name):
    with open(os.path.join(RES, name), 'rb') as f:
        return pickle.load(f)


def paired_stats(delta):
    """delta = np.array of paired ΔAUC values."""
    delta = np.asarray(delta, dtype=np.float64)
    n = len(delta)
    mean = float(np.mean(delta))
    median = float(np.median(delta))
    std = float(np.std(delta, ddof=1)) if n > 1 else float('nan')
    cohens_dz = mean / std if std > 0 else float('nan')
    # paired Wilcoxon signed-rank (zero handling: zsplit avoids dropping zeros silently)
    try:
        w = stats.wilcoxon(delta, alternative='two-sided', zero_method='zsplit')
        wstat, pval = float(w.statistic), float(w.pvalue)
    except ValueError:
        wstat, pval = float('nan'), 1.0
    # bootstrap 95% CI for mean Δ (paired resample of indices)
    rng = np.random.default_rng(RNG_SEED)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.mean(delta[idx])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=n, mean=mean, median=median, std=std,
                cohens_dz=cohens_dz, wilcoxon_stat=wstat, wilcoxon_p=pval,
                ci_lo=float(ci_lo), ci_hi=float(ci_hi))


def main():
    print('=== Scaffold final analysis ===', flush=True)
    base = load('scaffold_baseline_results.pkl')
    endpoints = list(base.keys())
    print(f'Endpoints (n={len(endpoints)}): {endpoints}', flush=True)

    # Build a long-form record table: one row per (strategy, EP, rep)
    rows = []
    lambda_records = []  # (strategy, EP, rep, lambda)
    for strat_name, fname, has_lambda in STRATEGIES:
        d = load(fname)
        for ep in endpoints:
            if ep not in d:
                print(f'  WARNING: {strat_name} missing endpoint {ep}', flush=True)
                continue
            reps = d[ep]
            if len(reps) != 10:
                print(f'  WARNING: {strat_name}/{ep} has {len(reps)} reps (expected 10)', flush=True)
            for rep_i, r in enumerate(reps):
                rows.append(dict(
                    strategy=strat_name, endpoint=ep, rep=rep_i,
                    auc=float(r['AUC']), baseline_auc=float(r['baseline_AUC']),
                    delta=float(r['AUC']) - float(r['baseline_AUC']),
                    lam=float(r.get('lambda', float('nan'))) if has_lambda else float('nan'),
                ))
                if has_lambda and 'lambda' in r:
                    lambda_records.append(dict(strategy=strat_name, endpoint=ep, rep=rep_i, lam=float(r['lambda'])))
    long_df = pd.DataFrame(rows)
    print(f'\nLong-form table: {len(long_df)} rows', flush=True)

    # Sanity: baseline_AUC stored in strategy pkls should match scaffold_baseline_results AUC
    print('\n[sanity] baseline_AUC alignment (first 3 EPs × first strategy):', flush=True)
    s1 = load(STRATEGIES[0][1])
    for ep in endpoints[:3]:
        for rep_i in range(min(3, len(s1[ep]))):
            stored = float(s1[ep][rep_i]['baseline_AUC'])
            actual = float(base[ep][rep_i]['AUC'])
            assert abs(stored - actual) < 1e-9, f'baseline mismatch at {ep}/{rep_i}: {stored} vs {actual}'
    print('  ✓ baseline_AUC fields match scaffold_baseline_results.pkl AUC values', flush=True)

    # ----- (1) PER-ENDPOINT STATS (per strategy × endpoint) -----
    per_ep_rows = []
    for strat_name, _, _ in STRATEGIES:
        sub = long_df[long_df.strategy == strat_name]
        for ep in endpoints:
            sub_ep = sub[sub.endpoint == ep]
            if len(sub_ep) == 0:
                continue
            st = paired_stats(sub_ep.delta.values)
            per_ep_rows.append(dict(
                strategy=strat_name, endpoint=ep,
                **st,
                mean_baseline=float(sub_ep.baseline_auc.mean()),
                mean_strategy=float(sub_ep.auc.mean()),
            ))
    per_ep = pd.DataFrame(per_ep_rows)

    # Bonferroni: 13 EP × 4 strategies = 52 tests
    n_tests = len(per_ep)
    per_ep['p_bonf'] = (per_ep['wilcoxon_p'] * n_tests).clip(upper=1.0)
    per_ep['sig_005'] = per_ep['p_bonf'] < 0.05
    per_ep['sig_001'] = per_ep['p_bonf'] < 0.01

    # ----- (2) PER-STRATEGY AGGREGATE (across all 13 EPs × 10 reps = 130 paired diffs) -----
    per_strat_rows = []
    for strat_name, _, _ in STRATEGIES:
        sub = long_df[long_df.strategy == strat_name]
        st = paired_stats(sub.delta.values)
        # also: endpoint-level summary (mean of per-EP means)
        ep_means = sub.groupby('endpoint').delta.mean().values
        st['n_ep_positive'] = int(np.sum(ep_means > 0))
        st['n_ep_negative'] = int(np.sum(ep_means < 0))
        st['n_ep_total'] = len(ep_means)
        # sign-test on endpoint means
        if len(ep_means) > 0:
            n_pos = int(np.sum(ep_means > 0))
            n_neg = int(np.sum(ep_means < 0))
            try:
                bt = stats.binomtest(n_pos, n_pos + n_neg, 0.5, alternative='two-sided')
                st['signtest_p'] = float(bt.pvalue)
            except Exception:
                st['signtest_p'] = float('nan')
        per_strat_rows.append(dict(strategy=strat_name, **st))
    per_strat = pd.DataFrame(per_strat_rows)
    # Bonferroni across 4 strategies
    per_strat['p_bonf'] = (per_strat['wilcoxon_p'] * len(per_strat)).clip(upper=1.0)
    per_strat['sig_005'] = per_strat['p_bonf'] < 0.05

    # ----- (3) LAMBDA DISTRIBUTION -----
    lam_df = pd.DataFrame(lambda_records)
    lam_dist = (lam_df.groupby(['strategy', 'lam']).size()
                .unstack(fill_value=0)
                .reset_index())
    lam_summary = (lam_df.groupby('strategy').lam
                   .agg(['mean', 'median', 'std', 'min', 'max', 'count']).reset_index())

    # ----- (4) DL-3 SHAP pair table -----
    dl = load('scaffold_dl_shap_pairs.pkl')
    pair_rows = []
    for k, reps in dl.items():
        # key format: "merge:A+B->test:T"
        try:
            merge_part, test_part = k.split('->test:')
            pair_str = merge_part.replace('merge:', '')
        except ValueError:
            pair_str, test_part = k, ''
        aucs = np.array([r['AUC'] for r in reps], dtype=np.float64)
        # paired baseline = scaffold_baseline_results[test_part] AUCs (same 10 reps)
        if test_part in base:
            base_aucs = np.array([r['AUC'] for r in base[test_part]], dtype=np.float64)
            delta = aucs - base_aucs
            st = paired_stats(delta)
        else:
            st = dict(n=len(aucs), mean=float('nan'), median=float('nan'), std=float('nan'),
                      cohens_dz=float('nan'), wilcoxon_stat=float('nan'), wilcoxon_p=float('nan'),
                      ci_lo=float('nan'), ci_hi=float('nan'))
        pair_rows.append(dict(
            pair=pair_str, test_endpoint=test_part,
            mean_pair_auc=float(aucs.mean()), mean_base_auc=float(base[test_part][0]['AUC']) if test_part in base else float('nan'),
            **st,
        ))
    dl_pairs = pd.DataFrame(pair_rows)
    # Bonferroni across DL-3 pair tests
    if len(dl_pairs):
        dl_pairs['p_bonf'] = (dl_pairs['wilcoxon_p'] * len(dl_pairs)).clip(upper=1.0)
        dl_pairs['sig_005'] = dl_pairs['p_bonf'] < 0.05

    # ----- Print summary tables -----
    print('\n\n========== PER-STRATEGY AGGREGATE (paired across 130 obs) ==========', flush=True)
    cols = ['strategy', 'n', 'mean', 'median', 'std', 'ci_lo', 'ci_hi',
            'cohens_dz', 'wilcoxon_p', 'p_bonf', 'sig_005',
            'n_ep_positive', 'n_ep_negative', 'signtest_p']
    print(per_strat[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'), flush=True)

    print('\n\n========== LAMBDA SUMMARY ==========', flush=True)
    print(lam_summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'), flush=True)
    print('\nLambda distribution (count per λ grid):', flush=True)
    print(lam_dist.to_string(index=False), flush=True)

    print('\n\n========== PER-ENDPOINT × STRATEGY (sig only) ==========', flush=True)
    sig = per_ep[per_ep['sig_005']]
    if len(sig):
        print(sig[['strategy', 'endpoint', 'mean', 'ci_lo', 'ci_hi', 'cohens_dz', 'wilcoxon_p', 'p_bonf']]
              .to_string(index=False, float_format=lambda x: f'{x:.4f}'), flush=True)
    else:
        print('  (no significant per-EP results after Bonferroni p<0.05)', flush=True)

    print('\n\n========== DL-3 SHAP PAIRS ==========', flush=True)
    print(dl_pairs[['pair', 'test_endpoint', 'n', 'mean', 'ci_lo', 'ci_hi', 'cohens_dz', 'wilcoxon_p', 'p_bonf', 'sig_005']]
          .to_string(index=False, float_format=lambda x: f'{x:.4f}'), flush=True)

    # ----- Save -----
    summary = dict(
        endpoints=endpoints,
        long_form=long_df,
        per_endpoint=per_ep,
        per_strategy=per_strat,
        lambda_summary=lam_summary,
        lambda_distribution=lam_dist,
        dl_pairs=dl_pairs,
        config=dict(n_boot=N_BOOT, seed=RNG_SEED, n_strategies=len(STRATEGIES),
                    bonf_n_per_ep_tests=n_tests, bonf_n_strat_tests=len(per_strat)),
    )
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(summary, f)
    print(f'\n✓ Saved pkl: {OUT_PKL}', flush=True)

    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as xw:
        per_strat.to_excel(xw, sheet_name='per_strategy', index=False)
        per_ep.to_excel(xw, sheet_name='per_endpoint', index=False)
        lam_summary.to_excel(xw, sheet_name='lambda_summary', index=False)
        lam_dist.to_excel(xw, sheet_name='lambda_distribution', index=False)
        dl_pairs.to_excel(xw, sheet_name='dl3_shap_pairs', index=False)
        long_df.to_excel(xw, sheet_name='long_form', index=False)
    print(f'✓ Saved xlsx: {OUT_XLSX}', flush=True)


if __name__ == '__main__':
    main()
