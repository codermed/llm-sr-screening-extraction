"""Screening results: Table 3, Table 4, Table S2 and Figure 2 panels b-c.

Input: data/screening/confusion_matrices.csv, the three 4x4 confusion matrices of
the full-corpus evaluation (rows = reference labels, columns = predicted labels):

  HUM vs LLM.IS     final human labels vs title-abstract screening (n = 11,678)
  HUM vs LLM.DS     final human labels vs full-text screening      (n = 2,077)
  LLM.IS vs LLM.DS  title-abstract vs full-text screening           (n = 2,069)

The confusion matrix is a sufficient statistic for every reported screening
metric. Each record was processed once by the model, so run-to-run variability
cannot be estimated for this evaluation. Uncertainty is expressed as 95%
percentile bootstrap confidence intervals: records are resampled with
replacement (multinomial resampling of the matrix cells), 10,000 times per
comparison, with a fixed seed.

Conventions follow the original implementation (scikit-learn): undefined
precision, recall or F1 values are set to 0 and kept in the four-class macro
average. Retention metrics group OK, POSSIBLE and REVIEW as "retained".
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, FIGURES, ROOT, ensure_dirs, fmt_est_ci, write_table  # noqa: E402

LABELS = ['EXCLUDED', 'POSSIBLE', 'OK', 'REVIEW']
COMPARISONS = ['HUM vs LLM.IS', 'HUM vs LLM.DS', 'LLM.IS vs LLM.DS']
SEED = 20261123
N_BOOT = 10_000


def load_matrices(path=DATA / 'screening' / 'confusion_matrices.csv'):
    df = pd.read_csv(path)
    mats = {}
    for comp in COMPARISONS:
        sub = df[df['comparison'] == comp]
        m = sub.pivot(index='reference_label', columns='predicted_label', values='count')
        mats[comp] = m.loc[LABELS, LABELS].to_numpy(dtype=np.int64)
    return mats


def screening_metrics(M):
    """Metrics for one matrix (4, 4) or a stack of matrices (..., 4, 4)."""
    M = np.asarray(M, dtype=float)
    n = M.sum(axis=(-2, -1))
    tp = np.diagonal(M, axis1=-2, axis2=-1)
    row = M.sum(axis=-1)  # reference counts
    col = M.sum(axis=-2)  # predicted counts
    fp, fn = col - tp, row - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(col > 0, tp / col, 0.0)
        recall = np.where(row > 0, tp / row, 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    out = {
        'n': n,
        'accuracy': tp.sum(axis=-1) / n,
        'macro_precision': precision.mean(axis=-1),
        'macro_recall': recall.mean(axis=-1),
        'macro_f1': f1.mean(axis=-1),
    }
    for k, lab in enumerate(LABELS):
        out[f'precision_{lab}'] = precision[..., k]
        out[f'recall_{lab}'] = recall[..., k]
        out[f'f1_{lab}'] = f1[..., k]
        out[f'support_{lab}'] = row[..., k]
        out[f'predicted_{lab}'] = col[..., k]
    ex, ok = LABELS.index('EXCLUDED'), LABELS.index('OK')
    keep = [i for i in range(len(LABELS)) if i != ex]
    ref_retained = M[..., keep, :].sum(axis=(-2, -1))
    both_retained = M[..., keep, :][..., :, keep].sum(axis=(-2, -1))
    with np.errstate(divide='ignore', invalid='ignore'):
        out['retention_sensitivity'] = both_retained / ref_retained
        out['ok_retention'] = M[..., ok, keep].sum(axis=-1) / M[..., ok, :].sum(axis=-1)
    out['retained_both'] = both_retained
    out['reference_retained'] = ref_retained
    out['ok_retained'] = M[..., ok, keep].sum(axis=-1)
    out['reference_ok'] = M[..., ok, :].sum(axis=-1)
    out['predicted_excluded'] = M[..., :, ex].sum(axis=-1)
    out['volume_reduction'] = out['predicted_excluded'] / n
    out['missed_retained'] = M[..., keep, ex].sum(axis=-1)
    out['missed_ok'] = M[..., ok, ex]
    return out


def bootstrap(M, seed, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    n = int(M.sum())
    draws = rng.multinomial(n, M.ravel() / n, size=n_boot).reshape(n_boot, *M.shape)
    return screening_metrics(draws)


RATE_METRICS = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
                'retention_sensitivity', 'ok_retention', 'volume_reduction'] + \
               [f'{m}_{lab}' for lab in LABELS for m in ('precision', 'recall', 'f1')]


def compute(mats, n_boot=N_BOOT):
    rows = []
    for idx, comp in enumerate(COMPARISONS):
        M = mats[comp]
        point = screening_metrics(M)
        boot = bootstrap(M, seed=[SEED, idx], n_boot=n_boot)
        for metric in RATE_METRICS:
            lo, hi = np.nanpercentile(boot[metric], [2.5, 97.5])
            rows.append(dict(comparison=comp, metric=metric, estimate=float(point[metric]),
                             ci_low=float(lo), ci_high=float(hi)))
        for metric in ['n', 'retained_both', 'reference_retained', 'ok_retained', 'reference_ok',
                       'predicted_excluded', 'missed_retained', 'missed_ok'] + \
                      [f'support_{lab}' for lab in LABELS] + [f'predicted_{lab}' for lab in LABELS]:
            rows.append(dict(comparison=comp, metric=metric, estimate=float(point[metric]),
                             ci_low=np.nan, ci_high=np.nan))
    return pd.DataFrame(rows)


def label_distribution(mats):
    hum = mats['HUM vs LLM.IS'].sum(axis=1)
    is_ = mats['HUM vs LLM.IS'].sum(axis=0)
    ds = mats['HUM vs LLM.DS'].sum(axis=0)
    order = ['EXCLUDED', 'POSSIBLE', 'OK', 'REVIEW']
    def cell(v, i):
        return f'{int(v[i]):,} ({100 * v[i] / v.sum():.1f}%)'
    rows = [{'Category': lab, 'HUM': cell(hum, LABELS.index(lab)), 'LLM.IS': cell(is_, LABELS.index(lab)),
             'LLM.DS': cell(ds, LABELS.index(lab))} for lab in order]
    rows.append({'Category': 'Total', 'HUM': f'{int(hum.sum()):,}', 'LLM.IS': f'{int(is_.sum()):,}',
                 'LLM.DS': f'{int(ds.sum()):,}'})
    return pd.DataFrame(rows)


def formatted_table4(res):
    get = lambda comp, m: res[(res.comparison == comp) & (res.metric == m)].iloc[0]
    def rate(comp, m, pct):
        r = get(comp, m)
        return fmt_est_ci(r.estimate, r.ci_low, r.ci_high, pct=pct)
    def frac(comp, num, den):
        return f'{int(get(comp, num).estimate):,}/{int(get(comp, den).estimate):,}'
    spec = [
        ('Records compared, n', lambda c: f"{int(get(c, 'n').estimate):,}", True),
        ('Overall accuracy, %', lambda c: rate(c, 'accuracy', True), True),
        ('Macro-averaged precision', lambda c: rate(c, 'macro_precision', False), True),
        ('Macro-averaged recall', lambda c: rate(c, 'macro_recall', False), True),
        ('Macro-averaged F1', lambda c: rate(c, 'macro_f1', False), True),
        ('Recall, EXCLUDED, %', lambda c: rate(c, 'recall_EXCLUDED', True), True),
        ('Recall, POSSIBLE, %', lambda c: rate(c, 'recall_POSSIBLE', True), True),
        ('Recall, OK, %', lambda c: rate(c, 'recall_OK', True), True),
        ('Recall, REVIEW, %', lambda c: rate(c, 'recall_REVIEW', True), True),
        ('Retention of reference non-EXCLUDED records, %',
         lambda c: f"{rate(c, 'retention_sensitivity', True)}; {frac(c, 'retained_both', 'reference_retained')}", False),
        ('Retention of reference OK records, %',
         lambda c: f"{rate(c, 'ok_retention', True)}; {frac(c, 'ok_retained', 'reference_ok')}", False),
        ('Potential screening-volume reduction, %',
         lambda c: f"{rate(c, 'volume_reduction', True)}; {frac(c, 'predicted_excluded', 'n')}", False),
        ('Reference non-EXCLUDED records excluded by the model, n (of which OK)',
         lambda c: f"{int(get(c, 'missed_retained').estimate):,} ({int(get(c, 'missed_ok').estimate):,})", False),
    ]
    rows = []
    for label, fn, all_comparisons in spec:
        row = {'Metric': label}
        for comp in COMPARISONS:
            human_ref = comp.startswith('HUM')
            row[comp.replace(' vs ', '–')] = fn(comp) if (all_comparisons or human_ref) else '—'
        rows.append(row)
    return pd.DataFrame(rows)


def formatted_s2(res):
    rows = []
    get = lambda comp, m: res[(res.comparison == comp) & (res.metric == m)].iloc[0]
    for comp in COMPARISONS:
        for lab in LABELS:
            row = {'Comparison': comp.replace(' vs ', '–'), 'Class': lab,
                   'Reference n': f"{int(get(comp, f'support_{lab}').estimate):,}",
                   'Predicted n': f"{int(get(comp, f'predicted_{lab}').estimate):,}"}
            for m in ('precision', 'recall', 'f1'):
                r = get(comp, f'{m}_{lab}')
                row[m.capitalize() if m != 'f1' else 'F1'] = fmt_est_ci(r.estimate, r.ci_low, r.ci_high)
            rows.append(row)
    return pd.DataFrame(rows)


def figure(mats):
    """Figure 2 panels b-c with the original plotting function."""
    import matplotlib
    matplotlib.use('Agg')
    sys.path.insert(0, str(ROOT))
    from Libs.LR.LLMInsightsEvaluator import LLMInsightsEvaluator
    # One block of rows per comparison; the third label column is left empty so that each
    # block enters only its own comparison (the plotting code keeps rows valid in both columns).
    columns = {'HUM vs LLM.IS': ('HUM', 'LLM.IS'), 'HUM vs LLM.DS': ('HUM', 'LLM.DS'),
               'LLM.IS vs LLM.DS': ('LLM.IS', 'LLM.DS')}
    blocks = []
    for comp, (ref_col, pred_col) in columns.items():
        M = mats[comp]
        recs = [(LABELS[i], LABELS[j]) for i in range(4) for j in range(4) for _ in range(int(M[i, j]))]
        block = pd.DataFrame({'HUM': '', 'LLM.IS': '', 'LLM.DS': ''}, index=range(len(recs)))
        block[ref_col] = [r for r, _ in recs]
        block[pred_col] = [p for _, p in recs]
        blocks.append(block)
    df_analysis = pd.concat(blocks, ignore_index=True)
    ev = LLMInsightsEvaluator.__new__(LLMInsightsEvaluator)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)  # seaborn deprecation notices
        out = ev.plot_screening_performance(df_analysis=df_analysis, show=False, verbose=False,
                                            return_results=True)
    fig = out['figure'] if isinstance(out, dict) and 'figure' in out else out
    ensure_dirs()
    fig.savefig(FIGURES / 'fig2bc_screening.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES / 'fig2bc_screening.pdf', bbox_inches='tight', metadata={'CreationDate': None})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-figure', action='store_true', help='skip Figure 2b-c')
    args = ap.parse_args()

    mats = load_matrices()
    res = compute(mats)
    ensure_dirs()
    res.to_csv(ROOT / 'results' / 'tables' / 'screening_metrics_long.csv', index=False, lineterminator='\n')

    write_table(label_distribution(mats), 'table3_label_distribution',
                title='Table 3. Distribution of screening labels in the analysis sets',
                note='HUM and LLM.IS: the 11,678 records with recognized labels in both sources. '
                     'LLM.DS: the 2,077 records in the HUM–LLM.DS comparison.')
    write_table(formatted_table4(res), 'table4_screening_performance',
                title='Table 4. Screening agreement with the reference labels (estimate, 95% CI)',
                note='The first label set in each comparison is the reference. 95% CIs: percentile '
                     f'bootstrap over records ({N_BOOT:,} resamples, seed {SEED}). The full-corpus '
                     'evaluation used one model call per record, so no run-to-run variability is '
                     'available. Retention groups OK, POSSIBLE and REVIEW as retained; volume '
                     'reduction assumes that every model exclusion is accepted.')
    write_table(formatted_s2(res), 'tableS2_screening_per_class',
                title='Table S2. Class-specific precision, recall and F1 (estimate, 95% CI)',
                note=f'95% CIs: percentile bootstrap over records ({N_BOOT:,} resamples, seed {SEED}).')
    if not args.no_figure:
        figure(mats)
    print(formatted_table4(res).to_string(index=False))


if __name__ == '__main__':
    main()
