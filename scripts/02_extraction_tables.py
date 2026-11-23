"""Extraction results: Table 5, Table S3 and Figure 2 panel d.

Input: data/extraction/insights_extraction_detailed.csv, the per-paper scores of
the 342 analysed papers for every field and metric, as computed by
Libs/LR/LLMInsightsEvaluator.py from the human and LLM extraction tables.
The original summary statistics (insights_extraction_summary.csv) are used as a
cross-check: every mean, SD, median, quartile and 95% CI computed here must equal
the saved value.

The LLM extraction was run once per paper, so the uncertainty reported is the
spread across papers (SD) and a t-based 95% confidence interval for the mean.

Metrics (as in Figure 2d):
  categorical and Boolean fields  exact match
  temporal fields                 precision, recall and F1 on the sets of years
  set-based fields                whole-set BERTScore F1 and pairwise BERTScore F1
                                  (one-to-one item matching, threshold 0.70)
  narrative fields                BERTScore F1 and cosine similarity (all-mpnet-base-v2)
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, FIGURES, ROOT, ensure_dirs, write_table  # noqa: E402

EXTRACTION = DATA / 'extraction'

FIELDS = {  # field types and fields in Figure 2d (Continent and Region were not reported)
    'categorical': ['Country', 'Income Group', 'Global'],
    'boolean': ['Population as Predictor?', 'Validation?'],
    'temporal': ['Baseline Years', 'Target Years'],
    'set_based': ['LULC Classes', 'Scenarios', 'Analysis Methods', 'Predictors', 'Data Sources',
                  'Tools/Soft Used', 'Validation Methods', 'Validation Data Sources'],
    'free_text': ['Objective', 'Sim. Method Advantages', 'Key Findings', 'Implications',
                  'Limitations', 'Future Directions', 'Scalability', 'Novelty'],
}
MAIN_METRICS = {
    'categorical': ['match'], 'boolean': ['match'], 'temporal': ['precision', 'recall', 'f1'],
    'set_based': ['set_text_bert_f1', 'soft_f1'], 'free_text': ['bert_f1', 'cosine'],
}
SUPP_METRICS = {
    'categorical': ['match'], 'boolean': ['match'],
    'temporal': ['precision', 'recall', 'f1', 'jaccard'],
    'set_based': ['precision', 'recall', 'f1', 'jaccard', 'set_text_bert_f1',
                  'soft_precision', 'soft_recall', 'soft_f1'],
    'free_text': ['bert_p', 'bert_r', 'bert_f1', 'cosine'],
}
TYPE_NAMES = {'categorical': 'Categorical', 'boolean': 'Boolean', 'temporal': 'Temporal',
              'set_based': 'Set-based', 'free_text': 'Narrative'}
FIELD_NAMES = {'Tools/Soft Used': 'Tools/Software Used', 'Sim. Method Advantages': 'Simulation Method Advantages',
               'Population as Predictor?': 'Population as Predictor?', 'Objective': 'Objective'}
METRIC_NAMES = {
    'match': 'Exact match', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'jaccard': 'Jaccard',
    'set_text_bert_f1': 'BERT-F1 (whole set)', 'soft_precision': 'Pairwise precision (BERTScore)',
    'soft_recall': 'Pairwise recall (BERTScore)', 'soft_f1': 'BERT-F1 (pairwise)',
    'bert_p': 'BERTScore precision', 'bert_r': 'BERTScore recall', 'bert_f1': 'BERT-F1', 'cosine': 'Cosine similarity',
}


def describe(values):
    x = pd.Series(values, dtype=float).dropna()
    n = len(x)
    mean, sd = x.mean(), x.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return dict(n=n, mean=mean, sd=sd, ci_low=mean - half, ci_high=mean + half, median=x.median(),
                q1=x.quantile(0.25), q3=x.quantile(0.75), min=x.min(), max=x.max())


def compute(metric_map):
    detailed = pd.read_csv(EXTRACTION / 'insights_extraction_detailed.csv')
    rows = []
    for ftype, fields in FIELDS.items():
        for field in fields:
            sub = detailed[detailed['Column'] == field]
            assert len(sub) == 342, (field, len(sub))
            for metric in metric_map[ftype]:
                rows.append(dict(field_type=ftype, field=field, metric=metric, **describe(sub[metric])))
    return pd.DataFrame(rows)


def check_against_saved_summary(res, tol=1e-9):
    saved = pd.read_csv(EXTRACTION / 'insights_extraction_summary.csv')
    saved = saved.set_index(['Column', 'Metric'])
    cols = {'mean': 'Mean', 'sd': 'Std_Dev', 'median': 'Median', 'q1': 'Q1', 'q3': 'Q3',
            'ci_low': 'CI95_Low', 'ci_high': 'CI95_High', 'min': 'Min', 'max': 'Max'}
    worst = 0.0
    for r in res.itertuples():
        s = saved.loc[(r.field, r.metric)]
        for ours, theirs in cols.items():
            d = abs(getattr(r, ours) - float(s[theirs]))
            worst = max(worst, d)
            if d > tol:
                raise AssertionError(f'{r.field}/{r.metric}/{ours}: {getattr(r, ours)} vs saved {s[theirs]}')
    return worst


def formatted(res, digits):
    out = []
    for r in res.itertuples():
        out.append({
            'Field type': TYPE_NAMES[r.field_type],
            'Field': FIELD_NAMES.get(r.field, r.field),
            'Metric': METRIC_NAMES[r.metric],
            'n': r.n,
            'Mean ± SD': f'{r.mean:.{digits}f} ± {r.sd:.{digits}f}',
            '95% CI': f'{r.ci_low:.{digits}f}–{r.ci_high:.{digits}f}',
            'Median [IQR]': f'{r.median:.{digits}f} [{r.q1:.{digits}f}–{r.q3:.{digits}f}]',
        })
    return pd.DataFrame(out)


def figure():
    """Figure 2 panel d with the original plotting function and settings."""
    import matplotlib
    matplotlib.use('Agg')
    sys.path.insert(0, str(ROOT))
    from Libs.LR.LLMInsightsEvaluator import LLMInsightsEvaluator
    ev = LLMInsightsEvaluator.from_cache_csv(
        detailed_csv=str(EXTRACTION / 'insights_extraction_detailed.csv'),
        summary_csv=str(EXTRACTION / 'insights_extraction_summary.csv'),
        cross_csv=str(EXTRACTION / 'insights_cross_columns_metrics.csv'))
    columns = [c for fields in FIELDS.values() for c in fields]
    order = ['Country', 'Income Group', 'Global', 'Objective', 'LULC Classes', 'Baseline Years', 'Target Years',
             'Scenarios', 'Analysis Methods', 'Sim. Method Advantages', 'Tools/Soft Used', 'Predictors',
             'Population as Predictor?', 'Data Sources', 'Validation?', 'Validation Methods',
             'Validation Data Sources', 'Key Findings', 'Implications', 'Limitations', 'Future Directions',
             'Scalability', 'Novelty']
    assert sorted(order) == sorted(columns)
    np.random.seed(20261123)  # the plotting code jitters data points with the global NumPy RNG
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = ev.plot_metric_boxplots_by_category_minimal_improved(
            figsize=(11, 4.5),
            categories=['categorical', 'boolean', 'temporal', 'set_based', 'free_text'],
            columns=order,
            metrics=dict(categorical=['match'], boolean=['match'], temporal=['f1', 'precision', 'recall'],
                         set_based=['set_text_bert_f1', 'soft_f1'], free_text=['bert_f1', 'cosine'],
                         default=['f1']),
            metric_color_map={'match': '#d62728', 'f1': '#1f77b4', 'bert_f1': '#2ca02c', 'cosine': '#ffbf00'},
            percentage=False, xtick_rotation=45, xtick_fontsize=10, ytick_fontsize=10, ylabel='Score',
            add_grid=True, y_lim=(0.0, 1.0), auto_extend_y=True, y_padding=0.1, wspace=0.5,
            top_line_pad_pts=20, title_y=1.20,
            show_violin=True, violin_alpha=0.5, violin_edgecolor=None, violin_linewidth=0.8, violin_width=1.25,
            box_width_factor=0.65, box_edgecolor='#222222', box_linewidth=1.4, bar_edgecolor='#222222',
            bar_edgewidth=1.2, whisker_color='#444444', whisker_linewidth=1.2, cap_color='#444444',
            cap_linewidth=1.2, median_color='#000000', median_linewidth=1.6, mean_edgecolor='#000000',
            show_data=True, data_jitter=0.08, data_size=5, data_marker='o', data_alpha=0.6,
            data_edgecolor=None, data_linewidth=0, data_color=None,
            show_outliers=True, flier_marker='x', flier_size=4, flier_edgecolor=None, flier_alpha=0.9,
            show_means=True, mean_marker_color='red', mean_marker_size=6, mean_marker_style='o',
            use_bars_for_match=True, bar_width_factor=0.65, bar_alpha=0.7, error_bar_color='black',
            error_bar_linewidth=1.5, capsize=4,
            print_stats=False, return_stats_df=False,
        )
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    ensure_dirs()
    fig.savefig(FIGURES / 'fig2d_extraction.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES / 'fig2d_extraction.pdf', bbox_inches='tight', metadata={'CreationDate': None})
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-figure', action='store_true', help='skip Figure 2d')
    args = ap.parse_args()

    main_res = compute(MAIN_METRICS)
    supp_res = compute(SUPP_METRICS)
    worst = max(check_against_saved_summary(main_res), check_against_saved_summary(supp_res))
    print(f'Cross-check with the saved summary statistics: max abs difference = {worst:.2e}')

    ensure_dirs()
    supp_res.to_csv(ROOT / 'results' / 'tables' / 'extraction_metrics_long.csv', index=False, lineterminator='\n')
    note = ('n = 342 analysed papers (345 attempted; two refusals and one unusable output excluded). '
            'SD: spread across papers. 95% CI: t-based interval for the mean. The extraction was run once '
            'per paper, so run-to-run variability is not available. Comparisons with two empty sets score 1 '
            'and with one empty set score 0.')
    write_table(formatted(main_res, 2), 'table5_extraction_performance',
                title='Table 5. Agreement between LLM and human extraction by field', note=note)
    write_table(formatted(supp_res, 3), 'tableS3_extraction_all_metrics',
                title='Table S3. Extraction agreement by field: all set, temporal and text metrics', note=note)
    if not args.no_figure:
        figure()
    print(formatted(main_res, 2).to_string(index=False))


if __name__ == '__main__':
    main()
