"""Table S1: BERTScore F1 for correctly matched and mismatched paper pairs.

Input: data/bertscore_baseline/pair_scores.csv, the stored BERTScore F1 of every
matched pair (same paper) and of every mismatched pair (10 random reassignments
of the human entries within a field, no same-paper matches) from the run reported
in the paper (roberta-large, layer 17, no IDF weighting, no baseline rescaling).

This script needs no model: it aggregates the stored pair scores and checks the
result against the saved summary (baseline_summary.csv). The full analysis,
including BERTScore computation, can be rerun with notebooks/bertscore_baseline.ipynb.

The SD, minimum and maximum describe variation of the mismatched mean across the
10 reassignments. They are not confidence intervals: repeated pairings are not
independent papers.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, ROOT, ensure_dirs, write_table  # noqa: E402

BASE = DATA / 'bertscore_baseline'
FIELD_NAMES = {'Objective': 'Objectives', 'Sim. Method Advantages': 'Simulation Method Advantages',
               'Tools/Soft Used': 'Tools/Software Used'}


def compute():
    pairs = pd.read_csv(BASE / 'pair_scores.csv', encoding='utf-8-sig')
    saved = pd.read_csv(BASE / 'baseline_summary.csv', encoding='utf-8-sig')
    rows = []
    for rec in saved.itertuples():
        sub = pairs[(pairs.field == rec.field) & (pairs.metric == rec.metric)]
        matched = sub[sub.pair_type == 'matched']
        per_reassignment = sub[sub.pair_type == 'mismatched'].groupby('reassignment').score.mean()
        rows.append(dict(
            field=rec.field, field_type=rec.field_type, metric=rec.metric,
            n_papers=matched.llm_doi.nunique(), n_reassignments=len(per_reassignment),
            matched_mean=matched.score.mean(), mismatched_mean=per_reassignment.mean(),
            matched_minus_mismatched=matched.score.mean() - per_reassignment.mean(),
            mismatched_sd=per_reassignment.std(ddof=1),
            mismatched_min=per_reassignment.min(), mismatched_max=per_reassignment.max()))
    res = pd.DataFrame(rows)
    # Cross-check with the saved summary of the reported run.
    checks = {'n_papers': 'n_papers', 'n_reassignments': 'n_reassignments', 'matched_mean': 'matched_mean',
              'mismatched_mean': 'mismatched_mean', 'matched_minus_mismatched': 'matched_minus_mismatched',
              'mismatched_sd': 'reassignment_mean_sd', 'mismatched_min': 'reassignment_mean_min',
              'mismatched_max': 'reassignment_mean_max'}
    worst = 0.0
    for ours, theirs in checks.items():
        d = np.max(np.abs(res[ours].to_numpy(float) - saved[theirs].to_numpy(float)))
        worst = max(worst, d)
        assert d < 1e-12, (ours, d)
    return res, worst


def formatted(res):
    out = []
    for r in res.itertuples():
        out.append({
            'Field': FIELD_NAMES.get(r.field, r.field),
            'BERTScore F1': 'Narrative text' if r.field_type == 'free_text' else 'Whole set',
            'Eligible papers': r.n_papers,
            'Matched mean': f'{r.matched_mean:.3f}',
            'Mismatched mean ± SD': f'{r.mismatched_mean:.3f} ± {r.mismatched_sd:.4f}',
            'Mismatched range': f'{r.mismatched_min:.3f}–{r.mismatched_max:.3f}',
            'Difference': f'{r.matched_minus_mismatched:.3f}',
        })
    return pd.DataFrame(out)


def main():
    res, worst = compute()
    print(f'Cross-check with the saved baseline summary: max abs difference = {worst:.2e}')
    ensure_dirs()
    res.to_csv(ROOT / 'results' / 'tables' / 'bertscore_baseline_long.csv', index=False, lineterminator='\n')
    write_table(formatted(res), 'tableS1_bertscore_baseline',
                title='Table S1. BERTScore F1 for correctly matched and mismatched paper pairs',
                note='Each comparison includes papers with substantive human and LLM entries for that field. '
                     'Mismatched values summarize 10 random reassignments without same-paper matches: mean, '
                     'SD and range of the 10 reassignment means (variation across reassignments, not a '
                     'confidence interval). Differences were calculated before rounding. These comparisons '
                     'describe similarity relative to other papers in the corpus and do not estimate factual '
                     'accuracy.')
    print(formatted(res).to_string(index=False))


if __name__ == '__main__':
    main()
