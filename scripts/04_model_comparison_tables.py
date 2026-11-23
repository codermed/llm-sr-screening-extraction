"""Supplementary four-model comparison: Table 8 (Table 6 in the first revision).

1. Rebuilds the summaries from the stored API responses, without calling any API:
   model_comparison/revision_experiment.py summarize
     data/model_comparison/run/{protocol.json, input_manifest.json, calls/*.json}
   once without and once with the documented manual label review
   (data/model_comparison/manual_label_review.csv: 120 Claude Haiku 4.5 responses whose
   final decision was in a format the parser did not recognize).
2. Aggregates the three repetitions per model: mean, SD (sample SD, n = 3) and range of
   the exact agreement with the human reference on the 156 randomly selected papers.

SDs across repetitions describe run-to-run variability on the same papers. They do not
describe sampling uncertainty; the SD across the three 52-paper subsets is given as an
indication of sampling variability.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, ROOT, ensure_dirs, write_table  # noqa: E402

RUN = DATA / 'model_comparison' / 'run'
REVIEW = DATA / 'model_comparison' / 'manual_label_review.csv'
OUT = ROOT / 'results' / 'model_comparison'
RUNNER = ROOT / 'model_comparison' / 'revision_experiment.py'
LABELS = ['EXCLUDED', 'POSSIBLE', 'OK', 'REVIEW']
MODELS = [  # display order of the paper
    ('qwen', 'Qwen3-Max', 'qwen/qwen3-max', 'Alibaba'),
    ('claude', 'Claude Haiku 4.5', 'anthropic/claude-haiku-4.5', 'Anthropic'),
    ('openai', 'GPT-4.1-mini', 'openai/gpt-4.1-mini', 'OpenAI'),
    ('deepseek', 'DeepSeek V3.2', 'deepseek/deepseek-v3.2', 'AtlasCloud (FP8)'),
]
KEEP = ['screening_metrics.csv', 'subset_screening_metrics.csv', 'screening_per_class.csv',
        'screening_confusion.csv', 'paper_recurrence.csv', 'paired_model_comparisons.csv',
        'subset_model_comparisons.csv', 'cost_summary.csv', 'deployment_audit.csv', 'INTERPRETATION.txt']


def summarize(out_dir, review=None):
    cmd = [sys.executable, str(RUNNER), 'summarize', '--run', str(RUN), '--out', str(out_dir)]
    if review:
        cmd += ['--review', str(review)]
    subprocess.run(cmd, check=True, cwd=RUNNER.parent, stdout=subprocess.DEVNULL)


def read(path):
    return pd.read_csv(path, encoding='utf-8-sig')


def macro_f1(conf):
    """Four-class macro-F1 from long-format confusion counts (undefined values = 0)."""
    m = conf.pivot_table(index='human_label', columns='predicted_label', values='count',
                         aggfunc='sum', fill_value=0).reindex(index=LABELS, columns=LABELS, fill_value=0)
    m = m.to_numpy(float)
    tp = np.diag(m)
    fp, fn = m.sum(0) - tp, m.sum(1) - tp
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-300), 0.0)
    return f1.mean()


def aggregate(initial, reviewed):
    sm = read(reviewed / 'screening_metrics.csv')
    sm_init = read(initial / 'screening_metrics.csv')
    rec = read(reviewed / 'paper_recurrence.csv')
    sub = read(reviewed / 'subset_screening_metrics.csv')
    conf = read(reviewed / 'screening_confusion.csv')
    rows = []
    for key, name, model_id, provider in MODELS:
        s = sm[sm.model_name == key].sort_values('replicate')
        assert list(s.replicate) == [1, 2, 3] and (s.usable == 156).all()
        acc = s.accuracy_among_usable.to_numpy()
        wacc = s.weighted_accuracy_among_usable.to_numpy()
        parsed = int(sm_init[sm_init.model_name == key].usable.sum())
        r = rec[(rec.model_name == key) & (rec.task == 'DS') & (rec.groups == 'DS_random_union')]
        assert len(r) == 156
        same = int(r.all_three_same_DS_label.sum())
        per_subset = sub[sub.model_name == key].groupby('subset_id').accuracy_among_usable.mean()
        assert len(per_subset) == 3
        f1 = np.array([macro_f1(conf[(conf.model_name == key) & (conf.replicate == rep)]) for rep in (1, 2, 3)])
        rows.append(dict(
            model=name, model_id=model_id, provider=provider,
            agreement_mean=acc.mean(), agreement_sd=acc.std(ddof=1), agreement_min=acc.min(), agreement_max=acc.max(),
            agreement_r1=acc[0], agreement_r2=acc[1], agreement_r3=acc[2],
            weighted_agreement_mean=wacc.mean(), weighted_agreement_sd=wacc.std(ddof=1),
            macro_f1_mean=f1.mean(), macro_f1_sd=f1.std(ddof=1),
            subset_agreement_sd=per_subset.std(ddof=1),
            same_label_all_three=same, same_label_pct=same / 156,
            auto_parsed_calls=parsed, calls=468))
    return pd.DataFrame(rows)


def formatted(res):
    return pd.DataFrame([{
        'Model (requested ID; provider)': f'{r.model} ({r.model_id}; {r.provider})',
        'Agreement, mean ± SD (range), %':
            f'{100 * r.agreement_mean:.1f} ± {100 * r.agreement_sd:.1f} ({100 * r.agreement_min:.1f}–{100 * r.agreement_max:.1f})',
        'Weighted agreement, mean ± SD, %': f'{100 * r.weighted_agreement_mean:.1f} ± {100 * r.weighted_agreement_sd:.1f}',
        'Macro-F1, mean ± SD': f'{r.macro_f1_mean:.3f} ± {r.macro_f1_sd:.3f}',
        'Same label in all three calls, n (%)': f'{r.same_label_all_three} ({100 * r.same_label_pct:.1f})',
        'Automatically parsed calls, n/468': f'{r.auto_parsed_calls}/468',
    } for r in res.itertuples()])


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        summarize(tmp / 'initial')
        summarize(tmp / 'reviewed', review=REVIEW)
        res = aggregate(tmp / 'initial', tmp / 'reviewed')
        for kind in ('initial', 'reviewed'):
            (OUT / kind).mkdir(parents=True, exist_ok=True)
            for name in KEEP:
                shutil.copy2(tmp / kind / name, OUT / kind / name)
    ensure_dirs()
    res.to_csv(ROOT / 'results' / 'tables' / 'model_comparison_long.csv', index=False, lineterminator='\n')
    sd_sub = ', '.join(f'{r.model} {100 * r.subset_agreement_sd:.1f}' for r in res.itertuples())
    write_table(formatted(res), 'table8_model_comparison',
                title='Table 8. Screening agreement and repeatability in the balanced random sample of 156 papers',
                note='Each model screened the same 156 randomly selected papers (39 per class) three times '
                     '(468 calls per model). Agreement: exact four-category agreement with the human reference; '
                     'mean, sample SD and range across the three repetitions (run-to-run variability). Weighted '
                     'agreement uses the class distribution of the eligible collection. Macro-F1: four-class '
                     'macro-averaged F1 per repetition. Agreement and repeatability include 105 Claude Haiku 4.5 '
                     'labels recovered by documented manual inspection; automatically parsed calls are reported '
                     f'separately. SD of mean agreement across the three 52-paper subsets (percentage points): {sd_sub}.')
    print(formatted(res).to_string(index=False))


if __name__ == '__main__':
    main()
