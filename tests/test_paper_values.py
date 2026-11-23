"""Check that the analysis code reproduces the values reported in the paper.

Run from the repository root:  pytest -q
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), ROOT / 'scripts' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def screening():
    mod = load('01_screening_tables.py')
    mats = mod.load_matrices()
    return mod, mats, {c: mod.screening_metrics(mats[c]) for c in mod.COMPARISONS}


def test_table3_label_counts(screening):
    _, mats, _ = screening
    is_, ds = mats['HUM vs LLM.IS'], mats['HUM vs LLM.DS']
    # order: EXCLUDED, POSSIBLE, OK, REVIEW
    assert is_.sum(axis=1).tolist() == [10797, 615, 140, 126]   # HUM
    assert is_.sum(axis=0).tolist() == [9887, 1238, 491, 62]    # LLM.IS
    assert ds.sum(axis=0).tolist() == [1730, 103, 176, 68]      # LLM.DS
    assert [int(m.sum()) for m in mats.values()] == [11678, 2077, 2069]


def test_accuracy_and_macro_f1(screening):
    _, _, m = screening
    assert round(100 * m['HUM vs LLM.IS']['accuracy'], 2) == 86.00
    assert round(100 * m['HUM vs LLM.DS']['accuracy'], 2) == 59.70
    assert round(100 * m['LLM.IS vs LLM.DS']['accuracy'], 2) == 25.33
    assert round(m['HUM vs LLM.IS']['macro_f1'], 3) == 0.505
    assert round(m['HUM vs LLM.DS']['macro_f1'], 3) == 0.464
    assert round(100 * m['HUM vs LLM.IS']['recall_OK'], 1) == 81.4
    assert round(100 * m['HUM vs LLM.DS']['recall_OK'], 1) == 75.7


def test_retention_and_volume_reduction(screening):
    _, _, m = screening
    a, b = m['HUM vs LLM.IS'], m['HUM vs LLM.DS']
    assert (a['retained_both'], a['reference_retained']) == (602, 881)
    assert (b['retained_both'], b['reference_retained']) == (219, 873)
    assert (a['ok_retained'], a['reference_ok']) == (134, 140)
    assert (b['ok_retained'], b['reference_ok']) == (127, 140)
    assert (a['predicted_excluded'], a['n']) == (9887, 11678)
    assert (b['predicted_excluded'], b['n']) == (1730, 2077)
    assert (a['missed_retained'], a['missed_ok']) == (279, 6)
    assert (b['missed_retained'], b['missed_ok']) == (654, 13)
    assert round(100 * a['retention_sensitivity'], 1) == 68.3
    assert round(100 * b['retention_sensitivity'], 1) == 25.1
    assert round(100 * a['volume_reduction'], 1) == 84.7


def test_bootstrap_is_reproducible(screening):
    mod, mats, _ = screening
    r1 = mod.compute(mats, n_boot=500)
    r2 = mod.compute(mats, n_boot=500)
    assert np.allclose(r1[['ci_low', 'ci_high']].fillna(-1), r2[['ci_low', 'ci_high']].fillna(-1))
    f1 = r1[(r1.comparison == 'HUM vs LLM.IS') & (r1.metric == 'macro_f1')].iloc[0]
    assert f1.ci_low < f1.estimate < f1.ci_high


def test_extraction_statistics():
    mod = load('02_extraction_tables.py')
    res = mod.compute(mod.MAIN_METRICS)
    assert mod.check_against_saved_summary(res) < 1e-9
    get = lambda f, m: res[(res.field == f) & (res.metric == m)].iloc[0]
    assert (res.n == 342).all()
    assert round(get('Country', 'match')['mean'], 3) == 0.775
    assert round(get('Global', 'match')['mean'], 3) == 0.892
    assert round(get('Income Group', 'match')['mean'], 3) == 0.857
    r = get('Population as Predictor?', 'match')
    assert (round(r['mean'], 2), round(r['ci_low'], 2), round(r['ci_high'], 2)) == (0.78, 0.74, 0.82)
    assert round(get('Baseline Years', 'f1')['mean'], 2) == 0.78
    assert round(get('Target Years', 'f1')['mean'], 2) == 0.85
    assert round(get('LULC Classes', 'set_text_bert_f1')['mean'], 2) == 0.94
    assert round(get('Objective', 'bert_f1')['mean'], 3) == 0.918
    assert round(get('Scalability', 'cosine')['mean'], 2) == 0.51


def test_table_s1():
    mod = load('03_bertscore_baseline.py')
    res, worst = mod.compute()
    assert worst < 1e-12
    row = res.set_index('field').loc['Scalability']
    assert (row.n_papers, round(row.matched_mean, 3), round(row.mismatched_mean, 3)) == (255, 0.877, 0.854)
    row = res.set_index('field').loc['Objective']
    assert round(row.matched_minus_mismatched, 3) == 0.076


def test_model_comparison(tmp_path):
    mod = load('04_model_comparison_tables.py')
    mod.summarize(tmp_path / 'initial')
    mod.summarize(tmp_path / 'reviewed', review=mod.REVIEW)
    res = mod.aggregate(tmp_path / 'initial', tmp_path / 'reviewed').set_index('model')
    expected = {  # model: (mean agreement %, weighted %, same label n, auto-parsed calls)
        'Qwen3-Max': (56.8, 60.3, 148, 468),
        'Claude Haiku 4.5': (53.6, 64.5, 136, 363),
        'GPT-4.1-mini': (49.6, 61.0, 152, 468),
        'DeepSeek V3.2': (45.9, 61.0, 135, 468),
    }
    for model, (acc, wacc, same, parsed) in expected.items():
        r = res.loc[model]
        assert round(100 * r.agreement_mean, 1) == acc
        assert round(100 * r.weighted_agreement_mean, 1) == wacc
        assert r.same_label_all_three == same
        assert r.auto_parsed_calls == parsed
    assert round(100 * res.agreement_sd.max(), 1) == 2.4
