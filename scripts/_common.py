"""Shared helpers for the table and figure scripts."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
TABLES = ROOT / 'results' / 'tables'
FIGURES = ROOT / 'results' / 'figures'


def ensure_dirs():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, name: str, title: str = '', note: str = '') -> None:
    """Write a table as CSV and as a Markdown file (for reading on GitHub)."""
    ensure_dirs()
    df.to_csv(TABLES / f'{name}.csv', index=False, encoding='utf-8', lineterminator='\n')
    lines = []
    if title:
        lines += [f'**{title}**', '']
    cols = [str(c) for c in df.columns]
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '|'.join('---' for _ in cols) + '|')
    for _, row in df.iterrows():
        lines.append('| ' + ' | '.join('' if pd.isna(v) else str(v) for v in row.tolist()) + ' |')
    if note:
        lines += ['', note]
    (TABLES / f'{name}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def fmt_num(x, digits=3):
    return f'{x:.{digits}f}'


def fmt_est_ci(est, lo, hi, digits=3, pct=False):
    """'0.505 (0.490–0.520)' or, with pct=True, '68.3 (65.2–71.4)' (values in %)."""
    if pct:
        est, lo, hi, digits = 100 * est, 100 * lo, 100 * hi, 1
    return f'{est:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f})'


def fmt_mean_sd(mean, sd, digits=3):
    return f'{mean:.{digits}f} ± {sd:.{digits}f}'
