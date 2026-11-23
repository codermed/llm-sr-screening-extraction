# LLM-assisted screening and extraction for systematic reviews: code and data

Code and data for:

> Derdouri, A., Oyama, T., Murayama, Y., & Masago, Y. *LLM-assisted screening and extraction for
> systematic reviews: performance and refusal-related missingness in a land-use evidence synthesis.*
> Environmental Evidence (under review).

The study evaluated a hosted large language model (Qwen Max, `qwen-max-2025-01-25`) for title–abstract
screening, full-text screening and structured data extraction in a corpus of national-scale
land-use/land-cover (LULC) prediction studies, against a single-assessor human reference. A
supplementary comparison tested four further model deployments with three repeated calls each.

The repository contains the code that ran the screening and extraction workflow, the evaluation code,
and the stored outputs needed to reproduce every table and figure panel computed in the paper
**offline, without API keys and without re-calling any model**.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/reproduce_all.sh      # or run scripts/01_*.py ... 04_*.py one by one
pytest                             # checks the reproduced values against the paper
```

Outputs are written to `results/tables/` (CSV and Markdown) and `results/figures/` (PNG and PDF).
The committed `results/` folder holds the outputs of this run. Tested with Python 3.13 on Linux
(numpy 2.5, pandas 2.3, scipy 1.18, matplotlib 3.11, seaborn 0.13, scikit-learn 1.9).

## What reproduces what

Table numbers follow the revised manuscript.

| Paper item | Script | Input data | Output |
|---|---|---|---|
| Table 3 (label distribution) | `scripts/01_screening_tables.py` | `data/screening/confusion_matrices.csv` | `results/tables/table3_label_distribution.*` |
| Table 4 (screening agreement, 95% CI), Table S2 (per-class metrics) | `scripts/01_screening_tables.py` | same | `table4_screening_performance.*`, `tableS2_screening_per_class.*` |
| Figure 2b–c (screening metrics, confusion matrices) | `scripts/01_screening_tables.py` | same | `results/figures/fig2bc_screening.*` |
| Table 5 (extraction agreement by field), Table S3 (all metrics) | `scripts/02_extraction_tables.py` | `data/extraction/insights_extraction_detailed.csv` | `table5_extraction_performance.*`, `tableS3_extraction_all_metrics.*` |
| Figure 2d (extraction scores) | `scripts/02_extraction_tables.py` | same | `results/figures/fig2d_extraction.*` |
| Table S1 (matched vs mismatched BERTScore F1) | `scripts/03_bertscore_baseline.py` | `data/bertscore_baseline/pair_scores.csv` | `tableS1_bertscore_baseline.*` |
| Table 8 (four-model comparison, mean ± SD over three runs) | `scripts/04_model_comparison_tables.py` | `data/model_comparison/` (2,028 stored API responses) | `table8_model_comparison.*`, `results/model_comparison/` |

Scripts 02–04 check their results against the saved outputs of the original analysis. The tests in
`tests/test_paper_values.py` assert the values reported in the paper, for example accuracy 86.00% and
59.70%, macro-F1 0.505 and 0.464, retention 602/881 and 219/873, and mean agreement 56.8%, 53.6%,
49.6% and 45.9% in the four-model comparison.

Processing times, API costs and refusal counts of the original deployment (Tables 6–7) were taken
from the run logs and the provider's billing records. They are reported in the paper and are not
recomputed here.

### Uncertainty

- **Full-corpus evaluation (Tables 3–5, Figure 2).** Each record was processed once by the model, so
  run-to-run variability cannot be estimated. Screening metrics carry 95% percentile bootstrap
  confidence intervals (10,000 resamples of records, seed 20261123). Extraction metrics are reported
  as mean ± SD across papers with t-based 95% confidence intervals.
- **Four-model comparison (Table 8).** Each paper was screened three times per model. Agreement is
  reported as mean ± SD (and range) across the three repetitions. The table note gives the SD across
  the three 52-paper subsets.
- **BERTScore baseline (Table S1).** SD and range across the 10 random reassignments.

## Models and settings

| Analysis | Model (requested identifier) | Access | Settings |
|---|---|---|---|
| Full-corpus evaluation (LLM.IS, LLM.DS, LLM.IE), Feb–Jun 2025 | Qwen Max (`qwen-max`, the stable alias of `qwen-max-2025-01-25`, Qwen2.5-Max) | Alibaba Cloud Model Studio, `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | temperature 0.6, top_p 1.0, max_tokens 4,096; one fixed prompt per task; full texts limited to the first 10,000 words, then cleaned (references, URLs and empty lines removed) |
| Four-model comparison, Sep 2026 | Qwen3-Max (`qwen/qwen3-max`) | OpenRouter, provider Alibaba | temperature 0.6, max_tokens 4,096, 3 repetitions, provider fallback disabled |
| | DeepSeek V3.2 (`deepseek/deepseek-v3.2`) | OpenRouter, provider AtlasCloud (FP8) | as above; reasoning disabled |
| | GPT-4.1-mini (`openai/gpt-4.1-mini`) | OpenRouter, provider OpenAI | as above |
| | Claude Haiku 4.5 (`anthropic/claude-haiku-4.5`) | OpenRouter, provider Anthropic | as above; reasoning disabled |
| Extraction similarity | `all-mpnet-base-v2` (sentence-transformers) for cosine similarity; BERTScore 0.3.13 with `roberta-large`, layer 17, no IDF weighting, no baseline rescaling | local | pairwise set matching: one-to-one assignment, BERT-F1 threshold 0.70 |

The model identifiers and request settings stored with every call are in
`data/model_comparison/run/protocol.json` and `data/model_comparison/run/calls/*.json`.

## Repository layout

```text
prompts/            Web of Science query and the screening/extraction prompts (Supplementary Notes 1–4)
Libs/               workflow code used for the full-corpus evaluation
  LLM/LLM.py          OpenAI-compatible client (default parameters)
  LR/                 InitialScreening (LLM.IS), DeepScreening (LLM.DS), InsightsExtraction (LLM.IE),
                      LLMInsightsEvaluator (extraction metrics and plots)
  Zotero/             Zotero access and PDF-to-text processing
model_comparison/   supplementary four-model comparison: sampling, API runner, summaries, tests
notebooks/          bertscore_baseline.ipynb: full recomputation of the BERTScore baseline
scripts/            table and figure scripts (offline), run_pipeline.py (original workflow)
data/               stored outputs used by the scripts (see data/README.md)
results/            regenerated tables and figures
tests/              regression tests against the reported values
```

## Re-running the model calls (optional)

Re-running hosted models is **not** needed to reproduce the results, and responses may differ because
generation is stochastic and hosted deployments change. The code is provided so that the workflow can
be audited and reused.

1. `pip install -r requirements-full.txt`
2. Copy `.env.example` to `.env`, fill in your own keys and export them. Keys are read from
   environment variables only.
3. Full-corpus workflow (requires a Zotero library holding the records, with PDFs for the full-text
   stages):
   ```bash
   python scripts/run_pipeline.py --stage is --collection <ZOTERO_COLLECTION_KEY>
   python scripts/run_pipeline.py --stage ds --collection <ZOTERO_COLLECTION_KEY>
   python scripts/run_pipeline.py --stage ie --collection <ZOTERO_COLLECTION_KEY>
   ```
   Labels are written to Zotero item fields: LLM.IS to `archiveLocation`, LLM.DS to `callNumber`.
   The human reference labels were stored in `rights`.
4. Four-model comparison (`model_comparison/revision_experiment.py`):
   - `prepare` freezes the text of each sampled paper;
   - `run --execute` sends the requests (dry run without `--execute`);
   - `summarize` rebuilds the summaries from the stored responses.

   The balanced sample was drawn with `collection_sampling.sample_records` (13 papers per class in
   each of 3 disjoint subsets, seed 20260917, human labels from the `rights` field) from the
   2,102-record deep-screening collection, plus 11 deliberately selected papers. The selection is in
   `data/model_comparison/subset_membership.csv` and `run/input_manifest.json`. Run
   `python model_comparison/revision_experiment.py --help` for the options.
5. BERTScore baseline: open `notebooks/bertscore_baseline.ipynb` and run all cells. The model is
   downloaded once; no API calls are made. Behind a network proxy, recent NLTK versions refuse to
   download the WordNet data unless `NLTK_ALLOW_PROXIED_URLOPEN=1` is set.

## Data not included

- **Full texts and PDFs** of the screened articles, and text extracted from them: copyright.
- **Web of Science records** (abstracts, keywords and other licensed metadata): database licence.
  The search query is in `prompts/wos_query.txt`.
- **Per-record screening labels:** these are stored in the authors' Zotero library. The three
  confusion matrices in `data/screening/` are sufficient statistics for every screening metric
  reported in the paper.
- **Run logs of the original deployment** (processing time, cost and refusal messages per call).

## License

- Code: MIT License (`LICENSE`).
- Data in `data/` and outputs in `results/`: Creative Commons Attribution 4.0 International
  (`LICENSE-DATA`).

## Citation

If you use this code or data, please cite the article above and this repository (see `CITATION.cff`).
