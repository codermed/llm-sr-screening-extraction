# Data

All files are licensed under CC BY 4.0 (see `../LICENSE-DATA`).

## screening/

`confusion_matrices.csv`: the three 4×4 confusion matrices of the full-corpus screening evaluation, one
row per cell. Rows are reference labels and columns are predicted labels.

| Comparison | Records |
|---|---|
| HUM vs LLM.IS | 11,678 |
| HUM vs LLM.DS | 2,077 |
| LLM.IS vs LLM.DS | 2,069 |

- HUM: final single-assessor human classifications.
- LLM.IS: Qwen Max title–abstract screening.
- LLM.DS: Qwen Max full-text screening.
- Records are included when both labels are recognized as EXCLUDED, OK, POSSIBLE or REVIEW.

The matrices were transcribed programmatically from the stored output of the original evaluation
notebook, which printed the scikit-learn confusion matrices together with the overall accuracies.

## extraction/

- `human_extraction.csv`, `llm_extraction.csv`: the human and LLM extraction tables for the 345
  attempted papers (identical to Supplementary Data 1). Rows of the two files refer to the same paper
  (same DOI).
- `excluded_papers.csv`: the three papers excluded from the extraction metrics. Two are API refusals
  (Zotero keys `MEE88MWP` and `4VZ2CD2X`); one produced an unusable output (`SR52IV4U`). This leaves
  342 analysed papers.
- `insights_extraction_detailed.csv`: per-paper scores of the 342 papers, one row per paper × field,
  with one column per metric.
  - `Paper_Index` is the row position in the original evaluation tables, which were kept in Google
    Sheets. These tables also held the `Global` and `Income Group` columns.
  - Computed by `Libs/LR/LLMInsightsEvaluator.py`.
- `insights_extraction_summary.csv`, `insights_cross_columns_metrics.csv`: the summary statistics
  saved by the original evaluation. They are used as a cross-check.

## bertscore_baseline/

Outputs of the mismatched-paper BERTScore baseline (Supplementary Note 6), produced by
`notebooks/bertscore_baseline.ipynb` from the files in `extraction/`:

- `pair_scores.csv`: scores of every matched and mismatched pair (10 reassignments).
- `baseline_summary.csv`, `baseline_by_reassignment.csv`: field-level results.
- `field_eligibility_audit.csv`: denominators per field.
- `historical_method_check.csv`: recomputed versus saved original means.
- `missing_value_rules.json`: placeholder values treated as missing.
- `run_manifest.json`: settings, package versions and model revision.

## model_comparison/

Stored data of the supplementary four-model comparison:

- `run/protocol.json`: model identifiers, providers, request settings and prompts.
- `run/calls/*.json`: one file per API call, 2,028 in total: 169 paper–task combinations × 4 models
  × 3 repetitions. Each file holds the full model response, parsed label, token usage, reported cost,
  returned model and provider, and timing. Prompts and paper texts are not stored.
- `run/input_manifest.json`: the sampled papers, with identifiers, DOI, title, the stored human and
  original LLM labels, sampling stratum, subset and weight.
- `manual_label_review.csv`: the documented manual recovery of labels from 120 Claude Haiku 4.5
  responses whose final decision the parser did not recognize.
- `subset_membership.csv`, `label_mapping.csv`: sample membership, and the mapping of raw human labels
  to the four classes.

### Changes made for publication

The following were removed, with no effect on any reported value:

- `run/input_manifest.json`:
  - local file paths;
  - Zotero library, collection and item-database identifiers, and the collection folder path;
  - raw label strings duplicated in other fields;
  - internal priority tags.

  In one row, a free-text note in `forced_reason` was replaced by the standard value
  "User-selected forced DS case".
- `manual_label_review.csv`: the `response_text` column, because the full responses are in
  `run/calls/`. Windows path separators were normalized.
- `bertscore_baseline/run_manifest.json`: the operating-system string.

`scripts/04_model_comparison_tables.py` reruns the summaries from these files. The summary tables it
produces are identical to those of the original analysis.
