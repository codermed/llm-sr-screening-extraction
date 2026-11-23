**Table 5. Agreement between LLM and human extraction by field**

| Field type | Field | Metric | n | Mean ± SD | 95% CI | Median [IQR] |
|---|---|---|---|---|---|---|
| Categorical | Country | Exact match | 342 | 0.77 ± 0.42 | 0.73–0.82 | 1.00 [1.00–1.00] |
| Categorical | Income Group | Exact match | 342 | 0.86 ± 0.35 | 0.82–0.89 | 1.00 [1.00–1.00] |
| Categorical | Global | Exact match | 342 | 0.89 ± 0.31 | 0.86–0.92 | 1.00 [1.00–1.00] |
| Boolean | Population as Predictor? | Exact match | 342 | 0.78 ± 0.41 | 0.74–0.82 | 1.00 [1.00–1.00] |
| Boolean | Validation? | Exact match | 342 | 0.85 ± 0.36 | 0.81–0.89 | 1.00 [1.00–1.00] |
| Temporal | Baseline Years | Precision | 342 | 0.86 ± 0.31 | 0.83–0.90 | 1.00 [1.00–1.00] |
| Temporal | Baseline Years | Recall | 342 | 0.81 ± 0.35 | 0.77–0.84 | 1.00 [0.75–1.00] |
| Temporal | Baseline Years | F1 | 342 | 0.78 ± 0.35 | 0.74–0.81 | 1.00 [0.67–1.00] |
| Temporal | Target Years | Precision | 342 | 0.92 ± 0.25 | 0.90–0.95 | 1.00 [1.00–1.00] |
| Temporal | Target Years | Recall | 342 | 0.87 ± 0.30 | 0.84–0.91 | 1.00 [1.00–1.00] |
| Temporal | Target Years | F1 | 342 | 0.85 ± 0.31 | 0.82–0.88 | 1.00 [1.00–1.00] |
| Set-based | LULC Classes | BERT-F1 (whole set) | 342 | 0.94 ± 0.06 | 0.93–0.95 | 0.95 [0.90–1.00] |
| Set-based | LULC Classes | BERT-F1 (pairwise) | 342 | 0.89 ± 0.18 | 0.87–0.91 | 1.00 [0.85–1.00] |
| Set-based | Scenarios | BERT-F1 (whole set) | 342 | 0.92 ± 0.07 | 0.92–0.93 | 0.92 [0.87–1.00] |
| Set-based | Scenarios | BERT-F1 (pairwise) | 342 | 0.91 ± 0.16 | 0.89–0.93 | 1.00 [0.86–1.00] |
| Set-based | Analysis Methods | BERT-F1 (whole set) | 342 | 0.89 ± 0.03 | 0.89–0.89 | 0.89 [0.87–0.91] |
| Set-based | Analysis Methods | BERT-F1 (pairwise) | 342 | 0.80 ± 0.16 | 0.79–0.82 | 0.83 [0.67–0.91] |
| Set-based | Predictors | BERT-F1 (whole set) | 342 | 0.89 ± 0.05 | 0.89–0.90 | 0.90 [0.86–0.93] |
| Set-based | Predictors | BERT-F1 (pairwise) | 342 | 0.83 ± 0.20 | 0.81–0.85 | 0.89 [0.75–1.00] |
| Set-based | Data Sources | BERT-F1 (whole set) | 342 | 0.88 ± 0.04 | 0.87–0.88 | 0.88 [0.85–0.90] |
| Set-based | Data Sources | BERT-F1 (pairwise) | 342 | 0.83 ± 0.15 | 0.81–0.85 | 0.86 [0.75–0.93] |
| Set-based | Tools/Software Used | BERT-F1 (whole set) | 342 | 0.90 ± 0.06 | 0.90–0.91 | 0.90 [0.86–0.94] |
| Set-based | Tools/Software Used | BERT-F1 (pairwise) | 342 | 0.79 ± 0.21 | 0.76–0.81 | 0.80 [0.67–1.00] |
| Set-based | Validation Methods | BERT-F1 (whole set) | 342 | 0.90 ± 0.07 | 0.89–0.90 | 0.89 [0.85–0.94] |
| Set-based | Validation Methods | BERT-F1 (pairwise) | 342 | 0.81 ± 0.21 | 0.79–0.83 | 0.86 [0.67–1.00] |
| Set-based | Validation Data Sources | BERT-F1 (whole set) | 342 | 0.90 ± 0.07 | 0.89–0.91 | 0.89 [0.84–0.95] |
| Set-based | Validation Data Sources | BERT-F1 (pairwise) | 342 | 0.84 ± 0.19 | 0.82–0.86 | 1.00 [0.67–1.00] |
| Narrative | Objective | BERT-F1 | 342 | 0.92 ± 0.03 | 0.91–0.92 | 0.92 [0.89–0.94] |
| Narrative | Objective | Cosine similarity | 342 | 0.87 ± 0.10 | 0.86–0.88 | 0.88 [0.81–0.94] |
| Narrative | Simulation Method Advantages | BERT-F1 | 342 | 0.87 ± 0.04 | 0.86–0.87 | 0.87 [0.85–0.89] |
| Narrative | Simulation Method Advantages | Cosine similarity | 342 | 0.61 ± 0.22 | 0.58–0.63 | 0.65 [0.49–0.76] |
| Narrative | Key Findings | BERT-F1 | 342 | 0.87 ± 0.03 | 0.86–0.87 | 0.86 [0.85–0.88] |
| Narrative | Key Findings | Cosine similarity | 342 | 0.81 ± 0.10 | 0.80–0.82 | 0.82 [0.77–0.89] |
| Narrative | Implications | BERT-F1 | 342 | 0.87 ± 0.02 | 0.87–0.87 | 0.87 [0.86–0.89] |
| Narrative | Implications | Cosine similarity | 342 | 0.74 ± 0.11 | 0.73–0.75 | 0.76 [0.67–0.82] |
| Narrative | Limitations | BERT-F1 | 342 | 0.87 ± 0.03 | 0.86–0.87 | 0.86 [0.84–0.88] |
| Narrative | Limitations | Cosine similarity | 342 | 0.66 ± 0.20 | 0.64–0.68 | 0.70 [0.56–0.79] |
| Narrative | Future Directions | BERT-F1 | 342 | 0.87 ± 0.04 | 0.87–0.88 | 0.87 [0.85–0.89] |
| Narrative | Future Directions | Cosine similarity | 342 | 0.65 ± 0.22 | 0.63–0.68 | 0.70 [0.58–0.79] |
| Narrative | Scalability | BERT-F1 | 342 | 0.87 ± 0.05 | 0.87–0.88 | 0.87 [0.84–0.89] |
| Narrative | Scalability | Cosine similarity | 342 | 0.51 ± 0.27 | 0.48–0.54 | 0.53 [0.33–0.68] |
| Narrative | Novelty | BERT-F1 | 342 | 0.88 ± 0.02 | 0.87–0.88 | 0.88 [0.86–0.89] |
| Narrative | Novelty | Cosine similarity | 342 | 0.74 ± 0.12 | 0.73–0.75 | 0.76 [0.68–0.82] |

n = 342 analysed papers (345 attempted; two refusals and one unusable output excluded). SD: spread across papers. 95% CI: t-based interval for the mean. The extraction was run once per paper, so run-to-run variability is not available. Comparisons with two empty sets score 1 and with one empty set score 0.
