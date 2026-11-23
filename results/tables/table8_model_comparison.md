**Table 8. Screening agreement and repeatability in the balanced random sample of 156 papers**

| Model (requested ID; provider) | Agreement, mean ± SD (range), % | Weighted agreement, mean ± SD, % | Macro-F1, mean ± SD | Same label in all three calls, n (%) | Automatically parsed calls, n/468 |
|---|---|---|---|---|---|
| Qwen3-Max (qwen/qwen3-max; Alibaba) | 56.8 ± 0.4 (56.4–57.1) | 60.3 ± 0.3 | 0.563 ± 0.005 | 148 (94.9) | 468/468 |
| Claude Haiku 4.5 (anthropic/claude-haiku-4.5; Anthropic) | 53.6 ± 1.3 (52.6–55.1) | 64.5 ± 2.0 | 0.467 ± 0.012 | 136 (87.2) | 363/468 |
| GPT-4.1-mini (openai/gpt-4.1-mini; OpenAI) | 49.6 ± 1.0 (48.7–50.6) | 61.0 ± 0.8 | 0.404 ± 0.015 | 152 (97.4) | 468/468 |
| DeepSeek V3.2 (deepseek/deepseek-v3.2; AtlasCloud (FP8)) | 45.9 ± 2.4 (44.2–48.7) | 61.0 ± 1.8 | 0.379 ± 0.022 | 135 (86.5) | 468/468 |

Each model screened the same 156 randomly selected papers (39 per class) three times (468 calls per model). Agreement: exact four-category agreement with the human reference; mean, sample SD and range across the three repetitions (run-to-run variability). Weighted agreement uses the class distribution of the eligible collection. Macro-F1: four-class macro-averaged F1 per repetition. Agreement and repeatability include 105 Claude Haiku 4.5 labels recovered by documented manual inspection; automatically parsed calls are reported separately. SD of mean agreement across the three 52-paper subsets (percentage points): Qwen3-Max 3.9, Claude Haiku 4.5 2.7, GPT-4.1-mini 2.1, DeepSeek V3.2 3.2.
