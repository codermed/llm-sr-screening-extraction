**Table 4. Screening agreement with the reference labels (estimate, 95% CI)**

| Metric | HUM–LLM.IS | HUM–LLM.DS | LLM.IS–LLM.DS |
|---|---|---|---|
| Records compared, n | 11,678 | 2,077 | 2,069 |
| Overall accuracy, % | 86.0 (85.4–86.6) | 59.7 (57.6–61.8) | 25.3 (23.4–27.2) |
| Macro-averaged precision | 0.519 (0.487–0.550) | 0.494 (0.454–0.533) | 0.522 (0.481–0.562) |
| Macro-averaged recall | 0.620 (0.593–0.647) | 0.498 (0.469–0.525) | 0.472 (0.439–0.505) |
| Macro-averaged F1 | 0.505 (0.478–0.530) | 0.464 (0.435–0.492) | 0.341 (0.310–0.371) |
| Recall, EXCLUDED, % | 89.0 (88.4–89.6) | 89.4 (87.6–91.1) | 95.1 (92.6–97.4) |
| Recall, POSSIBLE, % | 45.7 (41.8–49.6) | 3.3 (2.0–4.8) | 4.9 (3.8–6.2) |
| Recall, OK, % | 81.4 (74.8–87.7) | 75.7 (68.5–82.7) | 28.2 (24.2–32.3) |
| Recall, REVIEW, % | 31.7 (23.6–40.0) | 30.6 (22.5–38.9) | 60.7 (48.2–72.7) |
| Retention of reference non-EXCLUDED records, % | 68.3 (65.3–71.4); 602/881 | 25.1 (22.3–28.0); 219/873 | — |
| Retention of reference OK records, % | 95.7 (92.1–98.6); 134/140 | 90.7 (85.7–95.3); 127/140 | — |
| Potential screening-volume reduction, % | 84.7 (84.0–85.3); 9,887/11,678 | 83.3 (81.7–84.9); 1,730/2,077 | — |
| Reference non-EXCLUDED records excluded by the model, n (of which OK) | 279 (6) | 654 (13) | — |

The first label set in each comparison is the reference. 95% CIs: percentile bootstrap over records (10,000 resamples, seed 20261123). The full-corpus evaluation used one model call per record, so no run-to-run variability is available. Retention groups OK, POSSIBLE and REVIEW as retained; volume reduction assumes that every model exclusion is accepted.
