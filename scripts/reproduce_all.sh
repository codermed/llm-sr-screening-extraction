#!/usr/bin/env bash
# Regenerate every table and figure in results/ from the stored data (offline, about 15 s).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
"$PY" scripts/01_screening_tables.py
"$PY" scripts/02_extraction_tables.py
"$PY" scripts/03_bertscore_baseline.py
"$PY" scripts/04_model_comparison_tables.py
echo "Done. Tables: results/tables/  Figures: results/figures/"
