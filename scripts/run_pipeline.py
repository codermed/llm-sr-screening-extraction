"""Run the original Qwen Max screening and extraction workflow (LLM.IS, LLM.DS, LLM.IE).

This script reproduces the sequence of calls used for the full-corpus evaluation
(February-June 2025). It is provided for transparency. Running it requires:

* a Zotero web library holding the records (title, abstract and PDF attachments),
* a local Zotero storage folder with the PDFs (``ZOTERO_STORAGE_DIR``),
* API keys in environment variables (see ``.env.example``).

Re-running the hosted model will not necessarily return the same responses,
because generation is stochastic (temperature 0.6) and hosted deployments change.
The analyses reported in the paper are reproduced offline by the other scripts in
this folder, from the stored outputs in ``data/``.

Labels are written back to Zotero item fields, as in the original runs:
LLM.IS -> ``archiveLocation``; LLM.DS -> ``callNumber``. The human reference
labels were stored in the ``rights`` field.

Examples
--------
python scripts/run_pipeline.py --stage is --collection <COLLECTION_KEY>
python scripts/run_pipeline.py --stage ds --collection <COLLECTION_KEY>
python scripts/run_pipeline.py --stage ie --collection <COLLECTION_KEY>
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Libs.Zotero.Zotero import Zotero  # noqa: E402
from Libs.Zotero.ZoteroCollection import ZoteroCollection  # noqa: E402
from Libs.LLM.LLM import LLM  # noqa: E402
from Libs.LR.LiteratureReview import LiteratureReview  # noqa: E402
from Libs.LR.InitialScreening import InitialScreening  # noqa: E402
from Libs.LR.DeepScreening import DeepScreening  # noqa: E402
from Libs.LR.InsightsExtraction import InsightsExtraction  # noqa: E402

PROMPT_FILE = ROOT / 'prompts' / 'prompts_NationalLULC.txt'

# Model configuration used for the full-corpus evaluation.
# 'qwen-max' was the stable alias of qwen-max-2025-01-25 (Qwen2.5-Max) at the time of use.
DEFAULT_MODEL = 'qwen-max'
DEFAULT_BASE_URL = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
PARAMS = {'temperature': 0.6}  # other parameters: Libs/LLM/LLM.py defaults

PDF_SETTINGS = dict(
    kwargs_pdf_extract=dict(method='markdown', chunk_size=None),
    kwargs_pdf_truncate=dict(word_count=10000),
    kwargs_pdf_clean=dict(remove_refs=True, remove_urls=True, remove_empty_lines=True),
    kwargs_pdf_save=dict(output_dir=None, format='md', save_raw=True, save_truncated=True,
                         save_cleaned=True, raw_suffix='raw', truncated_suffix='trn',
                         cleaned_suffix='cln'),
    use_pdf_version='cleaned',
)


def require_env(name):
    value = os.environ.get(name)
    if not value:
        sys.exit(f'Environment variable {name} is not set (see .env.example).')
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['is', 'ds', 'ie'],
                    help='is = title-abstract screening; ds = full-text screening; ie = insight extraction')
    ap.add_argument('--collection', required=True, help='Zotero collection key holding the records')
    ap.add_argument('--project-id', default='NationalLULC')
    ap.add_argument('--output-dir', default=str(ROOT / 'output'))
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--base-url', default=DEFAULT_BASE_URL)
    ap.add_argument('--no-zotero-update', action='store_true',
                    help='do not write screening labels back to Zotero')
    args = ap.parse_args()

    zotero = Zotero(api_key=require_env('ZOTERO_API_KEY'), user_id=require_env('ZOTERO_USER_ID'))
    # Full-text stages need the PDF attachments.
    collection = ZoteroCollection(zotero, collection_key=args.collection, item_types=['journalArticle'],
                                  attachments=args.stage in ('ds', 'ie'))
    llm = LLM(model=args.model, use_api=True, api_key=require_env('QWEN_API_KEY'), base_url=args.base_url)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    review = LiteratureReview(project_id=args.project_id, llm=llm, zotero_collection=collection,
                              output_dir=args.output_dir, prompt_file=str(PROMPT_FILE))
    update = not args.no_zotero_update

    if args.stage == 'is':
        InitialScreening(review).run(
            prompt_key='initial_screening_prompt',
            categories=['EXCLUDED', 'OK', 'POSSIBLE', 'REVIEW', 'UNDECIDED'],
            params=PARAMS, update_zotero=update, zotero_field='archiveLocation',
            rewrite=False, resume=True)
    elif args.stage == 'ds':
        DeepScreening(review).run(
            prompt_key='deep_screening_prompt',
            categories=['EXCLUDED', 'OK', 'POSSIBLE', 'REVIEW'],
            params=PARAMS, update_zotero=update, zotero_field='callNumber',
            rewrite=True, resume=True, **PDF_SETTINGS)
    else:
        InsightsExtraction(review).run(
            prompt_key='insights_extraction_prompt', params=PARAMS,
            rewrite=True, resume=True, **PDF_SETTINGS)


if __name__ == '__main__':
    main()
