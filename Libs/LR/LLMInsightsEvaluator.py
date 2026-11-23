"""Evaluation of LLM-extracted insights against a human reference.

Used for the extraction results (per-paper scores, summary statistics and the
extraction panel of Figure 2) and for the screening confusion matrices.

The text-similarity models (sentence-transformers ``all-mpnet-base-v2`` and
BERTScore with ``roberta-large``) and NLTK resources are only needed when
metrics are recomputed from the extraction tables. Loading the cached results
with ``LLMInsightsEvaluator.from_cache_csv`` and plotting them does not need
these packages.
"""
import pandas as pd
import numpy as np
import re

# Optional dependencies: required only to recompute text-similarity metrics.
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
except ImportError:  # pragma: no cover
    nltk = WordNetLemmatizer = stopwords = None
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None
try:
    from bert_score import BERTScorer
except ImportError:  # pragma: no cover
    BERTScorer = None
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import jaccard_score
except ImportError:  # pragma: no cover
    cosine_similarity = jaccard_score = None
try:
    import textdistance
except ImportError:  # pragma: no cover
    textdistance = None
import warnings
import unicodedata
from typing import Iterable, Optional
from scipy.optimize import linear_sum_assignment
# ----------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator


def _ensure_nltk_data():
    """Download the NLTK resources used for text preprocessing (first use only)."""
    if nltk is None:
        raise ImportError("nltk is required to recompute extraction metrics "
                          "(pip install -r requirements-full.txt)")
    for resource, path in (('wordnet', 'corpora/wordnet'), ('stopwords', 'corpora/stopwords')):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


class LLMInsightsEvaluator:
    def __init__(self, df_hum, df_llm, synonym_map=None, 
                 tversky_alpha: float = 0.5, tversky_beta: float = 0.5,
                 # ---  knobs for semantic set comparison ---
                 set_use_semantics: bool = True,
                 set_semantic_mode: str = "both",  # "none" | "whole" | "pairwise" | "both"
                 set_pair_threshold: float = 0.60,  # minimum similarity to count as a "match"
                 set_pair_metric: str = "cosine",  # "cosine" (SentenceTransformer) or "bert_f1"
                 set_joiner: str = "; ",           # how to join items for whole-text similarity
                 set_separators: tuple = (",",";","/"," and "," & ","、","，","|"),
                 temporal_tolerance: int = 0       # NEW: soft matching: +/- years as equal (0 = exact)
                ):
        """
        Initialize the evaluator with human and LLM extracted data
        
        Parameters:
        df_hum (pd.DataFrame): Human-extracted dataframe
        df_llm (pd.DataFrame): LLM-extracted dataframe
        synonym_map (dict): Optional dictionary for normalizing categorical terms
        """
        self.df_hum = df_hum
        self.df_llm = df_llm
        self.synonym_map = synonym_map or {}
        self.dict_results = {} 
        self.dict_detailed = {} 
        self.df_detailed = None
        self.df_summary = None
        self.cross_column_metrics = {}
        
        # ['Country', 'Continent', 'Region', 'Income Group', 'Global', 
        #  'Objective', 'Major Climate', 'Major Biome', 'Major Urban Trend', 'LULC Classes', 
        #  'Baseline Years', 'Target Year(s)', 'Scenarios', 
        #  'Analysis Methods', 'Simulation Method Advantage(s)', 'Tools/Soft Used', 'Predictors', 
        #  'Population?', 'Population? (short)', 'Data Sources', 
        #  'Validation?', 'Validation Methods', 'Validation Data Sources', 
        #  'Key Findings', 'Implications', 'Limitations', 'Future Directions', 'Scalability', 'Novelty']
        
        # Define column types and metrics
        self.column_types = {
            'set_based': [
                #'Major Climate', 'Major Biome', 'Major Urban Trend', 
                'LULC Classes', 'Scenarios', 'Analysis Methods', 'Predictors', 'Data Sources', 
                'Tools/Soft Used', 'Validation Methods', 'Validation Data Sources'
            ],
            'free_text': [
                'Objective', 'Sim. Method Advantages',  #'Population?',
                'Key Findings', 'Implications', 'Limitations', 'Future Directions', 'Scalability', 'Novelty'
            ],
            'categorical': [
                'Country', 'Continent', 'Region', 'Income Group', 'Global'
            ],
            'temporal': ['Baseline Years', 'Target Years'],
            'boolean': ['Population as Predictor?', 'Validation?']
        }
        
        # Create a set of all valid columns from column_types
        self.valid_columns = set()
        for cols in self.column_types.values():
            self.valid_columns.update(cols)
        
        # Initialize NLP tools
        _ensure_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.st_model = SentenceTransformer('all-mpnet-base-v2')
        self.bert_scorer = BERTScorer(lang='en')
        
        # --- set/temporal similarity knobs ---
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta  = float(tversky_beta)
        
        # --- 
        self.set_use_semantics   = bool(set_use_semantics)
        self.set_semantic_mode   = str(set_semantic_mode).lower()
        self.set_pair_threshold  = float(set_pair_threshold)
        self.set_pair_metric     = set_pair_metric
        self.set_joiner          = set_joiner
        self.set_separators      = tuple(set_separators)
        self.temporal_tolerance  = int(temporal_tolerance)
        
        # tiny embedding cache to avoid recompute
        self._embed_cache = {}

    # ---------- optional: lazy model init if you ever need to compute new text metrics ----------
    def _init_models(self):
        if getattr(self, "lemmatizer", None) is None or getattr(self, "stop_words", None) is None:
            _ensure_nltk_data()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        if getattr(self, "st_model", None) is None:
            self.st_model = SentenceTransformer('all-mpnet-base-v2')
        if getattr(self, "bert_scorer", None) is None:
            self.bert_scorer = BERTScorer(lang='en')

    # ---------- build from cached CSVs ----------
    @classmethod
    def from_cache_csv(cls,
                       detailed_csv: str,
                       summary_csv: str,
                       cross_csv: str | None = None,
                       df_hum: pd.DataFrame | None = None,
                       df_llm: pd.DataFrame | None = None,
                       synonym_map: dict | None = None):
        df_detailed = pd.read_csv(detailed_csv)
        df_summary  = pd.read_csv(summary_csv)
        df_cross    = pd.read_csv(cross_csv) if cross_csv else None
        return cls.from_cache_df(df_detailed, df_summary, df_cross,
                                 df_hum=df_hum, df_llm=df_llm, synonym_map=synonym_map)

    # ---------- build from cached DataFrames (already loaded) ----------
    @classmethod
    def from_cache_df(cls,
                      df_detailed: pd.DataFrame,
                      df_summary: pd.DataFrame,
                      df_cross: pd.DataFrame | None = None,
                      df_hum: pd.DataFrame | None = None,
                      df_llm: pd.DataFrame | None = None,
                      synonym_map: dict | None = None):
        # bypass __init__ to avoid loading heavy models
        obj = cls.__new__(cls)

        # minimal core state
        obj.df_hum = df_hum if df_hum is not None else pd.DataFrame()
        obj.df_llm = df_llm if df_llm is not None else pd.DataFrame()
        obj.synonym_map = synonym_map or {}

        # same column_types and valid_columns as your __init__
        obj.column_types = {
            'set_based': [
                'Major Climate', 'Major Biome', 'Major Urban Trend', 
                'LULC Classes', 'Scenarios', 'Analysis Methods', 'Predictors', 'Data Sources', 
                'Tools/Soft Used', 'Validation Methods', 'Validation Data Sources'
            ],
            'free_text': [
                'Objective', 'Sim. Method Advantages', 'Population?',
                'Key Findings', 'Implications', 'Limitations', 'Future Directions', 'Scalability', 'Novelty'
            ],
            'categorical': [
                'Country', 'Continent', 'Region', 'Income Group', 'Global'
            ],
            'temporal': ['Baseline Years', 'Target Years'],
            'boolean': ['Population as Predictor?', 'Validation?']
        }
        obj.valid_columns = set()
        for _cols in obj.column_types.values():
            obj.valid_columns.update(_cols)

        # NLP/light knobs; models stay None until/if needed
        # NLP helpers and models are loaded by _init_models() only if new metrics are computed
        obj.lemmatizer = None
        obj.stop_words = None
        obj.st_model = None
        obj.bert_scorer = None
        obj.tversky_alpha = 0.5
        obj.tversky_beta  = 0.5

        # cached tables
        obj.df_detailed = df_detailed.copy()
        obj.df_summary  = df_summary.copy()
        obj.cross_column_metrics = obj._parse_cross_df(df_cross) if df_cross is not None else {}

        # derived dicts
        obj.dict_results = {}
        obj.dict_detailed = {}

        # rebuild dict_results so downstream code works
        obj._rebuild_dict_results_from_cache()

        return obj

    # ---------- helpers to parse cross-metrics and rebuild dict_results ----------
    @staticmethod
    def _parse_cross_df(df_cross: pd.DataFrame) -> dict:
        if df_cross is None or df_cross.empty:
            return {}
        out = {}

        # try common shapes
        # shape A: columns like ['col_type','mean_performance','range_min','range_max',...]
        if {'col_type','mean_performance'}.issubset(df_cross.columns):
            for _, r in df_cross.iterrows():
                out[str(r['col_type'])] = {
                    'mean_performance': float(r.get('mean_performance', float('nan'))),
                    'performance_range': (float(r.get('range_min', float('nan'))),
                                          float(r.get('range_max', float('nan')))),
                    'most_reliable_column': r.get('most_reliable_column', 'N/A'),
                    'least_reliable_column': r.get('least_reliable_column', 'N/A')
                }
            return out

        # shape B: pivoted or differently named
        # try to infer by Type column
        if 'Type' in df_cross.columns:
            for typ in df_cross['Type'].dropna().unique():
                sub = df_cross[df_cross['Type'] == typ]
                d = {}
                if 'Mean_Performance' in sub.columns:
                    d['mean_performance'] = float(sub['Mean_Performance'].iloc[0])
                if {'Range_Min','Range_Max'}.issubset(sub.columns):
                    d['performance_range'] = (float(sub['Range_Min'].iloc[0]),
                                              float(sub['Range_Max'].iloc[0]))
                for k in ['most_reliable_column', 'least_reliable_column',
                          'Most_Reliable_Column', 'Least_Reliable_Column']:
                    if k in sub.columns:
                        key = 'most_reliable_column' if 'Most' in k or 'most' in k else 'least_reliable_column'
                        d[key] = sub[k].iloc[0]
                out[str(typ)] = d
            return out

        # fallback: empty
        return out

    def _rebuild_dict_results_from_cache(self):
        if self.df_summary is None or self.df_detailed is None:
            return

        # list of metric columns (try to be robust)
        meta_cols = {
            'Paper_Index','Column','Column_Type',
            'Human_Completeness','LLM_Completeness','Completeness_Gap',
            'Both_Missing','LLM_Hallucination','LLM_Missed'
        }
        # everything numeric in detailed that is not meta is a metric
        metric_cols = [c for c in self.df_detailed.columns if c not in meta_cols]

        eval_dict = {}
        for col in self.df_summary['Column'].dropna().unique():
            sub_s = self.df_summary[self.df_summary['Column'] == col]
            stats = {}
            data_quality = {}

            # take the first non-na data-quality values (they repeat per metric)
            for _, r in sub_s.iterrows():
                # stats for this metric
                m = r['Metric']
                stats[m] = {
                    'mean': r.get('Mean',  np.nan),
                    'std':  r.get('Std_Dev', np.nan),
                    'median': r.get('Median', np.nan),
                    'min': r.get('Min', np.nan),
                    'max': r.get('Max', np.nan),
                    'q1': r.get('Q1', np.nan),
                    'q3': r.get('Q3', np.nan),
                    'confidence_interval_95': (
                        r.get('CI95_Low', np.nan),
                        r.get('CI95_High', np.nan)
                    ),
                    'variance': r.get('Variance', np.nan),
                    'coefficient_variation': r.get('CV', np.nan),
                    'skewness': r.get('Skewness', np.nan),
                    'kurtosis': r.get('Kurtosis', np.nan),
                    'pct_excellent': r.get('Pct_Excellent', np.nan),
                    'pct_good': r.get('Pct_Good', np.nan),
                    'pct_fair': r.get('Pct_Fair', np.nan),
                    'pct_poor': r.get('Pct_Poor', np.nan),
                    'consistency_score': r.get('Consistency_Score', np.nan),
                    'worst_case': r.get('Worst_Case', np.nan),
                    'best_case': r.get('Best_Case', np.nan),
                    'iqr': r.get('IQR', np.nan),
                }
                # capture dq once
                for k_csv, k_int in [
                    ('Human_Completeness','human_completeness'),
                    ('LLM_Completeness','llm_completeness'),
                    ('Completeness_Gap','completeness_gap'),
                    ('Both_Missing','both_missing'),
                    ('LLM_Hallucination','llm_hallucination'),
                    ('LLM_Missed','llm_missed')
                ]:
                    val = r.get(k_csv, np.nan)
                    if k_int not in data_quality and pd.notna(val):
                        data_quality[k_int] = val

            # slice detailed rows for this column
            sub_d = self.df_detailed[self.df_detailed['Column'] == col].copy()

            # raw_results as list of dicts with only metric fields
            raw = []
            if not sub_d.empty:
                only_metrics = sub_d[metric_cols].to_dict(orient='records')
                # keep None instead of NaN in raw dicts
                for rec in only_metrics:
                    raw.append({k: (None if pd.isna(v) else v) for k, v in rec.items()})

            # store
            eval_dict[col] = {
                'raw_results': raw,
                'results_df': sub_d[metric_cols].copy() if not sub_d.empty else pd.DataFrame(columns=metric_cols),
                'stats': stats,
                'data_quality': data_quality
            }

        self.dict_results = eval_dict

    # ---------------------------
    # PRE-CLEANING ENTRY POINT
    # ---------------------------
    def clean_selected_columns(
        self,
        categories: Iterable[str] = ("temporal", "set_based"),
        config: Optional[dict] = None,
        in_place: bool = True,
        columns_override: Optional[dict] = None,
    ):
        """
        Clean selected column categories in self.df_hum / self.df_llm.
    
        Parameters
        ----------
        categories : list[str]
            Any of: 'temporal', 'set_based', 'categorical', 'free_text', 'boolean'.
        config : dict
            Per-category options. All optional. Example:
            {
              "temporal": {
                "strip_parentheses": True,
                "expand_ranges": True,
                "min_year": 1800, "max_year": 2100,
                "join_with": "; ",
                "sort_unique": True
              },
              "set_based": {
                "separators": [",",";","/"," and "," & ","、","，","|"],
                "strip_parentheses": True,
                "lower": True,
                "lemmatize": True,
                "dedupe": True,
                "sort": False,
                "join_with": "; ",
                "synonym_map": None,        # overrides self.synonym_map if given
                "ignore_tokens": []         # optional stop-phrases to drop
              }
            }
        in_place : bool
            If True, mutate self.df_hum / self.df_llm. If False, return cleaned copies.
        columns_override : dict
            Optional mapping of category -> explicit column list to clean,
            e.g., {"temporal": ["Baseline Years","Target Years"]}
    
        Returns
        -------
        If in_place is False: (df_hum_clean, df_llm_clean)
        Else: None
        """
        cfg = self._default_clean_config()
        if config:
            # shallow merge per category
            for k, v in config.items():
                if k in cfg and isinstance(v, dict):
                    cfg[k].update(v)
    
        # ensure we are not writing into a view
        if in_place:
            self.df_hum = self.df_hum.copy(deep=True)
            self.df_llm = self.df_llm.copy(deep=True)
            dfh, dfl = self.df_hum, self.df_llm
        else:
            dfh = self.df_hum.copy(deep=True)
            dfl = self.df_llm.copy(deep=True)
    
        # which cols per category
        def cols_for(cat):
            if columns_override and cat in columns_override:
                return [c for c in columns_override[cat] if c in dfh.columns and c in dfl.columns]
            return [c for c in self.column_types.get(cat, []) if c in dfh.columns and c in dfl.columns]
    
        # clean per category
        for cat in categories:
            if cat == "temporal":
                opts = cfg["temporal"]
                for col in cols_for("temporal"):
                    dfh.loc[:, col] = dfh[col].map(lambda x: self._clean_temporal_cell(x, **opts))
                    dfl.loc[:, col] = dfl[col].map(lambda x: self._clean_temporal_cell(x, **opts))
    
            elif cat == "set_based":
                opts = cfg["set_based"].copy()
                if opts.get("synonym_map") is None:
                    opts["synonym_map"] = self.synonym_map
                for col in cols_for("set_based"):
                    dfh.loc[:, col] = dfh[col].map(lambda x: self._clean_set_cell(x, **opts))
                    dfl.loc[:, col] = dfl[col].map(lambda x: self._clean_set_cell(x, **opts))
    
            elif cat == "categorical":
                for col in cols_for("categorical"):
                    dfh.loc[:, col] = dfh[col].map(self._simple_cat_clean)
                    dfl.loc[:, col] = dfl[col].map(self._simple_cat_clean)
    
            elif cat == "free_text":
                for col in cols_for("free_text"):
                    dfh.loc[:, col] = dfh[col].map(self._normalize_unicode_whitespace)
                    dfl.loc[:, col] = dfl[col].map(self._normalize_unicode_whitespace)
    
            elif cat == "boolean":
                for col in cols_for("boolean"):
                    dfh.loc[:, col] = dfh[col].map(self._clean_boolean_cell)
                    dfl.loc[:, col] = dfl[col].map(self._clean_boolean_cell)
    
        if not in_place:
            return dfh, dfl
        return None

    # ---------------------------
    # DEFAULT CONFIG
    # ---------------------------
    def _default_clean_config(self):
        return {
            "temporal": {
                "strip_parentheses": True,
                "expand_ranges": False,     # set True if you want 1980-1983 -> 1980;1981;1982;1983
                "min_year": 1500,
                "max_year": 2100,
                "join_with": "; ",
                "sort_unique": True
            },
            "set_based": {
                "separators": [",",";","/"," and "," & ","、","，","|"],
                "strip_parentheses": True,
                "lower": True,
                "lemmatize": True,
                "dedupe": True,
                "sort": False,             # keep source order by default
                "join_with": "; ",
                "synonym_map": None,
                "ignore_tokens": []
            }
        }

    # ---------------------------
    # HELPERS
    # ---------------------------
    def _normalize_unicode_whitespace(self, x):
        if pd.isna(x): return x
        t = unicodedata.normalize("NFKC", str(x))
        # unify dashes for later parsing
        t = t.replace("–", "-").replace("—", "-").replace("−", "-")
        # unify separators often seen in LLM output
        t = t.replace("•", "; ").replace("·", "; ").replace("‧", "; ")
        # collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _strip_parentheses(self, t: str) -> str:
        return re.sub(r"\([^)]*\)", " ", t)

    def _clean_temporal_cell(
        self,
        x,
        strip_parentheses: bool = True,
        expand_ranges: bool = False,
        min_year: int = 1500,
        max_year: int = 2100,
        join_with: str = "; ",
        sort_unique: bool = True,
    ) -> str:
        """
        Normalize a temporal field to a simple 'YYYY' and/or 'YYYY-YYYY' list,
        joined by `join_with`. Removes non-year text.
        """
        if pd.isna(x): return ""
        t = self._normalize_unicode_whitespace(x)
        if strip_parentheses:
            t = self._strip_parentheses(t)
    
        # find ranges first to avoid double counting singles
        yr = r"(1[5-9]\d{2}|20\d{2}|2100)"
        range_pat = re.compile(rf"\b{yr}\s*-\s*{yr}\b")
        ranges = range_pat.findall(t)
        range_items = []
        singles_set = set()
    
        # remove matched ranges from text
        t_wo_ranges = range_pat.sub(" ", t)
    
        for a, b in ranges:
            a_i, b_i = int(a), int(b)
            if not (min_year <= a_i <= max_year and min_year <= b_i <= max_year):
                continue
            if expand_ranges:
                lo, hi = sorted((a_i, b_i))
                singles_set.update(range(lo, hi + 1))
            else:
                # store canonical "lo-hi"
                lo, hi = sorted((a_i, b_i))
                range_items.append(f"{lo}-{hi}")
    
        # now grab singles from remaining text
        singles = re.findall(rf"\b{yr}\b", t_wo_ranges)
        for s in singles:
            s_i = int(s)
            if min_year <= s_i <= max_year:
                singles_set.add(s_i)
    
        # compose string
        singles_list = list(singles_set)
        if sort_unique:
            singles_list.sort()
    
        parts = [str(y) for y in singles_list] + range_items
        return join_with.join(parts).strip()

    def _clean_set_cell(
        self,
        x,
        separators: Iterable[str],
        strip_parentheses: bool = True,
        lower: bool = True,
        lemmatize: bool = True,
        dedupe: bool = True,
        sort: bool = False,
        join_with: str = "; ",
        synonym_map: Optional[dict] = None,
        ignore_tokens: Optional[Iterable[str]] = None,
    ) -> str:
        """
        Normalize a set-like cell to a deduped, separator-joined list.
        """
        if pd.isna(x): return ""
        t = self._normalize_unicode_whitespace(x)
        if strip_parentheses:
            t = self._strip_parentheses(t)
    
        # build split regex from the chosen separators
        seps = [re.escape(s) for s in (separators or [",",";"])]
        split_pat = re.compile(rf"(?:{'|'.join(seps)})")
    
        raw_items = [s.strip() for s in split_pat.split(t)]
        items = []
        for it in raw_items:
            if not it:
                continue
            tok = it
            if lower:
                tok = tok.lower()
            if lemmatize:
                # lemma per word but preserve multi-word phrase
                words = [self.lemmatizer.lemmatize(w) for w in tok.split()]
                tok = " ".join(words)
            # apply synonym map if provided (exact match)
            if synonym_map and tok in synonym_map:
                tok = synonym_map[tok]
            items.append(tok)
    
        # optional ignore/drop tokens (exact match after normalize)
        if ignore_tokens:
            ignore_set = set([self.preprocess_text(s) for s in ignore_tokens])
            items = [it for it in items if self.preprocess_text(it) not in ignore_set]
    
        # dedupe while preserving order
        if dedupe:
            seen = set()
            deduped = []
            for it in items:
                if it not in seen:
                    seen.add(it)
                    deduped.append(it)
            items = deduped
    
        if sort:
            items = sorted(items)
    
        return join_with.join(items).strip()

    def _simple_cat_clean(self, x):
        if pd.isna(x): return ""
        t = self._normalize_unicode_whitespace(x)
        return t.lower().strip()

    def _clean_boolean_cell(self, x):
        if pd.isna(x): return ""
        t = self._normalize_unicode_whitespace(x).lower()
        # harmonize common variants
        if t in {"yes","y","true","1"}:
            return "yes"
        if t in {"no","n","false","0"}:
            return "no"
        return t

    def _split_set_items(self, text: str) -> list[str]:
        """Split a set-like cell into ordered items using the same separators used in cleaning."""
        if pd.isna(text) or str(text).strip() == "":
            return []
        t = self._normalize_unicode_whitespace(text)
        t = self._strip_parentheses(t)
        seps = [re.escape(s) for s in (self.set_separators or [",",";"])]
        pat = re.compile(rf"(?:{'|'.join(seps)})")
        items = [s.strip() for s in pat.split(t)]
        return [it for it in items if it]

    def _embed_sentences(self, sentences: list[str]) -> np.ndarray:
        """SentenceTransformer embeddings with caching."""
        to_compute = []
        for s in sentences:
            key = ("st", s)
            if key not in self._embed_cache:
                to_compute.append(s)
        if to_compute:
            vecs = self.st_model.encode(to_compute, normalize_embeddings=True)
            for s, v in zip(to_compute, vecs):
                self._embed_cache[("st", s)] = v
        # stack in input order
        return np.vstack([self._embed_cache[("st", s)] for s in sentences])

    def _pairwise_similarity_matrix(self, A: list[str], B: list[str]) -> np.ndarray:
        """Build |A|x|B| matrix using the selected metric."""
        if len(A) == 0 or len(B) == 0:
            return np.zeros((len(A), len(B)), dtype=float)
    
        if self.set_pair_metric == "bert_f1":
            # token-level BERTScore pairwise (slower). We only need F1.
            sims = np.zeros((len(A), len(B)), dtype=float)
            for i, a in enumerate(A):
                aa = self.preprocess_text(a)
                bb = [self.preprocess_text(b) for b in B]
                # scorer expects lists
                _, _, f1s = self.bert_scorer.score([aa]*len(bb), bb)
                sims[i, :] = np.array([float(x.item()) for x in f1s])
            return sims
        else:
            # default: cosine of sentence embeddings
            EA = self._embed_sentences([self.preprocess_text(x) for x in A])
            EB = self._embed_sentences([self.preprocess_text(x) for x in B])
            # cosine since normalized
            return (EA @ EB.T)

    def set_semantic_whole_text(self, llm_items: list[str], hum_items: list[str]) -> dict:
        """Join items and reuse free-text pipeline."""
        t_llm = self.set_joiner.join(llm_items)
        t_hum = self.set_joiner.join(hum_items)
        out = self.text_similarity(t_llm, t_hum)
        return {
            'set_text_cosine': out['cosine'],
            'set_text_bert_p': out['bert_p'],
            'set_text_bert_r': out['bert_r'],
            'set_text_bert_f1': out['bert_f1'],
            'set_text_jaro_winkler': out['jaro_winkler'],
            'set_text_levenshtein': out['levenshtein'],
            'set_text_ratcliff': out['ratcliff'],
            'set_text_rouge1_f': out['rouge1_f'],
            'set_text_rougeL_f': out['rougeL_f'],
        }

    def set_semantic_pairwise(self, llm_items: list[str], hum_items: list[str]) -> dict:
        """
        Soft matching via bipartite maximum similarity.
        Returns soft_precision/recall/F1, soft_jaccard, soft_dice, soft_overlap, plus basics.
        """
        A = [x for x in llm_items if x]
        B = [x for x in hum_items if x]
        a_n, b_n = len(A), len(B)
    
        if a_n == 0 and b_n == 0:
            return {
                'soft_precision': 1.0, 'soft_recall': 1.0, 'soft_f1': 1.0,
                'soft_jaccard': 1.0, 'soft_dice': 1.0, 'soft_overlap': 1.0,
                'soft_size_llm': 0, 'soft_size_hum': 0, 'soft_matched': 0,
                'soft_mean_pair_sim': 1.0, 'soft_median_pair_sim': 1.0
            }
        if a_n == 0 or b_n == 0:
            return {
                'soft_precision': 0.0, 'soft_recall': 0.0, 'soft_f1': 0.0,
                'soft_jaccard': 0.0, 'soft_dice': 0.0, 'soft_overlap': 0.0,
                'soft_size_llm': a_n, 'soft_size_hum': b_n, 'soft_matched': 0,
                'soft_mean_pair_sim': 0.0, 'soft_median_pair_sim': 0.0
            }
    
        S = self._pairwise_similarity_matrix(A, B)  # shape a_n x b_n
    
        # Hungarian maximization → minimize negative
        cost = -S
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_sims = S[row_ind, col_ind]
    
        # thresholding: count only strong matches as true positives
        tp_soft = float(np.sum(matched_sims >= self.set_pair_threshold))
        fp_soft = float(a_n - tp_soft)
        fn_soft = float(b_n - tp_soft)
    
        precision = tp_soft / a_n if a_n else 0.0
        recall    = tp_soft / b_n if b_n else 0.0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
    
        # soft Jaccard/Dice via thresholded tp
        union_soft = tp_soft + fp_soft + fn_soft
        jaccard = tp_soft/union_soft if union_soft > 0 else 1.0
        dice    = (2*tp_soft)/(2*tp_soft + fp_soft + fn_soft) if (2*tp_soft + fp_soft + fn_soft) > 0 else 1.0
        overlap = tp_soft / min(a_n, b_n) if min(a_n, b_n) > 0 else 1.0
    
        return {
            'soft_precision': precision,
            'soft_recall': recall,
            'soft_f1': f1,
            'soft_jaccard': jaccard,
            'soft_dice': dice,
            'soft_overlap': overlap,
            'soft_size_llm': a_n,
            'soft_size_hum': b_n,
            'soft_matched': int(tp_soft),
            'soft_mean_pair_sim': float(np.mean(matched_sims)) if len(matched_sims) else 0.0,
            'soft_median_pair_sim': float(np.median(matched_sims)) if len(matched_sims) else 0.0
        }

    def temporal_soft_similarity(self, llm_years: list[int], hum_years: list[int]) -> dict:
        """
        Counts a pair as a match if |a-b| <= tolerance. Greedy best match (Hungarian on -|a-b|).
        """
        A = sorted(set(llm_years))
        B = sorted(set(hum_years))
        a_n, b_n = len(A), len(B)
        if a_n == 0 and b_n == 0:
            return {'soft_f1_time': 1.0}
        if a_n == 0 or b_n == 0:
            return {'soft_f1_time': 0.0}
    
        # similarity = 1 if within tol else 0 (you can make this a ramp if needed)
        tol = abs(self.temporal_tolerance)
        # make cost = 0 if match within tol, 1 otherwise
        C = np.ones((a_n, b_n), dtype=float)
        for i,a in enumerate(A):
            for j,b in enumerate(B):
                if abs(a-b) <= tol:
                    C[i,j] = 0.0
        row, col = linear_sum_assignment(C)
        tp = float(np.sum(C[row, col] == 0.0))
        precision = tp / a_n if a_n else 0.0
        recall    = tp / b_n if b_n else 0.0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
        return {'soft_f1_time': f1}

    def preprocess_text(self, text):
        """Normalize text for comparison"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.strip()

    def tokenize_set(self, text):
        """Convert set-based text to normalized token set"""
        text = self.preprocess_text(text)
        if not text:
            return set()
        
        tokens = []
        for token in re.split(r'[,;]|\band\b', text):
            token = token.strip()
            if token and token not in self.stop_words:
                lemma = self.lemmatizer.lemmatize(token)
                tokens.append(lemma)
        return set(tokens)

    def normalize_categorical(self, text):
        """Normalize categorical values using synonym mapping"""
        text = self.preprocess_text(text)
        for pattern, standard in self.synonym_map.items():
            if pattern in text:
                return standard
        return text

    def parse_years(self, text):
        """Parse year values into comparable format"""
        if pd.isna(text):
            return []
        
        years = []
        text = str(text)
        
        # Handle year ranges
        for part in re.split(r'[,\s;]+', text):
            if re.match(r'^\d{4}$', part):
                years.append(int(part))
            elif re.match(r'^\d{4}-\d{4}$', part):
                start, end = map(int, part.split('-'))
                years.extend(range(start, end + 1))
        
        return list(set(years))

    def set_similarity(self, set1: set, set2: set):
        """
        Returns:
          jaccard, precision, recall, f1,
          dice, overlap, tversky,
          size_hum, size_llm, size_intersection, size_union,
          hum_only, llm_only
        """
        # both empty: perfect agreement
        if not set1 and not set2:
            return {
                'jaccard': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                'dice': 1.0, 'overlap': 1.0, 'tversky': 1.0,
                'size_hum': 0, 'size_llm': 0, 'size_intersection': 0, 'size_union': 0,
                'hum_only': 0, 'llm_only': 0
            }

        inter = set1 & set2
        uni   = set1 | set2
        a_minus_b = set1 - set2
        b_minus_a = set2 - set1

        inter_n = len(inter)
        union_n = len(uni)
        a_n = len(set1)
        b_n = len(set2)

        precision = inter_n / a_n if a_n else 0.0
        recall    = inter_n / b_n if b_n else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0

        dice = (2*inter_n) / (a_n + b_n) if (a_n + b_n) > 0 else 1.0
        overlap = inter_n / min(a_n, b_n) if min(a_n, b_n) > 0 else 1.0

        alpha = self.tversky_alpha
        beta  = self.tversky_beta
        denom = inter_n + alpha*len(a_minus_b) + beta*len(b_minus_a)
        tversky = inter_n / denom if denom > 0 else 1.0

        return {
            'jaccard': inter_n/union_n if union_n else 1.0,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'dice': dice,
            'overlap': overlap,
            'tversky': tversky,
            'size_hum': b_n,              # human is the reference in your calls
            'size_llm': a_n,
            'size_intersection': inter_n,
            'size_union': union_n,
            'hum_only': len(b_minus_a),
            'llm_only': len(a_minus_b),
        }

    def text_similarity(self, text1, text2):
        """
        Returns:
          cosine,
          bert_p, bert_r, bert_f1,
          jaro_winkler, levenshtein, ratcliff,
          rouge1_f, rougeL_f
          (also rouge1_p/rouge1_r, rougeL_p/rougeL_r for reference)
        """
        t1 = self.preprocess_text(text1)
        t2 = self.preprocess_text(text2)

        if not t1 and not t2:
            return {
                'cosine': 1.0,
                'bert_p': 1.0, 'bert_r': 1.0, 'bert_f1': 1.0,
                'jaro_winkler': 1.0, 'levenshtein': 1.0, 'ratcliff': 1.0,
                'rouge1_p': 1.0, 'rouge1_r': 1.0, 'rouge1_f': 1.0,
                'rougeL_p': 1.0, 'rougeL_r': 1.0, 'rougeL_f': 1.0
            }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Sentence-level embedding cosine
            emb1 = self.st_model.encode([t1])[0]
            emb2 = self.st_model.encode([t2])[0]
            cos_sim = float(cosine_similarity([emb1], [emb2])[0][0])

            # BERTScore (P, R, F1)
            bert_p, bert_r, bert_f1 = self.bert_scorer.score([t1], [t2])
            bert_p = float(bert_p.item())
            bert_r = float(bert_r.item())
            bert_f1 = float(bert_f1.item())

            # Char/seq similarities (normalized 0..1)
            jw = float(textdistance.jaro_winkler.normalized_similarity(t1, t2))
            lev = float(textdistance.levenshtein.normalized_similarity(t1, t2))
            rat = float(textdistance.ratcliff_obershelp.normalized_similarity(t1, t2))

            # ROUGE-1 / ROUGE-L (word level)
            ref_tok = self._word_tokens(t2)  # human is reference
            hyp_tok = self._word_tokens(t1)  # llm is hypothesis
            r1_p, r1_r, r1_f = self._rouge1_f(ref_tok, hyp_tok)
            rl_p, rl_r, rl_f = self._rougeL_f(ref_tok, hyp_tok)

        return {
            'cosine': cos_sim,
            'bert_p': bert_p, 'bert_r': bert_r, 'bert_f1': bert_f1,
            'jaro_winkler': jw, 'levenshtein': lev, 'ratcliff': rat,
            'rouge1_p': r1_p, 'rouge1_r': r1_r, 'rouge1_f': r1_f,
            'rougeL_p': rl_p, 'rougeL_r': rl_r, 'rougeL_f': rl_f
        }

    # ---------- helpers for free-text metrics ----------
    def _word_tokens(self, text: str) -> list[str]:
        """Whitespace word tokens after your normalize step."""
        t = self.preprocess_text(text)
        if not t:
            return []
        return [w for w in t.split() if w not in self.stop_words]

    def _lcs_len(self, a: list[str], b: list[str]) -> int:
        """Longest Common Subsequence length for ROUGE-L."""
        n, m = len(a), len(b)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(1, n+1):
            ai = a[i-1]
            row_i = dp[i]
            row_im1 = dp[i-1]
            for j in range(1, m+1):
                if ai == b[j-1]:
                    row_i[j] = row_im1[j-1] + 1
                else:
                    row_i[j] = max(row_im1[j], row_i[j-1])
        return dp[n][m]

    def _rouge1_f(self, ref_tokens: list[str], hyp_tokens: list[str]):
        """ROUGE-1 P/R/F1 at word level."""
        if not ref_tokens and not hyp_tokens:
            return 1.0, 1.0, 1.0
        if not ref_tokens or not hyp_tokens:
            return 0.0, 0.0, 0.0
        from collections import Counter
        ref_cnt = Counter(ref_tokens)
        hyp_cnt = Counter(hyp_tokens)
        overlap = sum((ref_cnt & hyp_cnt).values())
        p = overlap / max(1, sum(hyp_cnt.values()))
        r = overlap / max(1, sum(ref_cnt.values()))
        f = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        return p, r, f

    def _rougeL_f(self, ref_tokens: list[str], hyp_tokens: list[str]):
        """ROUGE-L P/R/F1 using LCS."""
        if not ref_tokens and not hyp_tokens:
            return 1.0, 1.0, 1.0
        if not ref_tokens or not hyp_tokens:
            return 0.0, 0.0, 0.0
        lcs = self._lcs_len(ref_tokens, hyp_tokens)
        p = lcs / len(hyp_tokens) if hyp_tokens else 0.0
        r = lcs / len(ref_tokens) if ref_tokens else 0.0
        f = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        return p, r, f

    def calculate_completeness_metrics(self, col):
        """Calculate data quality metrics"""
        hu_complete = self.df_hum[col].notna().sum()
        llm_complete = self.df_llm[col].notna().sum()
        total = len(self.df_hum)
        
        return {
            'human_completeness': hu_complete / total,
            'llm_completeness': llm_complete / total,
            'completeness_gap': (hu_complete - llm_complete) / total,
            'both_missing': (self.df_hum[col].isna() & self.df_llm[col].isna()).sum() / total,
            'llm_hallucination': (self.df_hum[col].isna() & self.df_llm[col].notna()).sum() / total,
            'llm_missed': (self.df_hum[col].notna() & self.df_llm[col].isna()).sum() / total
        }

    def calculate_statistical_metrics(self, results_df, metric):
        """Calculate statistical significance metrics with robust error handling"""
        from scipy import stats
        import numpy as np
        
        values = results_df[metric].dropna()
        if len(values) < 2:
            return {}
        
        try:
            # Handle near-constant data
            if values.nunique() == 1:
                # All values are identical
                constant_value = values.iloc[0]
                return {
                    'confidence_interval_95': (constant_value, constant_value),
                    'variance': 0.0,
                    'coefficient_variation': 0.0,
                    'skewness': 0.0,  # Constant data has no skew
                    'kurtosis': -3.0  # Excess kurtosis for constant data
                }
            
            sem = stats.sem(values)
            mean_val = np.mean(values)
            
            # Robust confidence interval calculation
            if sem > 0 and len(values) > 1:
                try:
                    ci_low, ci_high = stats.t.interval(
                        0.95, len(values)-1, loc=mean_val, scale=sem
                    )
                    # Handle potential NaN in confidence intervals
                    if np.isnan(ci_low) or np.isnan(ci_high):
                        ci_low, ci_high = mean_val, mean_val
                except (ValueError, RuntimeError):
                    ci_low, ci_high = mean_val, mean_val
            else:
                ci_low, ci_high = mean_val, mean_val
            
            # Robust moment calculations
            variance = np.var(values, ddof=1)  # Sample variance
            
            # Calculate coefficient of variation safely
            if mean_val != 0:
                coef_var = stats.variation(values) if len(values) > 1 else 0
            else:
                coef_var = np.nan
            
            # Calculate skewness and kurtosis with error handling
            try:
                skewness = stats.skew(values) if len(values) > 2 else 0.0
            except (RuntimeWarning, ValueError):
                skewness = 0.0
                
            try:
                kurtosis = stats.kurtosis(values) if len(values) > 3 else 0.0
            except (RuntimeWarning, ValueError):
                kurtosis = 0.0
            
            return {
                'confidence_interval_95': (float(ci_low), float(ci_high)),
                'variance': float(variance),
                'coefficient_variation': float(coef_var) if not np.isnan(coef_var) else 0.0,
                'skewness': float(skewness),
                'kurtosis': float(kurtosis)
            }
            
        except Exception as e:
            print(f"Warning: Statistical calculation failed for metric {metric}: {str(e)}")
            return {}

    def calculate_performance_distribution(self, results_df, metric):
        """Calculate performance distribution metrics (only for [0,1] bounded metrics)."""
        values = results_df[metric].dropna()
        if len(values) == 0:
            return {}

        vmin, vmax = float(values.min()), float(values.max())
        # skip counts and unbounded metrics
        if not (0.0 - 1e-9 <= vmin and vmax <= 1.0 + 1e-9):
            return {}

        thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.3
        }
        dist_metrics = {}
        for level, thr in thresholds.items():
            dist_metrics[f'pct_{level}'] = (values >= thr).mean()

        dist_metrics.update({
            'consistency_score': (values >= 0.7).mean(),
            'worst_case': values.min(),
            'best_case': values.max(),
            'iqr': values.quantile(0.75) - values.quantile(0.25)
        })
        return dist_metrics

    def calculate_cross_column_metrics(self, strategy='auto'):
        """
        Compute cross-type performance with safe fallbacks.
        strategy:
          'primary' -> use the first available metric in the fallback list per type
          'auto'    -> try primary chain; if none present, use median of any 0..1 metrics
        """
        import numpy as np
    
        if not getattr(self, 'dict_results', None):
            return {}
    
        # metric preference per type
        PRIMARY = {
            'set_based':  ['soft_f1', 'f1', 'jaccard'],
            'temporal':   ['f1', 'jaccard', 'soft_f1_time'],
            'free_text':  ['bert_f1', 'cosine'],
            'categorical':['match'],
            'boolean':    ['match'],
        }
    
        cross_metrics = {}
    
        for col_type, columns in self.column_types.items():
            values = []
            cols_used = []
            metrics_used = []
    
            for col in columns:
                if col not in self.dict_results:
                    continue
                stats = self.dict_results[col].get('stats', {})
    
                chosen = None
                chosen_val = None
    
                # try primary/fallback order
                for m in PRIMARY.get(col_type, []):
                    if m in stats and np.isfinite(stats[m].get('mean', np.nan)):
                        chosen = m
                        chosen_val = float(stats[m]['mean'])
                        break
    
                # optional aggregate fallback
                if chosen is None and strategy == 'auto':
                    cand = [
                        float(s['mean']) for m, s in stats.items()
                        if ('mean' in s and np.isfinite(s['mean'])
                            and 0.0 <= float(s['mean']) <= 1.0)
                    ]
                    if len(cand):
                        chosen = 'median_0_1'
                        chosen_val = float(np.median(cand))
    
                if chosen is None:
                    # nothing usable for this column
                    continue
    
                values.append(chosen_val)
                cols_used.append(col)
                metrics_used.append(chosen)
    
            if not values:
                # no usable columns for this type
                continue
    
            vals = np.asarray(values, dtype=float)
            best_idx = int(np.nanargmax(vals))
            worst_idx = int(np.nanargmin(vals))
    
            cross_metrics[col_type] = {
                'mean_performance': float(np.nanmean(vals)),
                'performance_range': (float(np.nanmin(vals)), float(np.nanmax(vals))),
                # use the filtered lists so indexing is correct
                'most_reliable_column': cols_used[best_idx],
                'least_reliable_column': cols_used[worst_idx],
                'n_columns_evaluated': len(cols_used),
                'columns_used': cols_used,
                'metrics_used_per_column': dict(zip(cols_used, metrics_used)),
                'strategy': strategy,
            }
    
        return cross_metrics

    def _is_truly_missing(self, value):
        """Check if value is truly missing/empty"""
        if pd.isna(value):
            return True
        if isinstance(value, str) and value.strip() in ['', 'nan', 'NaN', 'None', 'null']:
            return True
        return False

    def _create_default_result(self, match_score):
        """Create a consistent default result structure"""
        return {
            'match': float(match_score),
            'f1': float(match_score),
            'precision': float(match_score),
            'recall': float(match_score)
        }

    def _create_empty_evaluation(self):
        """Create empty evaluation structure for problematic columns"""
        return {
            'raw_results': [],
            'results_df': pd.DataFrame(),
            'stats': {},
            'data_quality': {'completeness': 0.0, 'valid_count': 0}
        }

    def _create_error_evaluation(self, error_msg):
        """Create error evaluation structure"""
        return {
            'raw_results': [],
            'results_df': pd.DataFrame(),
            'stats': {'error': error_msg},
            'data_quality': {'completeness': 0.0, 'valid_count': 0, 'error': error_msg}
        }

    def validate_data_before_evaluation(self):
        """Validate data quality before running evaluation"""
        issues = []
        for col in self.df_hum.columns:
            if col in self.df_llm.columns:
                human_nulls = self.df_hum[col].isna().sum()
                llm_nulls = self.df_llm[col].isna().sum()
                if human_nulls > 0.8 * len(self.df_hum) or llm_nulls > 0.8 * len(self.df_llm):
                    issues.append(f"High null percentage in column {col}")
        
        if issues:
            print("Data quality issues found:")
            for issue in issues:
                print(f"  - {issue}")
        return issues

    def evaluate_column(self, col):
        """Evaluate performance for a single column with validation"""
        results = []
        
        for idx in self.df_hum.index:
            hu_val = self.df_hum.loc[idx, col]
            llm_val = self.df_llm.loc[idx, col]
            
            # Enhanced missing value handling
            if self._is_truly_missing(hu_val) and self._is_truly_missing(llm_val):
                results.append(self._create_default_result(1.0))
                continue
            if self._is_truly_missing(hu_val) or self._is_truly_missing(llm_val):
                results.append(self._create_default_result(0.0))
                continue
            
            # Handle missing values
            if pd.isna(hu_val) and pd.isna(llm_val):
                results.append({'match': 1.0})
                continue
            if pd.isna(hu_val) or pd.isna(llm_val):
                results.append({'match': 0.0})
                continue
            
            # Column type-specific evaluation
            if col in self.column_types['free_text']:
                hu_text = self.preprocess_text(hu_val)
                llm_text = self.preprocess_text(llm_val)
                results.append(self.text_similarity(llm_text, hu_text))
                
            elif col in self.column_types['categorical']:
                hu_cat = self.normalize_categorical(hu_val)
                llm_cat = self.normalize_categorical(llm_val)
                results.append({'match': 1.0 if hu_cat == llm_cat else 0.0})
                
            elif col in self.column_types['set_based']:
                # ordered lists for semantics; also build sets for exact metrics
                hu_items_raw  = self._split_set_items(hu_val)
                llm_items_raw = self._split_set_items(llm_val)
            
                # exact set tokens (your original behavior)
                hu_set  = set(self.tokenize_set(hu_val))
                llm_set = set(self.tokenize_set(llm_val))
                base = self.set_similarity(llm_set, hu_set)
            
                extra = {}
                if self.set_use_semantics and self.set_semantic_mode in ("whole","both"):
                    extra.update(self.set_semantic_whole_text(llm_items_raw, hu_items_raw))
                if self.set_use_semantics and self.set_semantic_mode in ("pairwise","both"):
                    extra.update(self.set_semantic_pairwise(llm_items_raw, hu_items_raw))
            
                res = {**base, **extra}
                results.append(res)
            
            elif col in self.column_types['temporal']:
                hu_years = self.parse_years(hu_val)
                llm_years = self.parse_years(llm_val)
                base = self.set_similarity(set(llm_years), set(hu_years))
                if self.temporal_tolerance > 0:
                    base.update(self.temporal_soft_similarity(llm_years, hu_years))
                results.append(base)
                
            elif col in self.column_types['boolean']:
                results.append({'match': 1.0 if str(hu_val).lower() == str(llm_val).lower() else 0.0})
                
            else:  # Default to exact match
                results.append({'match': 1.0 if hu_val == llm_val else 0.0})
        
        return results

    def evaluate_all(self, selected_columns='all', crosscols_strategy='auto', print_output=False):
        """
        Evaluate specified columns and compute aggregate statistics with enhanced error handling
        
        Parameters:
        selected_columns (str or list): 
            'all' to evaluate all valid columns, or list of specific columns to evaluate
        """
        # Determine columns to evaluate
        if selected_columns == 'all':
            cols_to_evaluate = [col for col in self.df_hum.columns if col in self.valid_columns]
        else:
            # Validate selected columns
            invalid_cols = [col for col in selected_columns if col not in self.valid_columns]
            if invalid_cols:
                raise ValueError(f"Invalid columns selected: {invalid_cols}. "
                                 f"Valid columns are: {list(self.valid_columns)}")
            cols_to_evaluate = selected_columns
        
        # Rows are paired by index label (see evaluate_column). Guard against
        # misaligned tables when an identifier column is available.
        for id_col in ('Key', 'DOI'):
            if id_col in self.df_hum.columns and id_col in self.df_llm.columns:
                hum_ids = self.df_hum[id_col].astype(str).str.strip()
                llm_ids = self.df_llm[id_col].reindex(self.df_hum.index).astype(str).str.strip()
                if not hum_ids.equals(llm_ids):
                    raise ValueError(f"Human and LLM tables are not aligned on '{id_col}'. "
                                     "Align both tables to the same index before evaluation.")
                break

        evaluation = {}
        
        for col in cols_to_evaluate:
            print(f"Evaluating column: {col}")
            try:
                col_results = self.evaluate_column(col)
                
                # Convert to DataFrame for analysis
                results_df = pd.DataFrame(col_results)
                
                # Skip statistical calculations if no valid data
                if len(results_df) == 0 or results_df.isna().all().all():
                    print(f"Warning: No valid data for column {col}")
                    evaluation[col] = self._create_empty_evaluation()
                    continue
                
                # Enhanced statistics
                enhanced_stats = {}
                for metric in results_df.columns:
                    metric_stats = {
                        'mean': results_df[metric].mean(),
                        'std': results_df[metric].std(),
                        'median': results_df[metric].median(),
                        'min': results_df[metric].min(),
                        'max': results_df[metric].max(),
                        'q1': results_df[metric].quantile(0.25),
                        'q3': results_df[metric].quantile(0.75)
                    }
                    
                    # Add statistical significance metrics
                    stat_metrics = self.calculate_statistical_metrics(results_df, metric)
                    metric_stats.update(stat_metrics)
                    
                    # Add performance distribution metrics
                    dist_metrics = self.calculate_performance_distribution(results_df, metric)
                    metric_stats.update(dist_metrics)
                    
                    enhanced_stats[metric] = metric_stats
                
                # Store column results with data quality metrics
                evaluation[col] = {
                    'raw_results': col_results,
                    'results_df': results_df,  # Store the DataFrame for reference
                    'stats': enhanced_stats,
                    'data_quality': self.calculate_completeness_metrics(col)
                }
                
            except Exception as e:
                print(f"Error evaluating column {col}: {str(e)}")
                evaluation[col] = self._create_error_evaluation(str(e))
        
        self.dict_results = evaluation
        
        # Calculate cross-column metrics
        self.cross_column_metrics = self.calculate_cross_column_metrics(strategy=crosscols_strategy)
        
        if print_output: 
            print("Evaluation completed. Use generate_summary_report() or get_detailed_results() to view results.")
        return evaluation

    def generate_summary_report(self, print_output=False):
        """Generate a comprehensive summary report including ALL enhanced statistics for ALL metrics"""
        if not self.dict_results:
            self.evaluate_all()
            
        summary_rows = []
        
        for col, data in self.dict_results.items():
            col_type = next((k for k, v in self.column_types.items() if col in v), 'other')
            
            # Process ALL metrics for this column, not just primary metric
            for metric_name, metric_stats in data['stats'].items():
                # Data quality metrics
                dq = data['data_quality']
                
                # Create comprehensive summary row for EACH metric
                row = {
                    'Column': col,
                    'Type': col_type,
                    'Metric': metric_name,
                    'Mean': metric_stats.get('mean', np.nan),
                    'Std_Dev': metric_stats.get('std', np.nan),
                    'Median': metric_stats.get('median', np.nan),
                    'Min': metric_stats.get('min', np.nan),
                    'Max': metric_stats.get('max', np.nan),
                    'Q1': metric_stats.get('q1', np.nan),
                    'Q3': metric_stats.get('q3', np.nan),
                    'CI95_Low': metric_stats.get('confidence_interval_95', (np.nan, np.nan))[0] if metric_stats.get('confidence_interval_95') else np.nan,
                    'CI95_High': metric_stats.get('confidence_interval_95', (np.nan, np.nan))[1] if metric_stats.get('confidence_interval_95') else np.nan,
                    'Variance': metric_stats.get('variance', np.nan),
                    'CV': metric_stats.get('coefficient_variation', np.nan),
                    'Skewness': metric_stats.get('skewness', np.nan),
                    'Kurtosis': metric_stats.get('kurtosis', np.nan),
                    'Pct_Excellent': metric_stats.get('pct_excellent', np.nan),
                    'Pct_Good': metric_stats.get('pct_good', np.nan),
                    'Pct_Fair': metric_stats.get('pct_fair', np.nan),
                    'Pct_Poor': metric_stats.get('pct_poor', np.nan),
                    'Consistency_Score': metric_stats.get('consistency_score', np.nan),
                    'Worst_Case': metric_stats.get('worst_case', np.nan),
                    'Best_Case': metric_stats.get('best_case', np.nan),
                    'IQR': metric_stats.get('iqr', np.nan),
                    'Human_Completeness': dq.get('human_completeness', np.nan),
                    'LLM_Completeness': dq.get('llm_completeness', np.nan),
                    'Completeness_Gap': dq.get('completeness_gap', np.nan),
                    'Both_Missing': dq.get('both_missing', np.nan),
                    'LLM_Hallucination': dq.get('llm_hallucination', np.nan),
                    'LLM_Missed': dq.get('llm_missed', np.nan)
                }
                summary_rows.append(row)
        
        self.df_summary = pd.DataFrame(summary_rows)
        
        if print_output: 
            print("COMPREHENSIVE SUMMARY REPORT - ALL METRICS:")
            print("\n" + "="*100)
            print(f"{'COMPREHENSIVE SUMMARY REPORT (ALL METRICS)':^100}")
            print("="*100)
            
            # Group by column for better readability
            for col in self.df_summary['Column'].unique():
                col_data = self.df_summary[self.df_summary['Column'] == col]
                print(f"\n{col} ({col_data['Type'].iloc[0]}):")
                print("-" * 80)
                
                # Display each metric for this column
                for _, row in col_data.iterrows():
                    print(f"  {row['Metric']:>12}: Mean={row['Mean']:.4f} | CI95%=[{row['CI95_Low']:.4f}, {row['CI95_High']:.4f}] | "
                          f"Dist[E:{row['Pct_Excellent']:.2%}, G:{row['Pct_Good']:.2%}, F:{row['Pct_Fair']:.2%}, P:{row['Pct_Poor']:.2%}]")
        
        return self.df_summary

    def get_detailed_results(self, as_dataframe=False, print_output=False):
        """Return detailed results including enhanced statistics"""
        if not self.dict_results:
            self.evaluate_all(print_output=print_output)
        
        if not as_dataframe:
            if print_output: 
                print("Detailed Results (Dictionary with Enhanced Stats):")
                for col, data in self.dict_results.items():
                    print(f"\n{col}:")
                    print(f"  Data Quality: {data['data_quality']}")
                    print("  Enhanced Stats:")
                    for metric, stats in data['stats'].items():
                        print(f"    {metric}: {stats}")
            return self.dict_results
        
        # Build comprehensive DataFrame from results
        rows = []
        for col, col_data in self.dict_results.items():
            # Get column type
            col_type = next(
                (k for k, v in self.column_types.items() if col in v),
                'other'
            )
            
            # Get data quality metrics for this column
            dq = col_data['data_quality']
            
            for idx, result_dict in enumerate(col_data['raw_results']):
                row = {
                    'Paper_Index': idx,
                    'Column': col,
                    'Column_Type': col_type,
                    'Human_Completeness': dq['human_completeness'],
                    'LLM_Completeness': dq['llm_completeness'],
                    'Completeness_Gap': dq['completeness_gap'],
                    'Both_Missing': dq['both_missing'],
                    'LLM_Hallucination': dq['llm_hallucination'],
                    'LLM_Missed': dq['llm_missed']
                }
                
                # Add ALL metrics for this column-paper combination
                for metric, value in result_dict.items():
                    row[metric] = value
                    
                rows.append(row)
        
        results_df = pd.DataFrame(rows)
        
        # Fill missing metrics with NaN
        # --- exact set/temporal metrics (order-insensitive, original ones) ---
        SET_EXACT = ['jaccard','precision','recall','f1','dice','overlap','tversky', 'size_hum',
                     'size_llm','size_intersection','size_union','hum_only','llm_only']
        
        # --- whole-text semantics for set lists (join items then run text sims) ---
        SET_WHOLE = ['set_text_cosine','set_text_bert_p','set_text_bert_r','set_text_bert_f1',
                     'set_text_jaro_winkler','set_text_levenshtein','set_text_ratcliff', 
                     'set_text_rouge1_f','set_text_rougeL_f']
        
        # --- pairwise semantics for set lists (Hungarian matching + threshold) ---
        SET_PAIR = ['soft_precision','soft_recall','soft_f1','soft_jaccard','soft_dice','soft_overlap',
            'soft_size_llm','soft_size_hum','soft_matched','soft_mean_pair_sim','soft_median_pair_sim']
        
        # --- optional temporal soft match (tolerance in years) ---
        TEMP_SOFT = ['soft_f1_time']  # present only if you enabled temporal_tolerance > 0
        
        # --- free-text metrics (unchanged) ---
        FREE_TEXT = ['cosine','bert_p','bert_r','bert_f1', 'jaro_winkler','levenshtein','ratcliff',
                     'rouge1_p','rouge1_r','rouge1_f','rougeL_p','rougeL_r','rougeL_f']
        
        # --- categorical/boolean (unchanged) ---
        CATEG_BOOL = ['match']
        
        # --- per-category map (used by your reporting/plots if helpful) ---
        POSSIBLE_METRICS_BY_CAT = {
            'set_based':   SET_EXACT + SET_WHOLE + SET_PAIR,
            'temporal':    SET_EXACT + TEMP_SOFT,   # exact set-style + optional soft time
            'free_text':   FREE_TEXT,
            'categorical': CATEG_BOOL,
            'boolean':     CATEG_BOOL,}
        
        # --- single flattened list (what you asked to "update") ---
        possible_metrics = list(dict.fromkeys(
            POSSIBLE_METRICS_BY_CAT['set_based']
            + POSSIBLE_METRICS_BY_CAT['temporal']
            + POSSIBLE_METRICS_BY_CAT['free_text']
            + POSSIBLE_METRICS_BY_CAT['categorical']
            + POSSIBLE_METRICS_BY_CAT['boolean']))
        
        for metric in possible_metrics:
            if metric not in results_df:
                results_df[metric] = np.nan
        
        self.df_detailed = results_df
        
        if print_output: 
            print("Detailed Results (DataFrame with Enhanced Stats):")
            print("\nFirst 5 rows:")
            print(self.df_detailed.head())
            print(f"\nShape: {self.df_detailed.shape}")
            print(f"\nAvailable metrics: {[col for col in self.df_detailed.columns if col not in ['Paper_Index', 'Column', 'Column_Type']]}")
            
            # Show metric availability per column type
            print("\nMetric Availability by Column Type:")
            for col_type in self.df_detailed['Column_Type'].unique():
                type_data = self.df_detailed[self.df_detailed['Column_Type'] == col_type]
                available_metrics = [col for col in type_data.columns if type_data[col].notna().any() and col not in ['Paper_Index', 'Column', 'Column_Type']]
                print(f"  {col_type}: {available_metrics}")
        
        return self.df_detailed

    def get_metric_summary(self, metric_name, print_output=False):
        """Get summary for a specific metric across all columns"""
        if not self.dict_results:
            self.evaluate_all()
        
        metric_rows = []
        for col, data in self.dict_results.items():
            col_type = next((k for k, v in self.column_types.items() if col in v), 'other')
            
            if metric_name in data['stats']:
                stats = data['stats'][metric_name]
                dq = data['data_quality']
                
                row = {
                    'Column': col,
                    'Type': col_type,
                    'Metric': metric_name,
                    'Mean': stats.get('mean', np.nan),
                    'Std_Dev': stats.get('std', np.nan),
                    'Median': stats.get('median', np.nan),
                    'Min': stats.get('min', np.nan),
                    'Max': stats.get('max', np.nan),
                    'CI95_Low': stats.get('confidence_interval_95', (np.nan, np.nan))[0] if stats.get('confidence_interval_95') else np.nan,
                    'CI95_High': stats.get('confidence_interval_95', (np.nan, np.nan))[1] if stats.get('confidence_interval_95') else np.nan,
                    'Pct_Excellent': stats.get('pct_excellent', np.nan),
                    'Pct_Good': stats.get('pct_good', np.nan),
                    'Pct_Fair': stats.get('pct_fair', np.nan),
                    'Pct_Poor': stats.get('pct_poor', np.nan),
                    'Human_Completeness': dq.get('human_completeness', np.nan),
                    'LLM_Completeness': dq.get('llm_completeness', np.nan)
                }
                metric_rows.append(row)
        
        metric_df = pd.DataFrame(metric_rows)
        
        if print_output and not metric_df.empty:
            print(f"\n{metric_name.upper()} METRIC SUMMARY ACROSS ALL COLUMNS:")
            print("="*80)
            for _, row in metric_df.iterrows():
                print(f"{row['Column']:.<30} Mean: {row['Mean']:.4f} | "
                      f"CI95%: [{row['CI95_Low']:.4f}, {row['CI95_High']:.4f}] | "
                      f"Quality: E:{row['Pct_Excellent']:.1%} G:{row['Pct_Good']:.1%} "
                      f"F:{row['Pct_Fair']:.1%} P:{row['Pct_Poor']:.1%}")
        
        return metric_df

    def generate_cross_column_report(self, as_dataframe=False, print_output=False):
        """Generate a dedicated report for cross-column analysis
        
        Parameters:
        as_dataframe (bool): If True, return results as DataFrame instead of dict
        print_output (bool): If True, print formatted report to console
        """
        if not hasattr(self, 'cross_column_metrics') or not self.cross_column_metrics:
            self.cross_column_metrics = self.calculate_cross_column_metrics()
        
        if as_dataframe:
            # Convert cross-column metrics to DataFrame
            rows = []
            for col_type, metrics in self.cross_column_metrics.items():
                row = {
                    'column_type': col_type,
                    'mean_performance': metrics.get('mean_performance', np.nan),
                    'performance_min': metrics.get('performance_range', (np.nan, np.nan))[0],
                    'performance_max': metrics.get('performance_range', (np.nan, np.nan))[1],
                    'performance_range': metrics.get('performance_range', (np.nan, np.nan)),
                    'most_reliable_column': metrics.get('most_reliable_column', 'N/A'),
                    'least_reliable_column': metrics.get('least_reliable_column', 'N/A'),
                    'num_columns_evaluated': len([col for col in self.column_types[col_type] 
                                                if col in self.dict_results]) if col_type in self.column_types else 0
                }
                # inside the for-loop building 'row' per col_type
                used_map = metrics.get('metrics_used_per_column', {})
                counts = {}
                for m in used_map.values():
                    counts[m] = counts.get(m, 0) + 1
                
                row['num_columns_evaluated'] = metrics.get(
                    'n_columns_evaluated',
                    len([c for c in self.column_types.get(col_type, []) if c in self.dict_results])
                )
                row['metric_strategy'] = metrics.get('strategy', 'primary')
                row['metric_used_summary'] = ', '.join(f'{k}:{v}' for k, v in sorted(counts.items()))
                rows.append(row)
            
            cross_df = pd.DataFrame(rows)
            
            # Add additional calculated metrics
            if not cross_df.empty:
                cross_df['performance_span'] = cross_df['performance_max'] - cross_df['performance_min']
                cross_df['consistency_ratio'] = cross_df['performance_span'] / cross_df['mean_performance']
                cross_df['rank'] = cross_df['mean_performance'].rank(ascending=False).astype(int)
            
            if print_output:
                print("\n" + "="*80)
                print(f"{'CROSS-COLUMN PERFORMANCE ANALYSIS (DATAFRAME)':^80}")
                print("="*80)
                print(cross_df.round(4))
                
                # Print performance ranking
                print(f"\n{'PERFORMANCE RANKING BY COLUMN TYPE':^80}")
                print("-" * 80)
                for _, row in cross_df.sort_values('mean_performance', ascending=False).iterrows():
                    print(f"{row['column_type'].upper():<15}: {row['mean_performance']:.4f} "
                          f"(Range: {row['performance_min']:.4f}-{row['performance_max']:.4f}) | "
                          f"Best: {row['most_reliable_column']}")
            
            return cross_df
        
        else:
            # Original dictionary output with enhanced printing
            if print_output:
                print("\n" + "="*80)
                print(f"{'CROSS-COLUMN PERFORMANCE ANALYSIS':^80}")
                print("="*80)
                
                # Calculate overall statistics
                all_means = [metrics.get('mean_performance', 0) 
                            for metrics in self.cross_column_metrics.values()]
                overall_mean = np.mean(all_means) if all_means else 0
                
                print(f"\n{'OVERALL SUMMARY':^80}")
                print(f"Average Performance Across All Types: {overall_mean:.4f}")
                print(f"Best Performing Type: {max(self.cross_column_metrics.items(), key=lambda x: x[1].get('mean_performance', 0))[0]}")
                print(f"Worst Performing Type: {min(self.cross_column_metrics.items(), key=lambda x: x[1].get('mean_performance', 0))[0]}")
                
                # Print detailed breakdown by column type
                for col_type, metrics in sorted(self.cross_column_metrics.items(), 
                                              key=lambda x: x[1].get('mean_performance', 0), 
                                              reverse=True):
                    print(f"\n{col_type.upper()} COLUMNS:")
                    print(f"  Mean Performance: {metrics.get('mean_performance', 0):.4f}")
                    perf_range = metrics.get('performance_range', (0, 0))
                    print(f"  Performance Range: {perf_range[0]:.4f} - {perf_range[1]:.4f} "
                          f"(Span: {perf_range[1] - perf_range[0]:.4f})")
                    print(f"  Most Reliable Column: {metrics.get('most_reliable_column', 'N/A')}")
                    print(f"  Least Reliable Column: {metrics.get('least_reliable_column', 'N/A')}")
                    
                    # Add performance interpretation
                    mean_perf = metrics.get('mean_performance', 0)
                    if mean_perf >= 0.9:
                        rating = "EXCELLENT"
                    elif mean_perf >= 0.7:
                        rating = "GOOD"
                    elif mean_perf >= 0.5:
                        rating = "FAIR"
                    else:
                        rating = "POOR"
                    print(f"  Performance Rating: {rating}")
            
            return self.cross_column_metrics

    @staticmethod
    def build_screening_df(ztrcol_df,
                           columns_to_keep=None,
                           rename_map=None):
        """
        Build and standardize the screening dataframe.
        Returns a new df with HUM, LLM.IS, LLM.DS, and Priority.
        """
        import numpy as np
        import pandas as pd
    
        if columns_to_keep is None:
            columns_to_keep = [
                'title', 'creators', 'publicationYear', 'publicationTitle', 'DOI',
                'archive', 'rights', 'archiveLocation', 'callNumber'
            ]
        if rename_map is None:
            rename_map = {
                'title': 'Title',
                'creators': 'Author(s)',
                'publicationYear': 'YoP',
                'publicationTitle': 'Journal',
                'DOI': 'DOI',
                'archive': 'HUM_NSTD_INI',
                'rights': 'HUM_NSTD_FNL',
                'archiveLocation': 'LLM_IS_NSTD',
                'callNumber': 'LLM_DS_NSTD'
            }
    
        df_analysis = ztrcol_df.copy()
        df_analysis = df_analysis[columns_to_keep].rename(columns=rename_map)
    
        def _standardize_human(value):
            value = str(value).upper().strip()
            if (
                value.startswith('EXCLUDED')
                or 'REFERENCE (PAPER)' in value
                or 'REFERENCE (REVIEW)' in value
                or value.startswith('GLOBAL')
                or 'DATA (GLOBAL)' in value
                or 'PLANTS/SPECIES' in value
                or 'TWN' in value
            ):
                return 'EXCLUDED'
            elif value.startswith('OK'):
                return 'OK'
            elif value.startswith('POSSIBLE'):
                return 'POSSIBLE'
            elif value.startswith('REVIEW'):
                return 'REVIEW'
            else:
                return 'OTHER'
    
        def _standardize_llm(value):
            value = str(value).upper().strip()
            return value if value in ['EXCLUDED', 'POSSIBLE', 'OK', 'REVIEW'] else 'OTHER'
    
        df_analysis['HUM'] = df_analysis['HUM_NSTD_FNL'].apply(_standardize_human)
        df_analysis['LLM.IS'] = df_analysis['LLM_IS_NSTD'].apply(_standardize_llm)
        df_analysis['LLM.DS'] = df_analysis['LLM_DS_NSTD'].apply(_standardize_llm)
    
        # Priority
        for c in ['HUM', 'LLM.IS', 'LLM.DS']:
            df_analysis[c] = df_analysis[c].fillna('').astype(str)
    
        conditions = [
            df_analysis['HUM'].str.startswith('OK'),
            (df_analysis['LLM.IS'].str.startswith('OK')) &
            (df_analysis['HUM'].str.startswith('POSSIBLE')),
        ]
        choices = ['HIGH', 'MEDIUM']
        df_analysis['Priority'] = np.select(conditions, choices, default='LOW')
    
        return df_analysis

    # ------------------------------------------------------------------
    # Screening comparison + plots
    # ------------------------------------------------------------------
    def plot_screening_performance(
        self,
        ztrcol_df=None,
        df_analysis=None,
        *,
        columns_to_keep=None,
        rename_map=None,
        valid_labels=None,
        comparisons=None,
        label_map=None,
        colormaps=None,
        metric_order=None,
        metric_colors=None,
        figsize=(17, 10),
        top_bottom_height_ratios=(1.2, 2.3),
        bar_ylim=(0.0, 1.05),
        show=True,
        verbose=True,
        return_results=True,
    ):
        """
        Build screening df (if needed), run HUM/LLM comparisons,
        and plot bar metrics + confusion matrices.

        Parameters
        ----------
        ztrcol_df : pd.DataFrame, optional
            Raw Zotero collection dataframe.
        df_analysis : pd.DataFrame, optional
            Pre-built screening dataframe (with HUM, LLM.IS, LLM.DS).
        columns_to_keep, rename_map : as in build_screening_df
        valid_labels : list[str]
            Screening labels to treat as valid classes.
        comparisons : list[tuple[str,str]]
            Column pairs to compare, e.g. ('HUM', 'LLM.IS').
        label_map : dict
            Mapping from label to pretty label on axes.
        colormaps : list[str]
            Matplotlib colormaps for each confusion matrix.
        metric_order : list[str]
            Order of bar metrics: ['Precision','Recall','F1','Accuracy'].
        metric_colors : dict
            Color per metric key.
        figsize : tuple
        top_bottom_height_ratios : tuple(float, float)
            Height share for (top row bars, bottom row matrices).
        bar_ylim : tuple(float, float)
        show : bool
            Whether to call plt.show().
        verbose : bool
            Print classification reports and accuracy.
        return_results : bool
            If True, return dict with df, results list, and figure.

        Returns
        -------
        dict or matplotlib.figure.Figure
        """
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Rectangle

        # --------------------------------------------------------------
        # Build or use provided df_analysis
        # --------------------------------------------------------------
        if df_analysis is None:
            if ztrcol_df is None:
                raise ValueError("Provide either 'ztrcol_df' or 'df_analysis'.")
            df_analysis = LLMInsightsEvaluator.build_screening_df(
                ztrcol_df,
                columns_to_keep=columns_to_keep,
                rename_map=rename_map,
            )
        else:
            df_analysis = df_analysis.copy()

        # These are already strings in build_screening_df, but we keep this safe.
        for c in ['HUM', 'LLM.IS', 'LLM.DS']:
            if c in df_analysis.columns:
                df_analysis[c] = df_analysis[c].fillna('').astype(str)

        # --------------------------------------------------------------
        # Defaults for labels, comparisons, colors
        # --------------------------------------------------------------
        if valid_labels is None:
            valid_labels = ['EXCLUDED', 'POSSIBLE', 'OK', 'REVIEW']

        if comparisons is None:
            comparisons = [
                ('HUM', 'LLM.IS'),
                ('HUM', 'LLM.DS'),
                ('LLM.IS', 'LLM.DS'),
            ]

        if label_map is None:
            label_map = {
                'EXCLUDED': 'Excluded',
                'POSSIBLE': 'Possible',
                'OK': 'OK',
                'REVIEW': 'Review'
            }

        if colormaps is None:
            colormaps = ['Blues', 'Greens', 'Purples']

        if metric_order is None:
            metric_order = ['Precision', 'Recall', 'F1', 'Accuracy']

        if metric_colors is None:
            metric_colors = {
                'Precision': '#1f77b4',
                'Recall': '#ff7f0e',
                'F1': '#2ca02c',
                'Accuracy': '#d62728'
            }

        # --------------------------------------------------------------
        # Run comparisons
        # --------------------------------------------------------------
        results = []
        for col1, col2 in comparisons:
            if col1 not in df_analysis.columns or col2 not in df_analysis.columns:
                if verbose:
                    print(f"[screening_performance_plot] Skip {col1} vs {col2}: column missing.")
                continue

            mask = df_analysis[col1].isin(valid_labels) & df_analysis[col2].isin(valid_labels)
            df_filtered = df_analysis[mask].copy()
            n_records = len(df_filtered)

            if n_records == 0:
                if verbose:
                    print(f"[screening_performance_plot] Skip {col1} vs {col2}: no valid rows.")
                continue

            cr_dict = classification_report(
                df_filtered[col1],
                df_filtered[col2],
                labels=valid_labels,
                output_dict=True,
                zero_division=0
            )
            accuracy = accuracy_score(df_filtered[col1], df_filtered[col2])
            cm = confusion_matrix(df_filtered[col1], df_filtered[col2], labels=valid_labels)

            if verbose:
                print(f"\n{col1} vs {col2}: ------------------")
                print("\nCLASSIFICATION REPORT:")
                print(classification_report(
                    df_filtered[col1],
                    df_filtered[col2],
                    labels=valid_labels,
                    zero_division=0
                ))
                print("\nCONFUSION MATRIX:")
                print(cm)
                print(f"\nOverall Accuracy: {accuracy:.2%}")

            results.append({
                'comparison': f'{col1} vs {col2}',
                'n_records': n_records,
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': cr_dict,
                'counts': {
                    col1: df_filtered[col1].value_counts().reindex(valid_labels, fill_value=0).to_dict(),
                    col2: df_filtered[col2].value_counts().reindex(valid_labels, fill_value=0).to_dict()
                }
            })

        if not results:
            if verbose:
                print("[screening_performance_plot] No comparisons could be computed.")
            return None

        # --------------------------------------------------------------
        # Plot
        # --------------------------------------------------------------
        top_ratio, bottom_ratio = top_bottom_height_ratios
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3,
                              height_ratios=[top_ratio, bottom_ratio],
                              hspace=0.08)

        def style_bar_ax(ax):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_linewidth(1.4)
            ax.spines['bottom'].set_color('black')
            ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
            ax.set_axisbelow(True)
            for lab in ax.get_xticklabels():
                lab.set_fontweight('bold')

        for idx, res in enumerate(results):
            # shared title
            shared_title = f"{res['comparison']}\nPapers: {res['n_records']}"

            # ------------------ top row: metric bars ------------------
            ax_bar = fig.add_subplot(gs[0, idx])

            cr = res['classification_report']
            values = [
                cr['macro avg']['precision'],
                cr['macro avg']['recall'],
                cr['macro avg']['f1-score'],
                res['accuracy']
            ]
            palette = [metric_colors[m] for m in metric_order]

            import seaborn as sns
            sns.barplot(
                x=metric_order,
                y=values,
                ax=ax_bar,
                palette=palette
            )
            ax_bar.set_ylim(*bar_ylim)
            ax_bar.set_ylabel('')
            ax_bar.set_xlabel('')
            ax_bar.set_title(shared_title, fontsize=16, fontweight='bold', pad=6)

            # annotate bars
            for p, v in zip(ax_bar.patches, values):
                h = p.get_height()
                ax_bar.annotate(
                    f"{v:.2f}",
                    (p.get_x() + p.get_width() / 2., h),
                    ha='center',
                    va='bottom',
                    fontsize=16,
                    fontweight='bold'
                )

            style_bar_ax(ax_bar)
            ax_bar.tick_params(axis='y', labelsize=14)
            ax_bar.tick_params(axis='x', labelsize=15)

            # ------------------ bottom row: confusion matrix ------------------
            ax = fig.add_subplot(gs[1, idx])

            cm = res['confusion_matrix']
            vmin = 0
            vmax = cm.max()

            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

            cmap = colormaps[idx % len(colormaps)]

            hm = sns.heatmap(
                cm,
                annot=False,
                fmt='d',
                cmap=cmap,
                xticklabels=[label_map.get(l, l) for l in valid_labels],
                yticklabels=[label_map.get(l, l) for l in valid_labels],
                vmin=vmin,
                vmax=vmax,
                cbar=True,
                ax=ax,
                annot_kws={'size': 14, 'weight': 'bold'},
                linewidths=1.5,
                linecolor='white',
                cbar_kws={
                    'orientation': 'horizontal',
                    'location': 'top',
                    'pad': 0.02,
                    'shrink': 1
                }
            )

            # text in cells: count + percent
            line_gap = 0.18

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    count = cm[i, j]
                    pct = cm_percent[i, j]
                    if np.isnan(pct):
                        pct = 0.0

                    # choose text color based on cell color
                    facecolor = hm.collections[0].get_facecolors()[i * cm.shape[1] + j]
                    r, g, b, _ = facecolor
                    luminance = 0.299 * r + 0.587 * g + 0.114 * b
                    txt_color = 'black' if luminance > 0.5 else 'white'

                    y_center = i + 0.5
                    x_center = j + 0.5

                    ax.text(
                        x_center,
                        y_center - line_gap/2,
                        f"{count:d}",
                        ha='center',
                        va='center',
                        fontsize=14,
                        fontweight='bold',
                        color=txt_color
                    )

                    ax.text(
                        x_center,
                        y_center + line_gap/2,
                        f"({pct:.1f}%)",
                        ha='center',
                        va='center',
                        fontsize=12,
                        fontweight='normal',
                        color=txt_color
                    )

            # colorbar ticks
            cbar = hm.collections[0].colorbar
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels([str(vmin), str(vmax)])
            cbar.ax.tick_params(labelsize=14)
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')

            # red diagonal boxes
            inset = 0.03
            for d in range(len(valid_labels)):
                rect = Rectangle(
                    (d + inset, d + inset),
                    1 - 2*inset,
                    1 - 2*inset,
                    edgecolor='red',
                    linewidth=2,
                    fill=False,
                    zorder=10,
                    clip_on=True
                )
                ax.add_patch(rect)

            right_label = res['comparison'].split('vs')[1].strip()
            left_label = res['comparison'].split('vs')[0].strip()

            ax.set_xlabel(right_label, fontsize=14, fontweight='bold')
            ax.set_ylabel(left_label, fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', labelsize=14, rotation=45)
            ax.tick_params(axis='y', labelsize=14, rotation=45)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontweight('bold')

        plt.subplots_adjust(
            left=0.08,
            right=0.980,
            bottom=0.165,
            top=0.915,
            wspace=0.345,
            hspace=0.515
        )

        if show:
            plt.show()
        else:
            plt.close(fig)

        if return_results:
            return {
                "df_analysis": df_analysis,
                "results": results,
                "figure": fig,
            }
        else:
            return fig

    def summarize_plot_stats(
        self,
        columns='all',
        metrics=None,              # now list OR dict per-category (supports "default" or "*")
        categories='all',
        group_set_semantics=True,
        percentage=False,
        ci_level=0.95,
        include_completeness=True,
        level='column_metric',     # 'category' | 'column' | 'column_metric'
        max_cols_per_cat=None,
        print_to_console=True,
        return_df=True,
    ):
        import numpy as np
        import pandas as pd
        from math import sqrt
    
        # --- prep detailed data ---
        if self.df_detailed is None:
            _ = self.get_detailed_results(as_dataframe=True, print_output=False)
        df_det = self.df_detailed.copy()
    
        def _scale(v):
            return v * 100.0 if percentage else v
    
        # --- categories / columns like the plot ---
        available_cols = set(self.dict_results.keys()) if getattr(self, 'dict_results', None) else set(df_det['Column'].unique())
    
        if categories == 'all':
            cat_order = list(self.column_types.keys())
        else:
            cat_order = [categories] if isinstance(categories, str) else list(categories)
    
        # map column -> category
        col_to_cat = {}
        for cat_name, col_list in self.column_types.items():
            for c in col_list:
                if c in available_cols:
                    col_to_cat[c] = cat_name
    
        # ordered columns within selected categories
        ordered_cols = []
        for cat in cat_order:
            if cat not in self.column_types:
                continue
            for c in self.column_types[cat]:
                if c in available_cols:
                    ordered_cols.append(c)
    
        cols_to_plot = ordered_cols if columns == 'all' else [c for c in columns if c in ordered_cols]
        if not cols_to_plot:
            raise ValueError("No valid columns selected after applying categories/columns.")
    
        # ---- resolve metrics config (list/None vs dict) ----
        metrics_is_dict = isinstance(metrics, dict)
        per_cat_allowed = {}
    
        if metrics_is_dict:
            # normalize to lists
            norm = {}
            for k, v in metrics.items():
                if v is None:
                    norm[k] = None
                elif isinstance(v, (list, tuple)):
                    norm[k] = list(v)
                else:
                    norm[k] = [v]
            default_key = 'default' if 'default' in norm else ('*' if '*' in norm else None)
            for cat in cat_order:
                per_cat_allowed[cat] = norm.get(cat, norm.get(default_key, None))
        else:
            if metrics is None:
                allowed_for_all = None
            elif isinstance(metrics, (list, tuple)):
                allowed_for_all = list(metrics)
            else:
                allowed_for_all = [metrics]
            per_cat_allowed = {cat: allowed_for_all for cat in cat_order}
    
        # --- metrics present per column (respect per-category filters) ---
        column_metrics = {}
        # fallback list of numeric metrics in df_det
        possible_mets = set(df_det.columns) - {
            'Paper_Index','Column','Column_Type',
            'Human_Completeness','LLM_Completeness',
            'Completeness_Gap','Both_Missing',
            'LLM_Hallucination','LLM_Missed'
        }
    
        # Build overall metric order (first appearance wins)
        encountered_order = []
        seen = set()
    
        for col in cols_to_plot:
            if getattr(self, 'dict_results', None) and col in self.dict_results:
                ms = list(self.dict_results[col]['stats'].keys())
            else:
                ms = [m for m in possible_mets if pd.api.types.is_numeric_dtype(df_det[m])]
            # apply per-category allowlist if provided
            cat = col_to_cat.get(col, 'unknown')
            allowed = per_cat_allowed.get(cat, None)
            if allowed is not None:
                allowset = set(allowed)
                # preserve ms order while filtering
                ms = [m for m in ms if m in allowset]
            column_metrics[col] = ms
            for m in ms:
                if m not in seen:
                    seen.add(m)
                    encountered_order.append(m)
    
        # final metric list for ordering/printing
        if metrics_is_dict:
            metric_list = encountered_order[:]  # keep discovered order
        else:
            if metrics is None:
                metric_list = sorted({m for ms in column_metrics.values() for m in ms})
            else:
                user_list = list(metrics) if isinstance(metrics, (list, tuple)) else [metrics]
                metric_list = [m for m in user_list if m in seen]
                if not metric_list:  # if user list yielded none, fall back to encountered
                    metric_list = encountered_order[:]
    
        # z for CI
        z_map = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.96, 0.975: 2.2414, 0.99: 2.5758}
        z = z_map.get(ci_level, 1.96)
    
        rows = []
    
        def _outlier_frac(x):
            if x.size < 4:
                return 0.0
            q1 = np.percentile(x, 25)
            q3 = np.percentile(x, 75)
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            return float(np.mean((x < lo) | (x > hi)))
    
        # collect per column/metric
        for col in cols_to_plot:
            cat = col_to_cat.get(col, 'unknown')
    
            if include_completeness:
                _c = df_det[df_det['Column'] == col]
                hc = float(_c['Human_Completeness'].dropna().mean()) if 'Human_Completeness' in _c else np.nan
                lc = float(_c['LLM_Completeness'].dropna().mean())   if 'LLM_Completeness' in _c else np.nan
                gp = float(_c['Completeness_Gap'].dropna().mean())   if 'Completeness_Gap' in _c else np.nan
            else:
                hc = lc = gp = np.nan
    
            for m in column_metrics[col]:
                if m not in df_det.columns:
                    continue
                v = df_det.loc[df_det['Column'] == col, m].astype(float)
                miss_frac = float(v.isna().mean())
                v = v.dropna().values
                if v.size == 0:
                    continue
    
                v = _scale(v)
    
                n = int(v.size)
                mean = float(v.mean())
                std = float(v.std(ddof=1)) if n > 1 else 0.0
                med = float(np.median(v))
                q1 = float(np.percentile(v, 25))
                q3 = float(np.percentile(v, 75))
                iqr = q3 - q1
                vmin = float(v.min())
                vmax = float(v.max())
                se = std / sqrt(n) if n > 1 else 0.0
                ci_low = mean - z * se
                ci_high = mean + z * se
                ofrac = _outlier_frac(v)
    
                rows.append({
                    'Category': cat,
                    'Column': col,
                    'Metric': m,
                    'N': n,
                    'Mean': mean,
                    'Std': std,
                    'Median': med,
                    'Q1': q1,
                    'Q3': q3,
                    'IQR': iqr,
                    'Min': vmin,
                    'Max': vmax,
                    'CI_Low': ci_low,
                    'CI_High': ci_high,
                    'Outlier_Frac': ofrac,
                    'Missing_Frac': miss_frac,
                    'Human_Comp_Mean': _scale(np.array([hc])).item() if include_completeness else np.nan,
                    'LLM_Comp_Mean': _scale(np.array([lc])).item() if include_completeness else np.nan,
                    'Gap_Mean': _scale(np.array([gp])).item() if include_completeness else np.nan,
                })
    
        if not rows:
            df_stats = pd.DataFrame(columns=[
                'Category','Column','Metric','N','Mean','Std','Median','Q1','Q3','IQR',
                'Min','Max','CI_Low','CI_High','Outlier_Frac','Missing_Frac',
                'Human_Comp_Mean','LLM_Comp_Mean','Gap_Mean'
            ])
            if print_to_console:
                print("[summarize_plot_stats] No data for the selection.")
            return df_stats if return_df else None
    
        df_stats = pd.DataFrame(rows)
    
        # ---- printing (ordered by category > column > metric) ----
        if print_to_console:
            def _fmt(x, pct=False):
                if pd.isna(x):
                    return "nan"
                return f"{x:,.2f}" if not pct else f"{x:,.1f}"

            # order rows
            cat_cat = pd.Categorical(df_stats['Category'], categories=cat_order, ordered=True)
            df_stats = df_stats.assign(__cat_order=cat_cat)

            # metric ordering fallback
            if len(metric_list) == 0:
                metric_list = sorted(df_stats['Metric'].unique().tolist())
            df_stats['__met_order'] = pd.Categorical(df_stats['Metric'], categories=metric_list, ordered=True)
            df_stats = df_stats.sort_values(['__cat_order','Column','__met_order']).drop(columns=['__cat_order','__met_order'])

            z = z_map.get(ci_level, 1.96)

            # helper for column/metric lines
            def _print_metric_line(df_cc, m):
                r = df_cc[df_cc['Metric'] == m]
                if r.empty:
                    return
                r = r.iloc[0]
                print(f"      {m:<16s} "
                      f"n={int(r['N'])}  "
                      f"mean={_fmt(r['Mean'], pct=percentage)}  "
                      f"CI{int(ci_level*100)}%=[{_fmt(r['CI_Low'], pct=percentage)}, {_fmt(r['CI_High'], pct=percentage)}]  "
                      f"med={_fmt(r['Median'], pct=percentage)}  IQR=[{_fmt(r['Q1'], pct=percentage)}, {_fmt(r['Q3'], pct=percentage)}]  "
                      f"min={_fmt(r['Min'], pct=percentage)}  max={_fmt(r['Max'], pct=percentage)}  "
                      f"outliers={_fmt(100*r['Outlier_Frac'], pct=True)}%  "
                      f"missing={_fmt(100*r['Missing_Frac'], pct=True)}%")

            # metric group lists (used only for set_based pretty printing)
            exact_set = [
                'jaccard','precision','recall','f1','dice','overlap','tversky',
                'size_hum','size_llm','size_intersection','size_union','hum_only','llm_only'
            ]
            whole_text = [
                'set_text_cosine','set_text_bert_p','set_text_bert_r','set_text_bert_f1',
                'set_text_jaro_winkler','set_text_levenshtein','set_text_ratcliff',
                'set_text_rouge1_f','set_text_rougeL_f'
            ]
            pairwise = [
                'soft_precision','soft_recall','soft_f1','soft_jaccard','soft_dice','soft_overlap',
                'soft_size_llm','soft_size_hum','soft_matched',
                'soft_mean_pair_sim','soft_median_pair_sim'
            ]

            for cat in [c for c in cat_order if (df_stats['Category'] == c).any()]:
                df_c = df_stats[df_stats['Category'] == cat]
                n_cols = df_c['Column'].nunique()

                print("=" * 100)
                print(f"{cat}  |  columns={n_cols}")

                # category summary first
                w = df_c['N'].to_numpy()
                w = np.where(w > 0, w, 0)
                m = df_c['Mean'].to_numpy()
                cat_mean = float(np.average(m, weights=w)) if w.sum() > 0 else float(m.mean())
                pooled_se = float(np.sqrt(np.average((df_c['Std']**2 / df_c['N']).fillna(0.0), weights=w)) if (df_c['N'] > 1).any() else 0.0)
                ciL = cat_mean - z * pooled_se
                ciH = cat_mean + z * pooled_se
                rngL = float(df_c['Min'].min())
                rngH = float(df_c['Max'].max())
                print(f"  Overall mean = { _fmt(cat_mean, pct=percentage) } "
                      f"(CI{int(ci_level*100)}% [{ _fmt(ciL, pct=percentage) }, { _fmt(ciH, pct=percentage) }], "
                      f"range [{ _fmt(rngL, pct=percentage) }, { _fmt(rngH, pct=percentage) }])")

                # then per-column blocks for this category
                if level == 'column':
                    print("  -- per column --")
                    for col in df_c['Column'].unique()[: (max_cols_per_cat or 10**9)]:
                        df_cc = df_c[df_c['Column'] == col]
                        w = df_cc['N'].to_numpy()
                        m = df_cc['Mean'].to_numpy()
                        val = float(np.average(m, weights=w)) if w.sum() > 0 else float(m.mean())
                        of = float(np.average(df_cc['Outlier_Frac'], weights=w)) if w.sum() > 0 else float(df_cc['Outlier_Frac'].mean())
                        miss = float(np.average(df_cc['Missing_Frac'], weights=w)) if w.sum() > 0 else float(df_cc['Missing_Frac'].mean())
                        if include_completeness:
                            hc = float(df_cc['Human_Comp_Mean'].mean())
                            lc = float(df_cc['LLM_Comp_Mean'].mean())
                            gp = float(df_cc['Gap_Mean'].mean())
                            comp_txt = f"; HC={_fmt(hc, pct=percentage)} | LC={_fmt(lc, pct=percentage)} | Gap={_fmt(gp, pct=percentage)}"
                        else:
                            comp_txt = ""
                        print(f"    {col}: mean={_fmt(val, pct=percentage)}"
                              f" | outliers={_fmt(100*of, pct=True)}%"
                              f" | missing={_fmt(100*miss, pct=True)}%{comp_txt}")

                elif level == 'column_metric':
                    print("  -- per column / metric --")
                    for col in df_c['Column'].unique()[: (max_cols_per_cat or 10**9)]:
                        print(f"    {col}:")
                        df_cc = df_c[df_c['Column'] == col]

                        if include_completeness:
                            hc = float(df_cc['Human_Comp_Mean'].mean())
                            lc = float(df_cc['LLM_Comp_Mean'].mean())
                            gp = float(df_cc['Gap_Mean'].mean())
                            print(f"      completeness  HC={_fmt(hc, pct=percentage)}  "
                                  f"LC={_fmt(lc, pct=percentage)}  Gap={_fmt(gp, pct=percentage)}")

                        # honor per-column metric allowlist/order collected earlier
                        ms_for_col = column_metrics.get(col, metric_list)

                        col_cat = col_to_cat.get(col, 'unknown')
                        use_groups = (group_set_semantics and col_cat == 'set_based')

                        if not use_groups:
                            for mname in ms_for_col:
                                _print_metric_line(df_cc, mname)
                            continue

                        # pretty grouping for set_based
                        def present(ms):
                            out = []
                            for mname in ms:
                                if mname in ms_for_col and not df_cc[df_cc['Metric'] == mname].empty:
                                    out.append(mname)
                            return out

                        exact_present = present(exact_set)
                        whole_present = present(whole_text)
                        pair_present  = present(pairwise)
                        other_present = [m for m in ms_for_col
                                         if (m not in exact_set)
                                         and (m not in whole_text)
                                         and (m not in pairwise)
                                         and (not df_cc[df_cc['Metric']==m].empty)]

                        if exact_present:
                            print("      --- Exact (set) ---")
                            for mname in exact_present:
                                _print_metric_line(df_cc, mname)
                        if whole_present:
                            print("      --- Whole-text semantics ---")
                            for mname in whole_present:
                                _print_metric_line(df_cc, mname)
                        if pair_present:
                            print("      --- Pairwise semantics ---")
                            for mname in pair_present:
                                _print_metric_line(df_cc, mname)
                        for mname in other_present:
                            _print_metric_line(df_cc, mname)

        return df_stats if return_df else None

    def _apply_axis_style(self, ax, labels, percentage, y_min, y_max,
                          xtick_rotation, xtick_fontsize, ytick_fontsize,
                          top_line_pad_pts):
    
        # --- labels / sizes ---
        labels = [str(l) for l in labels]
        n = len(labels)
    
        # x-limits keep half-step margin for plotting widths
        left_edge_data, right_edge_data = -0.5, n - 0.5
    
        # --- y-limits (trust caller; only fix degenerate cases) ---
        try:
            y0 = float(y_min)
            y1 = float(y_max)
        except Exception:
            y0, y1 = 0.0, 1.0
        if not (y1 > y0):
            # tiny spread to avoid zero-height axis
            eps = 1e-6
            y1 = y0 + eps
    
        # --- apply limits ---
        ax.set_xlim(left_edge_data, right_edge_data)
        ax.set_ylim(y0, y1)
    
        # --- spines (left + bottom only; shifted) ---
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('outward', 12))
        ax.spines['bottom'].set_position(('outward', 12))
    
        # trim visible length of the spines
        # bottom: from first column center (0) to last center (n-1)
        ax.spines['bottom'].set_bounds(0, max(0, n - 1))
        # left: from current y-min to y-max (use actual limits)
        ax.spines['left'].set_bounds(y0, y1)
    
        # --- X ticks: centers with your labels ---
        ax.xaxis.set_major_locator(FixedLocator(list(range(n))))
        ax.xaxis.set_major_formatter(FixedFormatter(labels))
        ax.tick_params(axis='x', which='major', direction='out', length=6, width=1,
                       top=False, labelsize=xtick_fontsize, labelbottom=True)
    
        # rotation / align
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(xtick_rotation)
            lbl.set_ha('right')
    
        # minor x ticks at the two edge centers (if any)
        if n > 0:
            ax.xaxis.set_minor_locator(FixedLocator([0, n - 1]))
        else:
            ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.tick_params(axis='x', which='minor', direction='out', length=6, width=1, labelbottom=False)
    
        # --- Y ticks ---
        if percentage:
            standard_ticks = [0, 20, 40, 60, 80, 100]
            y_ticks = [t for t in standard_ticks if (t >= y0 and t <= y1)]
            if not y_ticks:
                # fallback: 5 nice ticks within [y0, y1]
                locator = MaxNLocator(nbins=5)
                y_ticks = locator.tick_values(y0, y1)
        else:
            # try to keep nice ticks even for custom ranges like (0.2, 0.5) or (-0.1, 1.1)
            locator = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
            y_ticks = locator.tick_values(y0, y1)
    
        ax.yaxis.set_major_locator(FixedLocator([float(t) for t in y_ticks]))
        ax.tick_params(axis='y', which='major', direction='out', length=6, width=1,
                       right=False, labelsize=ytick_fontsize)
    
        # minor ticks exactly at the edges
        ax.yaxis.set_minor_locator(FixedLocator([y0, y1]))
        ax.tick_params(axis='y', which='minor', direction='out', length=4, width=1,
                       right=False, labelleft=False)
    
        # --- (re)build the top line once ---
        old_top = getattr(ax, "_custom_topaxis", None)
        if old_top is not None:
            try:
                old_top.remove()
            except Exception:
                pass
    
        ax_top = ax.secondary_xaxis('top')
        ax._custom_topaxis = ax_top
    
        # style the top line (no ticks except edges)
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['left'].set_visible(False)
        ax_top.spines['bottom'].set_visible(False)
        ax_top.spines['top'].set_linewidth(ax.spines['bottom'].get_linewidth())
        ax_top.spines['top'].set_position(('outward', float(top_line_pad_pts)))
    
        # keep the top line trimmed to the same data-x segment as bottom
        ax_top.spines['top'].set_bounds(0, max(0, n - 1))
    
        # only edge ticks on the top axis
        if n > 0:
            ax_top.xaxis.set_major_locator(FixedLocator([0, n - 1]))
        else:
            ax_top.xaxis.set_major_locator(FixedLocator([]))
        ax_top.tick_params(axis='x', which='major', direction='in', length=6, width=1, labeltop=False)

    def plot_metric_boxplots_by_category_minimal_improved(
        self,
        columns='all',
        metrics=None,
        categories='all',
        percentage=False,
        figsize=(18, 6),
        wspace=0.15,
        show_outliers=True,
        show_means=True,
        mean_marker_size=8,
        mean_marker_color='red',
        mean_marker_style='o',
        y_lim=None,
        y_lim_by_category=None,     # NEW: dict like {'categorical': (0.2, 0.5)}
        auto_extend_y=True,
        y_padding=0.1,
        xtick_rotation=45,
        xtick_fontsize=10,
        ytick_fontsize=11,
        ylabel='Score',
        add_grid=True,
        top_line_pad_pts=12,
        title_y=1.08,
        # Bars for 'match'
        use_bars_for_match=True,
        bar_alpha=0.7,
        error_bar_color='black',
        error_bar_linewidth=1.5,
        capsize=4,
        bar_width_factor=0.6,
        # NEW: bar edge styling
        bar_edgecolor='black',
        bar_edgewidth=1.2,
        # Raw data dots (only for boxplots)
        show_data=False,
        data_jitter=0.08,
        data_size=18,
        data_marker='o',
        data_alpha=0.7,
        data_edgecolor='black',
        data_linewidth=0.3,
        data_color=None,
        # Outlier styling (boxplot fliers)
        flier_marker='o',
        flier_size=6,
        flier_facecolor='none',
        flier_edgecolor='black',
        flier_linewidth=0.8,
        flier_alpha=0.9,
        # Violin underlay (only for boxplots)
        show_violin=False,
        violin_alpha=0.25,
        violin_edgecolor='black',
        violin_linewidth=0.8,
        violin_width=0.95,
        violin_facecolor=None,
        violin_showmeans=False,
        violin_showmedians=False,
        violin_showextrema=False,
        # Box styling
        box_edgecolor='black',
        box_linewidth=1.2,
        box_face_alpha=0.7,
        # NEW: box width factor and line colors
        box_width_factor=0.8,            # <- width of each box as a fraction of slot width
        whisker_color='black',
        whisker_linewidth=1.2,
        cap_color='black',
        cap_linewidth=1.2,
        median_color='black',
        median_linewidth=1.4,
        mean_edgecolor='black',
        # Custom colors per metric
        metric_color_map=None,
        # PRINT STATS
        print_stats=False, stats_level='column_metric',   # 'category' | 'column' | 'column_metric'
        stats_ci=0.95, stats_include_completeness=True, stats_max_cols_per_cat=None, return_stats_df=False,
    ):
        """
        Bars with error for 'match' (if enabled).
        Others: optional violin -> boxplot -> raw dots. Draw order is:
        violin, box, outliers, dots, means.
        If `metrics` is a dict: choose metrics per category. Keys are category names
        used in `self.column_types`. A fallback key "default" or "*" is used when
        a category key is missing. If `metrics` is a list or None, old behavior.
        Bars with error for 'match' (if enabled); others are violin/box/dots.
        """
        if not hasattr(self, 'dict_results') or not self.dict_results:
            self.evaluate_all()
    
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    
        metric_aliases = {
            'f1': 'F1', 'precision': 'Precision', 'recall': 'Recall',
            'match': 'Match', 'bert_f1': 'BERT F1', 'cosine': 'Cosine',
            'jaccard': 'Jaccard', 
            'soft_f1': 'BERT F1 (Pairwise)', 'set_text_bert_f1': 'BERT F1 (Whole Set)' 
        }
        cat_label_map = {
            'set_based': 'Set-based', 'free_text': 'Free text',
            'categorical': 'Categorical', 'temporal': 'Temporal', 'boolean': 'Boolean',
        }
    
        # Map columns -> category
        available_cols = list(self.dict_results.keys())
        col_to_cat = {}
        for cat_name, col_list in self.column_types.items():
            for c in col_list:
                if c in available_cols:
                    col_to_cat[c] = cat_name
    
        # Category order
        if categories == 'all':
            cat_order = list(self.column_types.keys())
        else:
            cat_order = [categories] if isinstance(categories, str) else list(categories)
    
        # Columns to plot in that order
        ordered_cols = []
        for cat in cat_order:
            if cat not in self.column_types:
                continue
            for c in self.column_types[cat]:
                if c in available_cols:
                    ordered_cols.append(c)
    
        cols_to_plot = ordered_cols if columns == 'all' else [c for c in columns if c in ordered_cols]
        if not cols_to_plot:
            raise ValueError("No valid columns selected after applying categories/columns.")
    
        # Keep only categories that actually have columns
        cats_with_columns, per_cat_cols = [], {}
        for cat in cat_order:
            cat_cols = [c for c in cols_to_plot if col_to_cat.get(c) == cat]
            if cat_cols:
                cats_with_columns.append(cat)
                per_cat_cols[cat] = cat_cols
        if not cats_with_columns:
            raise ValueError("No categories contain the selected columns.")
    
        # ---- NEW: resolve metrics configuration ----
        metrics_is_dict = isinstance(metrics, dict)
        default_key = None
        per_cat_metric_filter = {}
    
        if metrics_is_dict:
            # Prefer explicit default keys if present
            if 'default' in metrics:
                default_key = 'default'
            elif '*' in metrics:
                default_key = '*'
            # Normalize values to lists
            tmp = {}
            for k, v in metrics.items():
                if v is None:
                    tmp[k] = None
                elif isinstance(v, (list, tuple)):
                    tmp[k] = list(v)
                else:
                    tmp[k] = [v]
            metrics = tmp
            for cat in cats_with_columns:
                per_cat_metric_filter[cat] = metrics.get(cat, metrics.get(default_key, None))
        else:
            # list or None -> same for all categories
            if metrics is None:
                per_cat_metric_filter = {cat: None for cat in cats_with_columns}  # auto
            elif isinstance(metrics, (list, tuple)):
                per_cat_metric_filter = {cat: list(metrics) for cat in cats_with_columns}
            else:
                per_cat_metric_filter = {cat: [metrics] for cat in cats_with_columns}
    
        # Determine which metrics exist per column, then apply per-category filter
        column_metrics = {}
        encountered_metrics = []  # keep insertion order for palette/legend
        seen = set()
    
        for col in cols_to_plot:
            data = self.dict_results[col]
            col_mets = list(data['stats'].keys())
            cat = col_to_cat[col]
            allowed = per_cat_metric_filter.get(cat, None)
            if allowed is not None:
                col_mets = [m for m in col_mets if m in allowed]
            # Keep order stable and remember which ones appear
            column_metrics[col] = col_mets
            for m in col_mets:
                if m not in seen:
                    seen.add(m)
                    encountered_metrics.append(m)
    
        # If nothing encountered (e.g., filters too strict), try falling back to auto for all
        if not encountered_metrics:
            for col in cols_to_plot:
                col_mets = list(self.dict_results[col]['stats'].keys())
                column_metrics[col] = col_mets
                for m in col_mets:
                    if m not in seen:
                        seen.add(m)
                        encountered_metrics.append(m)
    
        # Compute global y-lims
        df_detailed = self.get_detailed_results(as_dataframe=True, print_output=False)
        all_plot_vals, all_whisker_vals = [], []
        for col in cols_to_plot:
            for metric in column_metrics[col]:
                vals = df_detailed[df_detailed['Column'] == col][metric].dropna().values
                if vals.size == 0:
                    continue
                vals = vals * 100 if percentage else vals
                all_plot_vals.extend(vals.tolist())
                q1 = np.percentile(vals, 25)
                q3 = np.percentile(vals, 75)
                iqr = q3 - q1
                lw = max(np.min(vals), q1 - 1.5 * iqr)
                uw = min(np.max(vals), q3 + 1.5 * iqr)
                all_whisker_vals.extend([lw, uw])
    
        if y_lim is not None:
            y_min, y_max = y_lim
        else:
            if len(all_plot_vals) == 0:
                y_min, y_max = (0, 100) if percentage else (0, 1.0)
            else:
                if auto_extend_y and len(all_whisker_vals) > 0:
                    data_min = float(np.min(all_whisker_vals))
                    data_max = float(np.max(all_whisker_vals))
                else:
                    data_min = float(np.min(all_plot_vals))
                    data_max = float(np.max(all_plot_vals))
                rng = max(1e-9, data_max - data_min)
                pad = rng * y_padding
                y_min = data_min - pad
                y_max = data_max + pad
                if percentage:
                    y_min = max(-5, y_min)
                    y_max = max(105, y_max) if data_max > 100 else min(105, y_max)
                else:
                    y_min = max(-0.05, y_min)
                    y_max = max(1.05, y_max) if data_max > 1.0 else min(1.05, y_max)
    
        # Figure + gridspec
        n_cats = len(cats_with_columns)
        width_ratios = [max(1, len(per_cat_cols[cat])) for cat in cats_with_columns]
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        gs = fig.add_gridspec(nrows=1, ncols=n_cats, width_ratios=width_ratios, wspace=wspace)
    
        # Colors: respect user map, then assign defaults in the order encountered
        default_colors = plt.cm.turbo(np.linspace(0, 1, max(1, len(encountered_metrics))))
        metric_colors = {}
        if metric_color_map:
            metric_colors.update(metric_color_map)
        ci = 0
        for m in encountered_metrics:
            if m not in metric_colors:
                metric_colors[m] = default_colors[ci % len(default_colors)]
                ci += 1
    
        # zorders
        z_violin = 1.4
        z_box = 2.0
        z_outliers = 2.4
        z_dots = 2.6
        z_means = 3.2
    
        legend_handles, legend_labels, legend_added = [], [], set()
        axes_dict = {}
    
        flierprops = dict(
            marker=flier_marker,
            markersize=flier_size,
            markerfacecolor=flier_facecolor,
            markeredgecolor=flier_edgecolor,
            markeredgewidth=flier_linewidth,
            alpha=flier_alpha,
            zorder=z_outliers,
        )
    
        for i, cat in enumerate(cats_with_columns):
            ax = fig.add_subplot(gs[0, i])
            axes_dict[cat] = ax
            cat_cols = per_cat_cols[cat]
    
            if add_grid:
                ax.grid(True, axis='y', alpha=0.3, zorder=0)
                ax.set_axisbelow(True)
    
            if i == 0:
                ax.set_ylabel(f"{'Score (%)' if percentage else (ylabel or 'Score')}", fontsize=11)
            else:
                ax.set_ylabel(None)
    
            # columns inside category
            for j, col in enumerate(cat_cols):
                col_mets = column_metrics[col]
                if not col_mets:
                    continue
    
                n_m = len(col_mets)
                total_width = 0.7
                bw = total_width / max(1, n_m)
                base_x = j
    
                for k, m in enumerate(col_mets):
                    vals = df_detailed[df_detailed['Column'] == col][m].dropna().values
                    if vals.size == 0:
                        continue
                    vals = vals * 100 if percentage else vals
                    x_pos = base_x + (k - (n_m - 1) / 2.0) * bw
                    c = metric_colors[m]
    
                    # bars for 'match'
                    if m == 'match' and use_bars_for_match:
                        mean_val = float(np.mean(vals))
                        std_val = float(np.std(vals))
                        sem_val = std_val / np.sqrt(len(vals))
                        ax.bar(
                            x_pos, mean_val,
                            width=bw * bar_width_factor,
                            color=c, alpha=bar_alpha,
                            edgecolor=bar_edgecolor, linewidth=bar_edgewidth,
                            yerr=sem_val,
                            error_kw=dict(
                                ecolor=error_bar_color,
                                elinewidth=error_bar_linewidth,
                                capsize=capsize,
                                capthick=error_bar_linewidth
                            ),
                            zorder=z_box,
                        )
                        if show_means:
                            ax.plot(
                                x_pos, mean_val,
                                marker=mean_marker_style,
                                markerfacecolor=mean_marker_color,
                                markeredgecolor=mean_edgecolor,
                                markersize=mean_marker_size,
                                zorder=z_means
                            )
                        if m not in legend_added:
                            legend_handles.append(Rectangle((0, 0), 1, 1,
                                                            facecolor=c, alpha=bar_alpha,
                                                            edgecolor=bar_edgecolor, linewidth=bar_edgewidth))
                            legend_labels.append(metric_aliases.get(m, m))
                            legend_added.add(m)
                        continue
    
                    # violin
                    if show_violin and vals.size > 1:
                        vp = ax.violinplot(
                            vals,
                            positions=[x_pos],
                            widths=bw * violin_width,
                            showmeans=violin_showmeans,
                            showmedians=violin_showmedians,
                            showextrema=violin_showextrema,
                        )
                        for body in vp['bodies']:
                            body.set_facecolor(c if violin_facecolor is None else violin_facecolor)
                            body.set_edgecolor(violin_edgecolor)
                            body.set_linewidth(violin_linewidth)
                            body.set_alpha(violin_alpha)
                            body.set_zorder(z_violin)
                        for key in ('cmeans', 'cmedians', 'cmaxes', 'cmins', 'cbars'):
                            if key in vp:
                                vp[key].set_color(violin_edgecolor)
                                vp[key].set_linewidth(violin_linewidth)
                                vp[key].set_alpha(violin_alpha)
                                vp[key].set_zorder(z_violin + 0.05)
    
                    # boxplot
                    bp = ax.boxplot(
                        vals,
                        positions=[x_pos],
                        widths=bw * box_width_factor,
                        patch_artist=True,
                        showfliers=show_outliers,
                        showmeans=show_means,
                        meanprops={
                            'marker': mean_marker_style,
                            'markerfacecolor': mean_marker_color,
                            'markeredgecolor': mean_edgecolor,
                            'markersize': mean_marker_size,
                            'zorder': z_means,
                        },
                        flierprops=flierprops,
                        zorder=z_box
                    )
                    box = bp['boxes'][0]
                    box.set_facecolor(c)
                    box.set_alpha(box_face_alpha)
                    box.set_edgecolor(box_edgecolor)
                    box.set_linewidth(box_linewidth)
    
                    for ln in bp['whiskers']:
                        ln.set_color(whisker_color)
                        ln.set_linewidth(whisker_linewidth)
                        ln.set_zorder(z_box + 0.05)
                    for ln in bp['caps']:
                        ln.set_color(cap_color)
                        ln.set_linewidth(cap_linewidth)
                        ln.set_zorder(z_box + 0.05)
                    for ln in bp['medians']:
                        ln.set_color(median_color)
                        ln.set_linewidth(median_linewidth)
                        ln.set_zorder(z_box + 0.05)
    
                    if show_data and vals.size > 0:
                        jitter = np.random.uniform(-data_jitter, data_jitter, size=vals.size)
                        x_data = x_pos + jitter
                        ax.scatter(
                            x_data, vals,
                            s=data_size,
                            marker=data_marker,
                            facecolors=(c if data_color is None else data_color),
                            edgecolors=data_edgecolor,
                            linewidths=data_linewidth,
                            alpha=data_alpha,
                            zorder=z_dots,
                            clip_on=False
                        )
    
                    if m not in legend_added:
                        legend_handles.append(box)
                        legend_labels.append(metric_aliases.get(m, m))
                        legend_added.add(m)
    
            # y-lims per category or global
            if y_lim_by_category and cat in y_lim_by_category:
                y_min_cat, y_max_cat = y_lim_by_category[cat]
            else:
                y_min_cat, y_max_cat = y_min, y_max
    
            self._apply_axis_style(
                ax=ax,
                labels=cat_cols,
                percentage=percentage,
                y_min=y_min_cat, y_max=y_max_cat,
                xtick_rotation=xtick_rotation,
                xtick_fontsize=xtick_fontsize,
                ytick_fontsize=ytick_fontsize,
                top_line_pad_pts=top_line_pad_pts,
            )
    
            ax.text(
                0.5, float(title_y), cat_label_map.get(cat, cat),
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10,
            )
    
        if legend_handles:
            fig.legend(
                legend_handles, legend_labels,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.001),
                framealpha=0, fontsize=9,
                ncol=min(8, len(legend_handles))
            )
    
        fig.subplots_adjust(left=0.095, right=0.975, bottom=0.385, top=0.810, wspace=wspace)
        plt.tight_layout()
    
        # stats (respects the same columns/metrics/cats you passed)
        stats_df = None
        if print_stats or return_stats_df:
            # Build the effective metrics argument to pass to summarize_plot_stats
            # For dict input we pass through the same dict, else the list/None.
            eff_metrics = metrics if metrics_is_dict else per_cat_metric_filter[cats_with_columns[0]]
            stats_df = self.summarize_plot_stats(
                columns=columns,
                metrics=metrics,  # pass the original (supports dict downstream if you add it there)
                categories=categories,
                percentage=percentage,
                ci_level=stats_ci,
                include_completeness=stats_include_completeness,
                level=stats_level,
                max_cols_per_cat=stats_max_cols_per_cat,
                print_to_console=print_stats,
                return_df=True,
            )
    
        if return_stats_df:
            return fig, axes_dict, stats_df
        return fig, axes_dict
