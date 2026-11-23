import pathlib
import re
from typing import List, Union, Optional, Literal

try:
    import pymupdf4llm
    _PYMUPDF4LLM_IMPORT_ERROR = None
except Exception as exc:
    # PyMuPDF contains a native extension that Windows application-control
    # policies may block. Keep PDF preparation usable via the pure-Python
    # pypdf fallback below without weakening the machine's security policy.
    pymupdf4llm = None
    _PYMUPDF4LLM_IMPORT_ERROR = exc

class AcademicPDFProcessor:
    """PDF processor with explicit raw/cleaned text handling and separate saving."""
    
    def __init__(self, pdf_path: Union[str, pathlib.Path]):
        self.pdf_path = pathlib.Path(pdf_path)
        self._raw_text = ""
        self._truncated_text = ""
        self._cleaned_text = ""
        self.extraction_backend = ""
        self._validate_pdf()
        
        
    # Built-in section variants dictionary (auto-expandable)
    SECTION_VARIANTS = {
        "methodology": [r"Methodology", r"Experimental Design", r"Methods?", r"Approach", r"Research Design"],
                       # r"^\d+\.\s*Methods?"],  # Numbered sections like "3. Method"          
        "discussion": [r"Discussion", r"Analysis", r"Results and Discussion", r"Findings"],
                       # r"^\d+\.\s*Discussion"],
        "acknowledgments": [r"Acknowledg\w+",  # Matches Acknowledgments/Acknowledgements
                            r"Thanks", r"Credits?", r"Declaration"],
        "references": [r"Reference", r"References", r"References Cited", r"Reference List", r"References and Further Reading",
                       r"Literature References", r"Bibliographical References", r"List of References", r"References and Notes", 
                       r"Notes and References", r"Cited References", r"Additional References", r"Reference Section", 
                       r"Referenced Materials",
                       r"Bibliography", r"Bibliographic References", r"Bibliography and References", r"Annotated Bibliography", 
                       r"Selected Bibliography", 
                       r"Works Cited", r"Cited Works", r"Works Referenced", r"Quoted Works", r"Works Quoted", r"Works Consulted",
                       r"List of Works Cited", 
                       r"Citations", r"Citations and Notes",
                       r"Literature Cited", r"Literature", r"Cited Literature", r"Cited Material", r"Documents Cited",
                       r"Sources", r"Sources Cited", r"Cited Sources", r"Sources Consulted",  
                       r"Further Reading", r"Endnotes", r"Footnotes"],
        "competing interest": [r"Competing Interests?", r"Conflict of Interest", r"Declaration"],
        "credit": [r"CRediT", r"Author Contributions", r"Authorship Statement"]
    } 

    def _validate_pdf(self):
        if not self.pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {self.pdf_path}")

    def extract_text(
        self,
        method: str = "markdown",
        chunk_size: Optional[int] = None,
    ) -> str:
        """
        Extract raw text without cleaning
        - method: 'markdown' (full text) or 'llama' (chunked)
        - chunk_size: Only used for 'llama' method
        - save_raw: Save unprocessed extracted text
        """
        if method == "llama":
            self._raw_text = self._process_llama(chunk_size)
        elif method == "markdown":
            self._raw_text = self._process_markdown()
        else:
            raise ValueError(f"Invalid extraction method: {method}")
            
        return self._raw_text

    def _process_llama(self, chunk_size: Optional[int]) -> str:
        """Llama-style extraction without any cleaning"""
        if pymupdf4llm is not None:
            try:
                if chunk_size:
                    docs = pymupdf4llm.LlamaMarkdownReader().load_data(
                        self.pdf_path, chunk_size=chunk_size
                    )
                    text = "\n\n".join(
                        f"### CHUNK {i}\n{doc.text}"
                        for i, doc in enumerate(docs)
                    )
                else:
                    text = pymupdf4llm.to_markdown(self.pdf_path)
                self.extraction_backend = "pymupdf4llm"
                return text
            except Exception as exc:
                primary_error = exc
        else:
            primary_error = _PYMUPDF4LLM_IMPORT_ERROR
        return self._process_pypdf(primary_error)

    def _process_markdown(self) -> str:
        """Raw markdown extraction without cleaning"""
        if pymupdf4llm is not None:
            try:
                text = pymupdf4llm.to_markdown(self.pdf_path)
                self.extraction_backend = "pymupdf4llm"
                return text
            except Exception as exc:
                primary_error = exc
        else:
            primary_error = _PYMUPDF4LLM_IMPORT_ERROR
        return self._process_pypdf(primary_error)

    def _process_pypdf(self, primary_error: Exception = None) -> str:
        """Extract directly from the PDF with a Windows-safe fallback."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(self.pdf_path))
            text = "\n\n".join(
                page_text
                for page in reader.pages
                if (page_text := (page.extract_text() or "").strip())
            )
            self.extraction_backend = "pypdf"
            return text
        except Exception as fallback_error:
            if primary_error is None:
                raise RuntimeError(
                    f"PDF extraction failed with pypdf: {fallback_error}"
                ) from fallback_error
            raise RuntimeError(
                "PDF extraction failed with both pymupdf4llm "
                f"({primary_error}) and pypdf ({fallback_error})"
            ) from fallback_error
        
    def get_processed_text(self, version='cleaned'):
        return {
            'raw': self._raw_text,
            'truncated': self._truncated_text,
            'cleaned': self._cleaned_text
        }.get(version, self._cleaned_text)

#%%

    def clean_text( # ISSUE: if "reference" is mentioned within the text, big part of the text is deleted!
        self,
        remove_refs: bool = True,
        remove_urls: bool = True,
        remove_empty_lines: bool = True
    ) -> str:
        text = self._raw_text
    
        if remove_refs:
            # Combine all reference variants into one regex pattern
            ref_variants = "|".join(self.SECTION_VARIANTS["references"])
            # Collect patterns from other sections to detect next section
            other_sections = []
            for section in self.SECTION_VARIANTS:
                if section != "references":
                    other_sections.extend(self.SECTION_VARIANTS[section])
            other_section_pattern = "|".join(other_sections)
            
            # Updated regex pattern to handle different header formats
            ref_pattern = re.compile(
                fr'(?i)(\n\s*)(\d+[.\s]*)?(\*\*)?(?:{ref_variants})(\*\*)?\b.*?(?=(\n\s*(\d+[.\s]*)?(\*\*)?(?:{other_section_pattern})(\*\*)?\b|$))',
                re.DOTALL
            )
            text = ref_pattern.sub('\n', text)
    
        if remove_urls:
            text = re.sub(r'https?://\S+', '', text)
    
        if remove_empty_lines:
            text = re.sub(r'\n\s*\n', '\n\n', text).strip()
                
        text = text.strip() # Remove leading and trailing whitespace
        words = text.split()
        text = ' '.join(words)
        
        self._cleaned_text = text
        
        return self._cleaned_text

#%%

    def truncate_text(self, word_count: int) -> str:
        """
        Truncate text to a specified word count from either raw or cleaned text.

        Parameters:
            word_count (int): Maximum number of words to retain.
            source (Literal['raw', 'cleaned']): Which version of the text to trim.

        Returns:
            str: The truncated text (either raw or cleaned).

        Raises:
            ValueError: If word_count is not a positive integer.
            ValueError: If source is 'cleaned' but no cleaned text exists.
        """
        if word_count <= 0:
            raise ValueError("word_count must be a positive integer")

        # Remove leading and trailing whitespace
        text = self._raw_text.strip()
    
        # Split into words
        words = text.split()
    
        # # If the text has fewer or equal words than requested, return it as-is
        # if len(words) <= word_count:
        #     self._truncated_text = text
        #     return text
    
        # Truncate to desired word count
        truncated = ' '.join(words[:word_count])
        self._truncated_text = truncated
        return truncated
    
#%%
    def save(
        self,
        output_dir: Union[str, pathlib.Path, None] = None,
        format: str = "md",
        save_raw: bool = False,
        save_truncated: bool = False,
        save_cleaned: bool = False,
        raw_suffix: str = "raw",
        truncated_suffix: str = "trn",
        cleaned_suffix: str = "cln"
    ) -> None:
        """
        Unified save method with flexible output options
        
        Args:
            output_dir: Directory to save files (None = same as PDF directory)
            format: File format - 'md' or 'txt'
            save_raw: Save raw extracted text
            save_cleaned: Save cleaned text
            raw_suffix: Suffix for raw files
            cleaned_suffix: Suffix for cleaned files
        """
        # Set output directory
        final_dir = self.pdf_path.parent if output_dir is None else pathlib.Path(output_dir)
        final_dir.mkdir(exist_ok=True, parents=True)
        
        stem = self.pdf_path.stem
        
        # Save requested versions
        if save_raw:
            (final_dir / f"{stem}_{raw_suffix}.{format}").write_text(
                self._raw_text, encoding="utf-8")
            
        if save_truncated:
            (final_dir / f"{stem}_{truncated_suffix}.{format}").write_text(
                self._truncated_text, encoding="utf-8")
        
        if save_cleaned:
            (final_dir / f"{stem}_{cleaned_suffix}.{format}").write_text(
                self._cleaned_text, encoding="utf-8")

#%%

    @classmethod
    def process_batch(
        cls,
        pdf_dir: pathlib.Path,
        extract_kwargs: dict = None,
        truncate_kwargs: dict = None,
        clean_kwargs: dict = None,
        save_kwargs: dict = None
    ):
        """
        Batch process PDFs with consistent single-doc parameters
        
        Example usage:
        AcademicPDFProcessor.process_batch(
            pdf_dir=pathlib.Path("data/pdfs"),
            extract_kwargs={
                "method": "markdown",
                "chunk_size": None
            },
            truncate_kwargs={
                "word_count": 10000
            },
            clean_kwargs={
                "remove_sections": ["Appendix", "Acknowledgements"],
                "remove_refs": True,
                "remove_urls": True,
                "remove_empty_lines": True
            },
            save_kwargs={
                "output_dir": None,
                "format": "md",
                "save_raw": True,
                "save_cleaned": True,
                "raw_suffix": "raw",
                "cleaned_suffix": "cln"
            }
        )
        """
        extract_kwargs = extract_kwargs or {}
        clean_kwargs = clean_kwargs or {}
        save_kwargs = save_kwargs or {}

        for pdf_path in pdf_dir.glob("*.pdf"):
            processor = cls(pdf_path)
            processor.extract_text(**extract_kwargs)
            processor.truncate_text(**truncate_kwargs)
            processor.clean_text(**clean_kwargs)
            processor.save(**save_kwargs)
            
#%%

           
            
