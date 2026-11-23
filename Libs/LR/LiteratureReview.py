import os
from pathlib import Path
import pandas as pd
import csv
from tqdm import tqdm
import time
import requests
import re
import bibtexparser
import json
from rdflib import Graph  # For parsing RDF files
import PyPDF2  # For reading PDF files
import pdfplumber
from openai import OpenAI
## *****************************************************
from Libs.LLM.LLM import LLM
## *****************************************************

class LiteratureReview:
    def __init__(self, project_id, llm, zotero_collection=None, input_file_path=None,
                 output_dir="output", prompt_file="prompts.json"):
        
        self.project_id = project_id
        self.llm = llm
        self.zotero_collection = zotero_collection
        self.input_file_path = input_file_path
        if input_file_path: self.input_dir = os.path.dirname(os.path.abspath(input_file_path))
        # if input_file_path: self.input_dir = Path(self.input_dir).resolve()
        self.output_dir = output_dir
        self.prompt_file = prompt_file
        self.entries = []
        self.processed_titles = set()
        
        # Initialization 
        os.makedirs(self.output_dir, exist_ok=True) # Ensure output directory exists
        # Load references based on input type
        if self.input_file_path:
            self.load_references()
        elif self.zotero_collection:
            self.load_zotero_references()
        else:
            raise ValueError("Either input_file_path or zotero_collection must be provided.")
        self.prompts = self.load_prompts() # Load prompts

    # ---------------------------------------------------------------------------------------------------------------------------
    def load_zotero_references(self):
        """Load entries from a Zotero collection."""
        print("Loading references from Zotero collection...")
        self.entries = self.zotero_collection.get_items_as_dict()
        print(f"Loaded {len(self.entries)} entries from Zotero collection.")
    
    def load_references(self):
        """Load entries from the input file (BibTeX or Zotero RDF)."""
        if self.input_file_path.endswith(".bib"):
            self.load_bibtex()
        elif self.input_file_path.endswith(".rdf"):
            self.load_zotero_rdf()
        else:
            raise ValueError("Unsupported file format. Use .bib or .rdf.")
        
    def load_bibtex(self):
        """Load entries from the BibTeX file."""
        with open(self.input_file_path, encoding='utf-8') as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)
        self.entries = bib_database.entries
        print(f"Lierature Review >> Topic: {self.project_id} ** Papers: {len(self.entries)} Loaded ** LLM: {self.llm.model}")

    ### RDF NOT WORKING PROPERLY!!!
    def load_zotero_rdf(self):
        """Load entries from the Zotero RDF file."""
        graph = Graph()
        try:
            graph.parse(self.input_file_path, format='xml')
        except Exception as e:
            print(f"Error parsing RDF file: {e}")
            return
    
        # Extract bibliographic information from RDF triples
        self.entries = []
        for subject, predicate, obj in graph:
            entry = {}
            try:
                if "title" in predicate:
                    entry["title"] = str(obj)
                if "author" in predicate:
                    entry["author"] = str(obj)
                if "year" in predicate:
                    entry["year"] = str(obj)
                if "abstract" in predicate:
                    entry["abstract"] = str(obj)
                if entry:
                    self.entries.append(entry)
            except Exception as e:
                print(f"Error processing RDF triple ({subject}, {predicate}, {obj}): {e}")
    
        print(f"Loaded {len(self.entries)} entries from Zotero RDF file.")

    def fix_zotero_rdf(input_path, output_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            rdf_content = f.read()
        
        # Replace <rdf:resource ... /> with a valid element (e.g., <z:file ... />)
        fixed_rdf = re.sub(
            r'<rdf:resource rdf:resource="([^"]+)"\s*/>',
            r'<z:file rdf:resource="\1" />',
            rdf_content
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(fixed_rdf)
        
    # ---------------------------------------------------------------------------------------------------------------------------
    def load_prompts(self):
        """Load prompts from the external file."""
        if self.prompt_file.endswith(".json"):
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                return json.load(f)
        elif self.prompt_file.endswith(".txt"):
            prompts = {}
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                content = f.read()  # Read the entire file content
                # Split the content into blocks based on the 'key = """' pattern
                blocks = re.split(r"(\w+)\s*=\s*\"\"\"", content)
                for i in range(1, len(blocks), 2):  # Iterate over keys and values
                    key = blocks[i].strip()
                    # Extract the value (everything until the next '"""')
                    value = blocks[i + 1].split('"""', 1)[0].strip()
                    prompts[key] = value
            return prompts
        else:
            raise ValueError("Unsupported prompt file format. Use .json or .txt.")

    # ---------------------------------------------------------------------------------------------------------------------------
    def initialize_csv(self, fieldnames, operation_type, rewrite=False):
        """Initialize the CSV file for saving results with a specific operation type."""
        
        endpoint = 'API' if self.llm.use_api else 'LCL' # Whether local server or through API
        
        csv_file_path = os.path.join(self.output_dir, 
                                     f"{endpoint}_{self.project_id}_{operation_type}_{self.llm.model.upper()}.csv")
        
        if os.path.exists(csv_file_path):
            if rewrite:
                print(f"Rewriting existing CSV file: {csv_file_path}")
                with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
            else:
                print(f"CSV file already exists: {csv_file_path}. Skipping rewrite.")
                with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        self.processed_titles.add(row['title'].strip())
        else:
            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        return csv_file_path
    
    # ---------------------------------------------------------------------------------------------------------------------------
    def clean_csv_file(self, csv_file_path):
        """
        Clean the CSV file by removing rows where:
        - The verdict is 'UNDECIDED' and full_response is empty,
        - But only if there exists another row for the same title with a valid verdict and non-empty full_response.
        """
        df = pd.read_csv(csv_file_path)
    
        # Group by title to identify papers with multiple rows
        grouped = df.groupby('title')
    
        # Create a mask to filter out invalid rows
        def filter_group(group):
            # Check if there's at least one valid row in the group
            has_valid_row = group[(group['verdict'] != 'UNDECIDED') & (group['full_response'].notna())].shape[0] > 0
    
            if has_valid_row:
                # Keep rows that are valid or have non-empty full_response
                return group[(group['verdict'] != 'UNDECIDED') | (group['full_response'].notna())]
            else:
                # If no valid row exists, keep all rows
                return group
    
        # Apply the filter to each group
        cleaned_df = grouped.apply(filter_group).reset_index(drop=True)
    
        # Save the cleaned DataFrame back to the CSV file
        cleaned_df.to_csv(csv_file_path, index=False, encoding='utf-8')
        print(f"\nCleaned CSV file saved to: {csv_file_path}")
       

    # ---------------------------------------------------------------------------------------------------------------------------
    def extract_final_category(self, llm_response): # INITIAL SCREENING
        """Extract the final category from the LLM response."""
        for line in reversed(llm_response.split('\n')):
            line = line.strip()
            if re.search(r'final\s+categorization\s*:', line, re.IGNORECASE):
                category_part = line.split(':', 1)[-1].strip()
                category = re.sub(r'[^a-zA-Z!]', '', category_part).upper()
                return category
        return None
    
    # ---------------------------------------------------------------------------------------------------------------------------
    def extract_final_category_and_justification(self, llm_response): # DEEP SCREENING
        """Extract the final category and justification from the LLM response."""
        # Use regex to find the final categorization and justification in the entire response
        match = re.search(
            r'Final\s+categorization\s*:\s*([^()]*?)\s*\(\s*Justification:\s*(.*)\s*\)',
            llm_response,
            re.IGNORECASE | re.DOTALL
        )
        if match:
            category = match.group(1).strip()
            justification = match.group(2).strip()
            # Clean the category by removing non-alphabetic characters except '!' and convert to uppercase
            category = re.sub(r'[^a-zA-Z!]', '', category).upper()
            return category, justification
        return None, None
    
    # def extract_final_category_and_justification(self, llm_response): # DEEP SCREENING (issue: only inline response not list!)
    #     """Extract the final category and justification from the LLM response."""
    #     for line in reversed(llm_response.split('\n')):
    #         line = line.strip()
    #         if re.search(r'final\s+categorization\s*:', line, re.IGNORECASE):
    #             category_part = line.split(':', 1)[-1].strip()
    #             # Use regex to separate category and justification
    #             match = re.match(
    #                 r'^\s*([^()]*?)\s*\(\s*Justification:\s*(.*)\)\s*$',
    #                 category_part,
    #                 re.IGNORECASE
    #             )
    #             if match:
    #                 category = match.group(1).strip()
    #                 justification = match.group(2).strip()
    #             else:
    #                 category = category_part
    #                 justification = None
    #             # Clean the category by removing non-alphabetic characters except '!' and convert to uppercase
    #             category = re.sub(r'[^a-zA-Z!]', '', category).upper()
    #             return category, justification
    #     return None, None


