## *****************************************************
import os
import pandas as pd
import csv
from tqdm import tqdm
import datetime
import time
import re
## *****************************************************
from Libs.LR.LiteratureReview import LiteratureReview
## *****************************************************
from Libs.Zotero.AcademicPDFProcessor import AcademicPDFProcessor

class DeepScreening(LiteratureReview):
    
    def __init__(self, literature_review):
        self.__dict__.update(literature_review.__dict__)  # Copy attributes from LiteratureReview


    def run(self, prompt_key, categories, params=None, update_zotero=False, zotero_field="callNumber", 
            kwargs_pdf_extract: dict = None, kwargs_pdf_truncate: dict = None, 
            kwargs_pdf_clean: dict = None, kwargs_pdf_save: dict = None, use_pdf_version='cleaned',
            rewrite=False, resume=False):
        """
        Perform initial screening based on title and abstract, and deep screening based on full-text.
        
        Args:
            prompt_key (str): The key of the prompt to use (e.g., 'initial_and_deep_screening_prompt').
            categories (list): List of valid categories for categorization.
            params (dict): Parameters for the LLM (e.g., temperature, max_tokens).
            rewrite (bool): Whether to rewrite the CSV file if it already exists.
            resume (bool): Whether to resume writing to an existing file from where it stopped.
        """
        fieldnames = ['title', 'authors', 'yop', 'verdict', 'justification', 'start_timestamp', 'processing_time', 'full_response']
        csv_file_path = self.initialize_csv(fieldnames, "DeepScreening", rewrite)
    
        # Get the prompt template
        prompt_user_template = self.prompts.get(prompt_key)
        if not prompt_user_template:
            raise ValueError(f"Prompt with key '{prompt_key}' not found in the prompt file.")
    
        # Load processed titles and entries to reprocess
        processed_titles, entries_to_reprocess = self._load_processed_data(csv_file_path, fieldnames, resume)
    
        # Prepare all entries to process
        all_entries = self._prepare_entries(processed_titles, entries_to_reprocess)
    
        # Process all unique entries
        prompts = {'system': "You are an academic researcher filtering papers.", 'user': prompt_user_template}
        self._process_entries(all_entries, prompts, categories, fieldnames, csv_file_path, params, update_zotero, zotero_field, 
                              kwargs_pdf_extract, kwargs_pdf_truncate, kwargs_pdf_clean, kwargs_pdf_save, use_pdf_version)
    
        # Clean and summarize results
        self._summarize_results(csv_file_path, categories)
    
    
    def _load_processed_data(self, csv_file_path, fieldnames, resume):
        """
        Load processed titles and identify entries to reprocess.
        """
        processed_titles = set()
        entries_to_reprocess = []
        if resume and os.path.exists(csv_file_path):
            with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    title = row['title'].strip()
                    verdict = row['verdict'].strip()
                    full_response = row['full_response'].strip()
    
                    # Add fully processed titles to the set
                    if verdict != 'UNDECIDED' or full_response:
                        processed_titles.add(title)
                    else:
                        # Identify entries with UNDECIDED verdict and empty full_response for reprocessing
                        entries_to_reprocess.append(row)
        return processed_titles, entries_to_reprocess
    
    
    def _prepare_entries(self, processed_titles, entries_to_reprocess):
        """
        Combine regular entries with entries to reprocess, avoiding duplicates.
        """
        all_entries = []
        unique_titles = set()
    
        # Add entries from self.entries first
        for entry in self.entries:
            title = entry.get("title", "").strip()
            if title and title not in processed_titles:
                all_entries.append(entry)
                unique_titles.add(title)
    
        # Add entries from entries_to_reprocess only if their titles are not already in all_entries
        for entry in entries_to_reprocess:
            title = entry.get("title", "").strip()
            if title and title not in unique_titles:
                all_entries.append({
                    "title": entry['title'],
                    "author": entry['authors'],
                    "year": entry['yop'],
                    "abstract": "",  # Abstract may not be available in the CSV; leave blank
                    "file": entry.get("file", ""),  # Include file path for full-text extraction
                })
                unique_titles.add(title)
    
        return all_entries
    
    
    def _process_entries(self, all_entries, prompts, categories, fieldnames, csv_file_path, params,
                         update_zotero=False, zotero_field="callNumber", 
                         kwargs_pdf_extract: dict = None, kwargs_pdf_truncate: dict = None, kwargs_pdf_clean: dict = None, 
                         kwargs_pdf_save: dict = None, use_pdf_version='cleaned'):
        """
        Process all unique entries and write results to the CSV file.
        """
        file_mode = 'a' if os.path.exists(csv_file_path) else 'w'
        with open(csv_file_path, mode=file_mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if file_mode == 'w':
                writer.writeheader()
    
            for entry in tqdm(all_entries, desc="Processing papers"):
                start_timestamp = datetime.datetime.now().isoformat()  # Get current ISO timestamp
                title = entry.get("title", "").strip()
                if not title or title in self.processed_titles:
                    continue
    
                abstract = entry.get("abstract", "")
                authors = entry.get("author", "")
                year = entry.get("year", "")
                zotero_item_key = entry.get("key", "")  # Zotero item key for updating
                
                # Extract full text from PDF if available
                pdf_file_path, full_text = '', ''
                if self.input_file_path:
                    pdf_file_info = entry.get("file", "")
                    if pdf_file_info:
                        pdf_file_parts = pdf_file_info.split(":")
                        if len(pdf_file_parts) > 1:
                            pdf_file_relative_path = pdf_file_parts[1].strip()
                            # pdf_file_path = os.path.normpath(os.path.join(self.input_dir, pdf_file_relative_path))
                            pdf_file_path = os.path.join(self.input_dir, pdf_file_relative_path)
                if self.zotero_collection:
                    # Access the attachments dictionary using the zotero_item_key
                    if zotero_item_key in self.zotero_collection.attachments:
                        attachment_details = self.zotero_collection.attachments[zotero_item_key]
                        attchment_folder_key = list(attachment_details.keys())[0]
                        pdf_filename = attachment_details[attchment_folder_key]            
                        pdf_file_path = os.path.join(
                            self.zotero_collection.storage_folder, attchment_folder_key, pdf_filename)
                
                # Initialize default values
                full_text = ''
                verdict = 'NO PDF'  # Default verdict if PDF check fails
                justification = '' 
                processing_time = 0
                llm_response = None
                
                # Check if PDF file is accessible
                if pdf_file_path and os.path.isfile(pdf_file_path):
                    try:
                        # Process PDF
                        kwargs_pdf_extract = kwargs_pdf_extract or {}
                        kwargs_pdf_truncate = kwargs_pdf_truncate  or {}
                        kwargs_pdf_clean = kwargs_pdf_clean or {}
                        kwargs_pdf_save = kwargs_pdf_save or {}
                        
                        processor = AcademicPDFProcessor(pdf_file_path)
                        processor.extract_text(**kwargs_pdf_extract)
                        processor.truncate_text(**kwargs_pdf_truncate)
                        processor.clean_text(**kwargs_pdf_clean)
                        processor.save(**kwargs_pdf_save)
                        
                        full_text = processor.get_processed_text(use_pdf_version)
                        #print(len(full_text))
                        
                        # Check text validity
                        word_count = len(full_text.strip().split()) if full_text.strip() else 0
                        min_word_threshold = 50
    
                        if word_count < min_word_threshold: verdict = 'PDF ISSUE'
                        else:
                            # Process LLM verdict
                            start_time = time.time()
                            verdict = 'UNDECIDED'
                            try:
                                query_inputs = {'title': title, 'abstract': abstract, 'full_text': full_text}
                                llm_response = self.llm.execute_prompt(prompts, query_inputs, params=params)
                                
                                final_category, justification = self.extract_final_category_and_justification(llm_response)
                                if final_category in categories:
                                    verdict = final_category
                            except Exception as e:
                                print(f"\nError processing {entry.get('ID', 'unknown')}: {e}")
                            finally:
                                processing_time = round(time.time() - start_time, 2)
                    except Exception as e:
                        print(f"\nError processing PDF for {title}: {e}")
                        verdict = 'NO PDF'
    
                # Write results to CSV
                row = {
                    'title': title,
                    'authors': authors,
                    'yop': year,
                    'verdict': verdict,
                    'justification': justification,
                    'start_timestamp': start_timestamp,
                    'processing_time': processing_time,
                    'full_response': llm_response,
                }
                writer.writerow(row)
                csvfile.flush()
                self.processed_titles.add(title)
                
                # Update Zotero item if requested
                if update_zotero and zotero_item_key and self.zotero_collection:
                    try:
                        updates = {zotero_field: verdict}
                        self.zotero_collection.update_item(zotero_item_key, updates, verbose=False)
                    except Exception as e:
                        print(f"Failed to update Zotero item '{title}': {e}")
    
                print(f"\nTitle: {title}")
                print(f"Authors: {authors}")
                print(f"YoP: {year}")
                print(f"Final categorization: {verdict}")
                print(f"Justification: {justification}")
                print(f"Processing time: {round(processing_time / 60, 1)} min")
                print("********************************************************\n")
    
    
    def _summarize_results(self, csv_file_path, categories):
        """
        Clean the CSV file and print final counts.
        """
        self.clean_csv_file(csv_file_path)
        df = pd.read_csv(csv_file_path)
        print("\nFinal counts:")
        for cat in categories:
            count = df[df['verdict'] == cat].shape[0]
            print(f"{cat}: {count}")
            

