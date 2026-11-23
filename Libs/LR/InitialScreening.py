## *****************************************************
import os
import pandas as pd
import csv
from tqdm import tqdm
import time
## *****************************************************
from Libs.LR.LiteratureReview import LiteratureReview
## *****************************************************

class InitialScreening(LiteratureReview):

    def __init__(self, literature_review):
        #super().__init__(project_id, llm, input_file_path, output_dir, prompt_file)
        self.__dict__.update(literature_review.__dict__)  # Copy attributes from LiteratureReview


    def run(self, prompt_key, categories, params=None, update_zotero=False, zotero_field="archiveLocation", 
            rewrite=False, resume=False):
        """
        Perform initial screening based on title and abstract.
        
        Args:
            prompt_key (str): The key of the prompt to use (e.g., 'initial_screening_prompt').
            categories (list): List of valid categories for categorization.
            params (dict): Parameters for the LLM (e.g., temperature, max_tokens).
            update_zotero (bool): Whether to update Zotero items with the verdict.
            zotero_field (str): The Zotero field to update with the verdict (e.g., "archiveLocation").
            rewrite (bool): Whether to rewrite the CSV file if it already exists.
            resume (bool): Whether to resume writing to an existing file from where it stopped.
        """
        fieldnames = ['title', 'authors', 'yop', 'verdict', 'processing_time', 'full_response']
        csv_file_path = self.initialize_csv(fieldnames, "InitialScreening", rewrite)

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
        self._process_entries(all_entries, prompts, categories, fieldnames, csv_file_path, params,
                              update_zotero, zotero_field)
    
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
                })
                unique_titles.add(title)
    
        return all_entries
    
    
    def _process_entries(self, all_entries, prompts, categories, fieldnames, csv_file_path, params,
                         update_zotero=False, zotero_field="archiveLocation"):
        """
        Process all unique entries and write results to the CSV file.
        If update_zotero is True, update Zotero items with the verdict during processing.
        
        Args:
            all_entries (list): A list of entries to process.
            prompts (dict): Prompts for the LLM.
            categories (list): List of valid categories for categorization.
            fieldnames (list): Field names for the CSV file.
            csv_file_path (str): Path to the CSV file.
            params (dict): Parameters for the LLM.
            update_zotero (bool): Whether to update Zotero items with the verdict.
            zotero_field (str): The Zotero field to update with the verdict.
        """
        file_mode = 'a' if os.path.exists(csv_file_path) else 'w'
        with open(csv_file_path, mode=file_mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if file_mode == 'w':
                writer.writeheader()
    
            for entry in tqdm(all_entries, desc="Processing papers"):
                title = entry.get("title", "").strip()
                if not title or title in self.processed_titles:
                    continue
    
                abstract = entry.get("abstract", "")
                authors = entry.get("author", "")
                year = entry.get("year", "")
                zotero_key = entry.get("key", "")  # Zotero item key for updating
    
                start_time = time.time()
                verdict = 'UNDECIDED'
                llm_response = None
    
                try:
                    query_inputs = {'title': title, 'abstract': abstract}
                    llm_response = self.llm.execute_prompt(prompts, query_inputs, params=params)
                    final_category = self.extract_final_category(llm_response)
                    if final_category in categories:
                        verdict = final_category
                except Exception as e:
                    print(f"\nError processing {entry.get('ID', 'unknown')}: {e}")
                finally:
                    processing_time = round(time.time() - start_time, 2)
                    row = {
                        'title': title,
                        'authors': authors,
                        'yop': year,
                        'verdict': verdict,
                        'processing_time': processing_time,
                        'full_response': llm_response,
                    }
                    writer.writerow(row)
                    csvfile.flush()
                    self.processed_titles.add(title)
                    
                    # Update Zotero item if requested
                    if update_zotero and zotero_key and self.zotero_collection:
                        try:
                            updates = {zotero_field: verdict}
                            self.zotero_collection.update_item(zotero_key, updates, verbose=False)
                        except Exception as e:
                            print(f"Failed to update Zotero item '{title}': {e}")
    
                    print(f"\nTitle: {title}")
                    print(f"Authors: {authors}")
                    print(f"YoP: {year}")
                    print(f"Final categorization: {verdict}")
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
    
