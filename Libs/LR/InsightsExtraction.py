## *****************************************************
import json
import os
import csv
import time
import datetime
from tqdm import tqdm
import pandas as pd
import re
## *****************************************************
from Libs.LR.LiteratureReview import LiteratureReview
## *****************************************************
from Libs.Zotero.AcademicPDFProcessor import AcademicPDFProcessor
## *****************************************************

class InsightsExtraction(LiteratureReview):
    
    def __init__(self, literature_review):
        self.__dict__.update(literature_review.__dict__)  # Copy attributes from LiteratureReview
        
    def run(self, prompt_key, params=None, 
            kwargs_pdf_extract: dict = None, kwargs_pdf_truncate: dict = None, 
            kwargs_pdf_clean: dict = None, kwargs_pdf_save: dict = None, 
            use_pdf_version='cleaned', rewrite=False, resume=False):
        """
        Extract structured insights from full-text papers.
        
        Args:
            prompt_key (str): The key of the prompt to use.
            params (dict): Parameters for the LLM.
            rewrite (bool): Whether to rewrite the CSV file if it exists.
            resume (bool): Whether to resume from existing progress.
        """
        fieldnames = [
            'title', 'authors', 'yop', 'start_timestamp', 'full_response',
            'objective', 'country', 'major_climate', 'major_biome', 'major_urban_trend', 
            'land_use_classes', 'baseline_years', 'target_future_years', 'scenarios', 
            'analysis_methods', 'simulation_method_advantages', 'predictors', 'population_selected', 
            'data_sources', 'tools_software_used', 
            'validation_performed', 'validation_methods','validation_data_sources', 
            'key_findings', 'implications', 'limitations', 'future_directions', 'scalability', 'novelty']
        
        csv_file_path = self.initialize_csv(fieldnames, "InsightsExtraction", rewrite)
    
        # Get the prompt template
        prompt_user_template = self.prompts.get(prompt_key)
        if not prompt_user_template:
            raise ValueError(f"Prompt with key '{prompt_key}' not found.")
    
        # Load processed titles and entries to reprocess
        processed_titles, entries_to_reprocess = self._load_processed_data(csv_file_path, fieldnames, resume)
    
        # Prepare all entries to process
        all_entries = self._prepare_entries(processed_titles, entries_to_reprocess)
    
        # Process all unique entries
        prompts = {'system': "You are an academic researcher extracting structured insights from papers.", 
                   'user': prompt_user_template}
        self._process_entries(all_entries, prompts, fieldnames, csv_file_path, params,
                              kwargs_pdf_extract, kwargs_pdf_truncate, kwargs_pdf_clean, kwargs_pdf_save, use_pdf_version)
    
        # Summarize results
        self._summarize_results(csv_file_path)
        
    def _load_processed_data(self, csv_file_path, fieldnames, resume):
        """Load processed titles and identify entries to reprocess."""
        processed_titles = set()
        entries_to_reprocess = []
        if resume and os.path.exists(csv_file_path):
            with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    title = row['title'].strip()
                    if any(row.get(field) for field in fieldnames[5:]):  # Check if any insight exists
                        processed_titles.add(title)
                    else:
                        entries_to_reprocess.append(row)
        return processed_titles, entries_to_reprocess
        
    def _prepare_entries(self, processed_titles, entries_to_reprocess):
        """Combine entries avoiding duplicates."""
        all_entries = []
        unique_titles = set()
        
        # Add new entries
        for entry in self.entries:
            title = entry.get("title", "").strip()
            if title and title not in processed_titles:
                all_entries.append(entry)
                unique_titles.add(title)
        
        # Add entries to reprocess
        for entry in entries_to_reprocess:
            title = entry.get("title", "").strip()
            if title and title not in unique_titles:
                all_entries.append({
                    "title": entry['title'],
                    "author": entry['authors'],
                    "year": entry['yop'],
                    "abstract": "",
                    "file": entry.get("file", ""),
                    "key": entry.get("key", "")
                })
                unique_titles.add(title)
                
        return all_entries
        
    def _process_entries(self, all_entries, prompts, fieldnames, csv_file_path, params,
                         kwargs_pdf_extract, kwargs_pdf_truncate, kwargs_pdf_clean, 
                         kwargs_pdf_save, use_pdf_version):
        """Process entries and extract insights."""
        file_mode = 'a' if os.path.exists(csv_file_path) else 'w'
        with open(csv_file_path, mode=file_mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if file_mode == 'w':
                writer.writeheader()
                
            for entry in tqdm(all_entries, desc="Extracting insights"):
                start_timestamp = datetime.datetime.now().isoformat()
                title = entry.get("title", "").strip()
                if not title or title in self.processed_titles:
                    continue
                    
                authors = entry.get("author", "")
                year = entry.get("year", "")
                abstract = entry.get("abstract", "")
                zotero_item_key = entry.get("key", "")
                
                # Initialize insights with defaults
                insights = {field: "Not specified" for field in fieldnames[5:]}
                processing_time = 0
                llm_response = None
                
                # Get PDF path
                pdf_file_path = self._get_pdf_path(entry, zotero_item_key)
                
                if pdf_file_path and os.path.isfile(pdf_file_path):
                    try:
                        # Process PDF
                        processor = AcademicPDFProcessor(pdf_file_path)
                        processor.extract_text(**(kwargs_pdf_extract or {}))
                        processor.truncate_text(**(kwargs_pdf_truncate or {}))
                        processor.clean_text(**(kwargs_pdf_clean or {}))
                        processor.save(**(kwargs_pdf_save or {}))
                        
                        full_text = processor.get_processed_text(use_pdf_version)
                        word_count = len(full_text.split())
                        
                        print(word_count)
                        
                        if word_count > 100:  # Minimum text threshold
                            start_time = time.time()
                            
                            # Prepare query with all needed components
                            query_inputs = {
                                'title': title,
                                'abstract': abstract,
                                'full_text': full_text
                            }
                            
                            # Get LLM response
                            llm_response = self.llm.execute_prompt(prompts, query_inputs, params=params)
                            
                            # Parse structured response
                            structured_response = self._parse_llm_response(llm_response)
                            
                            # Update insights with extracted data
                            for key in insights.keys():
                                if key in structured_response: insights[key] = structured_response[key]
                                    
                            processing_time = time.time() - start_time
                            
                    except Exception as e:
                        print(f"Error processing {title}: {str(e)}")
                        
                # Prepare CSV row
                row = {
                    'title': title,
                    'authors': authors,
                    'yop': year,
                    'start_timestamp': start_timestamp,
                    'full_response': llm_response,
                    **insights
                }
                
                writer.writerow(row)
                csvfile.flush()
                self.processed_titles.add(title)
                
                print(f"\nProcessed: {title}")
                print(f"Insights: {json.dumps(insights, indent=2)}")
                print(f"Time: {processing_time:.2f}s")
                print("=" * 80)
                
    def _get_pdf_path(self, entry, zotero_item_key):
        """Get PDF path from either file system or Zotero collection."""
        if self.input_file_path and entry.get("file"):
            parts = entry["file"].split(":")
            if len(parts) > 1:
                return os.path.join(self.input_dir, parts[1].strip())
                
        if self.zotero_collection and zotero_item_key:
            if zotero_item_key in self.zotero_collection.attachments:
                attachment = self.zotero_collection.attachments[zotero_item_key]
                folder_key = list(attachment.keys())[0]
                filename = attachment[folder_key]
                return os.path.join(
                    self.zotero_collection.storage_folder, 
                    folder_key, 
                    filename
                )
        return None

#%%
    
    def _parse_llm_response(self, response):
        """
        Enhanced parser for structured LLM responses with markdown formatting,
        numbered lists, and multi-line values.
        """
        # Handle non-string responses (NaN values)
        if not isinstance(response, str): return {}
    
        # First try JSON parsing
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback to key-value parsing
        parsed = {}
        lines = response.split('\n')
        current_key = None
        current_value = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check for markdown key pattern (e.g., **key:**)
            md_key_match = re.match(r'^\*\*(\s*[\w\s-]+\s*):\*\*', line)
            if md_key_match:
                # Reset current_key if no value accumulated (likely a header)
                if current_key and not current_value:
                    current_key = None
                    current_value = []
                    
                # Save previous key-value pair
                if current_key:
                    parsed[current_key] = '\n'.join(current_value).strip()
                    current_value = []
                    
                # Extract and normalize new key
                raw_key = md_key_match.group(1).strip()
                current_key = raw_key.lower().replace(' ', '_').replace('-', '_')
                
                # Add remaining content if any
                remaining = line[md_key_match.end():].strip()
                if remaining:
                    current_value.append(remaining)
                continue
                
            # Check for numbered list pattern (e.g., 1. key: value)
            numbered_key_match = re.match(
                r'^\d+[\.\)]\s*(?:\*\*)?\s*([\w\s-]+)\s*(?:\*\*)?\s*:\s*(.*)', 
                line
            )
            if numbered_key_match:
                # Reset current_key if no value accumulated (likely a header)
                if current_key and not current_value:
                    current_key = None
                    current_value = []
                    
                # Save previous key-value pair
                if current_key:
                    parsed[current_key] = '\n'.join(current_value).strip()
                    current_value = []
                    
                # Extract and normalize new key
                raw_key = numbered_key_match.group(1).strip()
                current_key = raw_key.lower().replace(' ', '_').replace('-', '_')
                
                # Add value if any
                value_part = numbered_key_match.group(2).strip()
                if value_part:
                    current_value.append(value_part)
                continue
                
            # Check for plain key pattern (key: value)
            plain_key_match = re.match(r'^([\w\s-]+):\s*(.*)', line)
            if plain_key_match:
                # Reset current_key if no value accumulated (likely a header)
                if current_key and not current_value:
                    current_key = None
                    current_value = []
                    
                # Save previous key-value pair
                if current_key:
                    parsed[current_key] = '\n'.join(current_value).strip()
                    current_value = []
                    
                # Extract and normalize new key
                raw_key = plain_key_match.group(1).strip()
                current_key = raw_key.lower().replace(' ', '_').replace('-', '_')
                
                # Add value if any
                value_part = plain_key_match.group(2).strip()
                if value_part:
                    current_value.append(value_part)
                continue
                
            # Accumulate value for current key
            if current_key:
                current_value.append(line)
        
        # Save last key-value pair
        if current_key:
            parsed[current_key] = '\n'.join(current_value).strip()
        
        # Handle special cases for your fields
        return self._post_process_parsed_data(parsed)
    
    def _post_process_parsed_data(self, parsed):
        """Handle special cases and normalize fields"""
        # List-type fields that should be converted to semicolon-separated
        list_fields = [
            'objective', 'country', 'major_climate', 'major_biome', 'major_urban_trend', 
            'land_use_classes', 'baseline_years', 'target_future_years', 'scenarios', 
            'analysis_methods', 'simulation_method_advantages', 'predictors', 'population_selected', 
            'data_sources', 'tools_software_used', 
            'validation_performed', 'validation_methods','validation_data_sources', 
            'key_findings', 'implications', 'limitations', 'future_directions', 'scalability', 'novelty']
        
        for field in list_fields:
            if field in parsed:
                # Convert list items to semicolon-separated format
                if '\n- ' in parsed[field]:
                    items = [item.strip('- ').strip() for item in parsed[field].split('\n- ') if item.strip()]
                    parsed[field] = '; '.join(items)
                elif '\n* ' in parsed[field]:
                    items = [item.strip('* ').strip() for item in parsed[field].split('\n* ') if item.strip()]
                    parsed[field] = '; '.join(items)
        
        # Boolean normalization
        bool_fields = ['validation_performed']
        for field in bool_fields:
            if field in parsed:
                if 'yes' in parsed[field].lower():
                    parsed[field] = 'Yes'
                elif 'no' in parsed[field].lower():
                    parsed[field] = 'No'
        
        return parsed
        
#%%
     
    def _summarize_results(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        
        # Ensure the required column exists
        if 'full_response' not in df.columns:
            print(f"Error: 'full_response' column not found in {csv_file_path}")
            return
    
        total_papers = len(df)
    
        # Identify skipped papers based on 'full_response' values
        skipped_mask = (df['full_response'].isna() | (df['full_response'] == '') | (df['full_response'] == 'Not specified'))
        skipped_count = skipped_mask.sum()
        processed_count = total_papers - skipped_count
    
        # Print detailed summary
        print(f"\nSuccessfully processed {processed_count} papers.")
        print(f"{skipped_count} papers were skipped due to missing or invalid responses.")
        print(f"Total papers analyzed: {total_papers}")
        print("Saved to:", csv_file_path)
        