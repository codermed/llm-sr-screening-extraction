## *****************************************************
import os
import pandas as pd
import re
from datetime import datetime
from tqdm import tqdm
import random
from collections import defaultdict
## *****************************************************

class ZoteroCollection:
    def __init__(self, zotero, collection_key, item_types=None, attachments=False):
        """
        Initialize a ZoteroCollection instance.
        
        Args:
            zotero (Zotero): An instance of the Zotero class to interact with the Zotero API.
            collection_key (str): The key of the target collection in Zotero.
        """
        self.zotero = zotero
        self.collection_key = collection_key
        self.items, self.attachments = self._load_items_and_attachments(item_types=item_types, attachments=attachments)
        # Local Zotero storage folder holding the PDF attachments.
        # Set the ZOTERO_STORAGE_DIR environment variable, e.g. ~/Zotero/storage
        self.storage_folder = os.environ.get(
            'ZOTERO_STORAGE_DIR', os.path.join(os.path.expanduser('~'), 'Zotero', 'storage'))
        #self.items = self._load_items(item_types=item_types)
        #self.attachments = self._load_attachments() if attachments else None
        self.df = self.get_dataframe()
        
        
    def _load_items_and_attachments(self, item_types=None, attachments=False, verbose=True):
        """
        Load items and optionally their PDF attachments from the specified Zotero collection.
        
        Args:
            item_types (list, optional): A list of item types to include (e.g., ["journalArticle", "conferencePaper"]).
                                         If None, all items are loaded.
            attachments (bool, optional): Whether to load PDF attachments for the items.
        
        Returns:
            tuple: A tuple containing:
                - A list of filtered items (articles).
                - A dictionary of PDF attachments (if attachments=True), where keys are parent item keys and values are paths.
        """
        print(f"Loading items from Zotero collection with key: {self.collection_key}...")
        all_items = self.zotero.get_items_from_collection(self.collection_key)
        
        # Separate items into parent items (articles) and attachments
        parent_items = []
        attachment_items = []
        for item in all_items:
            item_data = item.get('data', {})
            if item_data.get('itemType') in (item_types or []):
                parent_items.append(item)
            elif item_data.get('itemType') == 'attachment':
                attachment_items.append(item)
    
        # Filter parent items if item_types is provided
        if verbose:
            if item_types:
                print(f"\nFound {len(all_items)} total items in the collection.")
                print(f"Filtered {len(parent_items)} items of types: {item_types} / total {len(all_items)} items")
            else:
                print(f"Successfully loaded {len(parent_items)} items from the collection.")
    
        # Load PDF attachments if requested
        attachments_dict = {}
        if attachments:
            print("Loading PDF attachments for items in the collection...")
            for attachment in tqdm(attachment_items, desc="Processing Attachments", unit="attachment"):
                attachment_data = attachment.get('data', {})
                if attachment_data.get('contentType') == 'application/pdf':
                    # Get the attachment key and parent item key
                    attachment_key = attachment_data.get('key')
                    parent_item_key = attachment_data.get('parentItem')
                    if attachment_key and parent_item_key:
                        # Extract the filename of the PDF
                        pdf_filename = attachment_data.get('filename')
                        if pdf_filename:
                            # Add the attachment to the dictionary
                            attachments_dict[parent_item_key] = {attachment_key: pdf_filename}
        
            if verbose: print(f"\nSuccessfully loaded PDF attachments for {len(attachments_dict)} attachments.")
    
        return parent_items, attachments_dict if attachments else None

    def _load_items(self, item_types=None):
        """
        Load all items from the specified Zotero collection, optionally filtering by item type.
        
        Args:
            item_types (list, optional): A list of item types to include (e.g., ["journalArticle", "conferencePaper"]).
                                         If None, all items are loaded.
        
        Returns:
            list: A list of filtered items in the collection.
        """
        print(f"Loading items from Zotero collection with key: {self.collection_key}...")
        items = self.zotero.get_items_from_collection(self.collection_key)
        
        # Filter items by itemType if item_types is provided
        if item_types:
            filtered_items = [item for item in items if item.get('data', {}).get('itemType') in item_types]
            print(f"Found {len(items)} total items in the collection.")
            print(f"Filtered {len(filtered_items)} items of types: {item_types} / total {len(items)} items")
            return filtered_items
        
        print(f"Successfully loaded {len(items)} items from the collection.")
        return items
    
    def get_dataframe(self, verbose=True):
        """
        Retrieve items from the collection as a pandas DataFrame, including all fields and optionally PDF attachments.
        
        Args:
            verbose (bool): Whether to print progress messages.        
        Returns:
            pandas.DataFrame: A DataFrame where each row represents an item with all fields and attachment paths.
        """
        
        items_data = []
        
        for item in tqdm(self.items, desc="Building DataFrame", unit="item"):
            try:
                # Merge top-level item fields with data fields
                item_data = item.copy()
                data_fields = item_data.pop('data', {})
                item_data.update(data_fields)
                
                # Extract publication year from date field
                date_str = item_data.get('date', '')
                item_data['publicationYear'] = self._extract_year(date_str)
                
                # Add PDF paths if attachments are enabled
                if self.attachments is not None:
                    parent_key = item_data.get('key')
                    pdf_paths = []
                    if parent_key in self.attachments:
                        for attach_key, filename in self.attachments[parent_key].items():
                            # Construct full path to PDF
                            pdf_path = os.path.join(self.storage_folder, attach_key, filename)
                            pdf_paths.append(pdf_path)
                    item_data['pdf_paths'] = pdf_paths
                else:
                    item_data['pdf_paths'] = []
                    
                items_data.append(item_data)
                
            except Exception as e:
                print(f"Failed to process item {item.get('key', 'unknown')}: {e}")
        
        # Create DataFrame and expand nested columns
        df = pd.json_normalize(items_data)
        
        return df

    def get_items_as_bibtex(self):
        """
        Retrieve all items from the collection in BibTeX format.
        
        Returns:
            str: A string containing the BibTeX representation of all items in the collection.
        """
        print("Converting Zotero items to BibTeX format...")
        bibtex_entries = []
        for item in tqdm(self.items, desc="Converting Items to BibTeX", unit="item"):
            try:
                bibtex = self.zotero.zot.item(item['key'], content="bibtex")
                bibtex_entries.append(bibtex)
            except Exception as e:
                print(f"Failed to convert item '{item['data']['title']}' to BibTeX: {e}")
        return "\n".join(bibtex_entries)

    def get_items_as_dict(self):
        """
        Retrieve all items from the collection as a list of dictionaries.
        
        Returns:
            list: A list of dictionaries, where each dictionary represents an item's metadata.
        """
        print("Extracting Zotero items as dictionaries...")
        items_as_dict = []
        for item in tqdm(self.items, desc="Extracting Item Metadata", unit="item"):
            try:                
                # Extract creators (authors) and format them as "FirstName LastName"
                authors = [
                    f"{creator.get('firstName', '')} {creator.get('lastName', '')}".strip()
                    for creator in item['data'].get('creators', [])]
                authors_str = ", ".join(authors)
                
                # Extract tags
                tags = [tag['tag'] for tag in item['data'].get('tags', [])]
    
                # Extract collections (if any)
                collections = item['data'].get('collections', [])
    
                # Extract publication date and split into year/month/day if available
                date_parts = item['data'].get('date', '').split('-')
                year = date_parts[0] if len(date_parts) > 0 else ''
                month = date_parts[1] if len(date_parts) > 1 else ''
                day = date_parts[2] if len(date_parts) > 2 else ''
    
                # Construct the metadata dictionary for the item
                item_data = {
                    "key": item['key'],  # Unique key for the item
                    "title": item['data'].get('title', ''),  # Title of the item
                    "author": authors_str,  # List of authors
                    "year": year,  # Year extracted from the date field
                    "month": month,  # Month extracted from the date field
                    "day": day,  # Day extracted from the date field
                    "abstract": item['data'].get('abstractNote', ''),  # Abstract or notes
                    "publicationTitle": item['data'].get('publicationTitle', ''),  # Journal or publication title
                    "volume": item['data'].get('volume', ''),  # Volume number
                    "issue": item['data'].get('issue', ''),  # Issue number
                    "pages": item['data'].get('pages', ''),  # Page range
                    "doi": item['data'].get('DOI', ''),  # DOI
                    "issn": item['data'].get('ISSN', ''),  # ISSN
                    "url": item['data'].get('url', ''),  # URL
                    "language": item['data'].get('language', ''),  # Language of the publication
                    "tags": tags,  # List of tags
                    "collections": collections,  # List of collection keys
                    "extra": item['data'].get('extra', ''),  # Extra information
                    "dateAdded": item['data'].get('dateAdded', ''),  # Date the item was added to Zotero
                    "dateModified": item['data'].get('dateModified', ''),  # Date the item was last modified
                    "pdf_paths": [],  # Placeholder for PDF paths
                }
                
                # Check for attachments
                attachments = item['data'].get('attachments', [])
                for attachment in attachments:
                    if attachment.get('contentType') == 'application/pdf':
                        print('OK')
                        
                        # Extract the path or filename of the PDF
                        pdf_path = attachment.get('path') or attachment.get('filename')
                        if pdf_path:
                            item_data["pdf_path"] = pdf_path
                            break  # Stop after finding the first PDF (since there's only one)
    
                # Append the metadata dictionary to the list
                items_as_dict.append(item_data)

            except Exception as e:
                print(f"Failed to extract metadata for item '{item['data'].get('title', '')}': {e}")
    
        return items_as_dict
    
    def _load_attachments(self):
        """
        Load PDF attachments for all items in the collection.
        
        Returns:
            dict: A dictionary where keys are item keys and values are lists of PDF attachment paths.
        """
        print("Loading PDF attachments for items in the collection...")
        attachments_dict = {}
    
        for item in tqdm(self.items, desc="Loading Attachments", unit="item"):
            item_key = item['key']
            attachments_dict[item_key] = []
    
            # Fetch attachments for the current item
            item_attachments = item['data'].get('attachments', [])
            for attachment in item_attachments:
                if attachment.get('contentType') == 'application/pdf':
                    # Extract the path or filename of the PDF
                    pdf_path = attachment.get('path') or attachment.get('filename')
                    if pdf_path:
                        attachments_dict[item_key].append(pdf_path)
    
        print(f"Successfully loaded PDF attachments for {len(attachments_dict)} items.")
        return attachments_dict
    
    
    def _extract_year(self, date_str):
        """Robust year extraction from various date formats"""
        if not date_str:
            return None
        
        # Try different parsing strategies
        try:
            # Strategy 1: Direct 4-digit match
            if re.match(r'^\d{4}$', date_str):
                return int(date_str)
            
            # Strategy 2: ISO-like formats (YYYY-MM-DD, YYYY-MM, etc.)
            if '-' in date_str:
                parts = date_str.split('-')
                if len(parts[0]) == 4:
                    return int(parts[0])
                
            # Strategy 3: Date parsing with multiple formats
            formats = [
                '%Y-%m-%d', '%Y-%m', '%Y',
                '%b %Y', '%B %Y',  # "Apr 2020", "April 2020"
                '%d %b %Y', '%d %B %Y',  # "15 Apr 2020"
                '%Y/%m/%d', '%Y/%m',
                '%m/%d/%Y',  # US format
                '%d.%m.%Y',  # European format
                '%Y年'  # Japanese/Chinese year format
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.year
                except ValueError:
                    continue
                    
            # Strategy 4: Fallback to regex search
            year_match = re.search(r'\b\d{4}\b', date_str)
            if year_match and 1000 <= int(year_match.group()) <= datetime.now().year + 5:
                return int(year_match.group())
                
        except Exception as e: pass
            
        return None  # Couldn't parse date

    
    def update_item(self, item_key, updates, overwrite=False, verbose=False):
        """
        Update a single Zotero item with new data.
        
        Args:
            item_key (str): The key of the Zotero item to update.
            updates (dict): A dictionary mapping field names to their new values.
            overwrite (bool): Whether to overwrite existing non-empty values (default: False).
                              If False, only empty fields will be updated.
            verbose (bool): Whether to display detailed log messages for the updated item.
        Returns:
            bool: True if the item was successfully updated, False otherwise.
        """
        # Retrieve the item by its key
        item = self.zotero.zot.item(item_key)
        if not item:
            print(f"No item found with key '{item_key}'.")
            return False
        
        # Helper function to check if a value is empty
        def is_empty(value):
            return value is None or str(value).strip() == ''
    
        # Apply updates to the item's data
        updated = False
        for field, new_value in updates.items():
            if field not in item['data']:
                if verbose:
                    print(f"Field '{field}' does not exist in the item's data.")
                continue
                
            current_value = item['data'][field]
            
            # Skip update if overwrite is disabled and field has existing value
            if not overwrite and not is_empty(current_value):
                if verbose:
                    title = item['data'].get('title', 'Untitled')
                    print(f"Skipping '{field}' for '{title}' - already has value: '{current_value}'")
                continue
                
            # Skip update if value wouldn't change
            if str(current_value) == str(new_value):
                if verbose:
                    title = item['data'].get('title', 'Untitled')
                    print(f"Skipping '{field}' for '{title}' - same value: '{current_value}'")
                continue
                
            # Apply the update
            item['data'][field] = new_value
            updated = True
            if verbose:
                title = item['data'].get('title', 'Untitled')
                action = "Updated" if overwrite else "Filled"
                print(f"{action} '{field}' for '{title}': '{current_value}' → '{new_value}'")
    
        # Update the item in Zotero if changes were made
        if updated:
            try:
                self.zotero.zot.update_item(item)
                return True
            except Exception as e:
                print(f"Failed to update item '{item['data'].get('title', '')}': {e}")
                return False
        else:
            if verbose:
                print(f"No updates applied to item '{item['data'].get('title', '')}'.")
            return False
        
    # def update_item(self, item_key, updates, verbose=False):
    #     """
    #     Update a single Zotero item with new data.
        
    #     Args:
    #         item_key (str): The key of the Zotero item to update.
    #         updates (dict): A dictionary mapping field names to their new values.
    #         verbose (bool): Whether to display detailed log messages for the updated item.
        
    #     Returns:
    #         bool: True if the item was successfully updated, False otherwise.
    #     """
    #     # Retrieve the item by its key
    #     item = self.zotero.zot.item(item_key)
    #     if not item:
    #         print(f"No item found with key '{item_key}'.")
    #         return False
    
    #     # Apply updates to the item's data
    #     updated = False
    #     for field, value in updates.items():
    #         if field in item['data']:
    #             item['data'][field] = value
    #             updated = True
    #         else:
    #             print(f"Field '{field}' does not exist in the item's data.")
    
    #     # Update the item in Zotero if changes were made
    #     if updated:
    #         try:
    #             self.zotero.zot.update_item(item)
    #             if verbose:
    #                 tqdm.write(f"Updated item '{item['data'].get('title', '')}' with fields: {updates}")
    #             return True
    #         except Exception as e:
    #             print(f"Failed to update item '{item['data'].get('title', '')}': {e}")
    #             return False
    #     else:
    #         print(f"No updates applied to item '{item['data'].get('title', '')}'.")
    #         return False

    def update_items(self, updates_map, fields=None, verbose=False):
        """
        Update multiple Zotero items based on a mapping of titles to updates.
        
        Args:
            updates_map (dict): A dictionary mapping normalized titles to dictionaries of updates.
            fields (list, optional): A list of fields to update. If None, all fields in the updates are applied.
            verbose (bool): Whether to display detailed log messages for each updated item.
        
        Returns:
            int: The number of updated items.
        """
        updated_count = 0
    
        # Wrap the items list with tqdm to show a progress bar
        for item in tqdm(self.items, desc="Updating Zotero Items", unit="item"):
            # Extract and normalize the title of the Zotero item
            zotero_title = item.get('data', {}).get('title', '')
            if not zotero_title:
                continue  # Skip items without a title
    
            normalized_zotero_title = self.zotero.normalize_title(zotero_title)
    
            # Check if the normalized title exists in the updates map
            if normalized_zotero_title in updates_map:
                updates = updates_map[normalized_zotero_title]
    
                # Filter updates to only include specified fields, if provided
                if fields:
                    updates = {field: updates[field] for field in fields if field in updates}
    
                # Apply updates to the item's data
                updated = False
                for field, value in updates.items():
                    if field in item['data']:
                        item['data'][field] = value
                        updated = True
                    else:
                        print(f"Field '{field}' does not exist in the item's data.")
    
                # Update the item in Zotero if changes were made
                if updated:
                    try:
                        self.zotero.zot.update_item(item)
                        updated_count += 1
                        if verbose:
                            tqdm.write(f"Updated item '{zotero_title}' with fields: {updates}")
                    except Exception as e:
                        print(f"Failed to update item '{zotero_title}': {e}")
    
        return updated_count
    
#%%

    def categorize_priority(self, hum_field, llm_field, priority_field, overwrite=False, verbose=True):
        """
        Categorizes items into HIGH/MEDIUM/LOW priority based on standardized values from three fields.
        Updates the target_field in Zotero and local data.
        
        Args:
            hum_field (str): Zotero human assessment field (e.g., "archive" for HUM)
            llm_field (str): Zotero llm assessment field (e.g., "archiveLocation" or "callNumber" for LLM_IS and LLM_DS respectively)
            priority_field (str): Field to store priority result based on hum_field and llm_field (will be overwritten)
            verbose (bool): Show update logs
        """
        
        # Display critical field mapping with warnings
        print("\n" + "="*80)
        print("⚠️ CRITICAL FIELD MAPPING VERIFICATION ⚠️")
        print("="*80)
        print(f"HUM FIELD (human assessment):\t'{hum_field}'")
        print(f"LLM FIELD (LLM assessment):\t'{llm_field}'")
        print(f"PRIORITY FIELD (will be overwritten!):\t'{priority_field}'")
        print("="*80)
        print("WARNING: This operation will PERMANENTLY modify your Zotero collection")
        print("         by overwriting values in the PRIORITY FIELD shown above.")
        print("="*80 + "\n")
        
        # Get user confirmation
        confirmation = input("Do you want to proceed? (y/n): ").strip().lower()
        if confirmation != 'y':
            print("\nOperation cancelled by user. No changes were made.")
            return

        # Standardization functions (nested for encapsulation)
        def standardize_human(value):
            value = str(value).upper().strip()
            if (value.startswith('EXCLUDED') or value.startswith('GLOBAL') or 
                'REFERENCE (PAPER)' in value or 'REFERENCE (REVIEW)' in value or 
                'DATA (GLOBAL)' in value or 'PLANTS/SPECIES' in value or 'TWN' in value): return 'EXCLUDED'
            elif value.startswith('OK'): return 'OK'
            elif value.startswith('POSSIBLE'): return 'POSSIBLE'
            elif value.startswith('REVIEW'): return 'REVIEW'
            else: return 'OTHER'

        def standardize_llm(value):
            value = str(value).upper()
            return value if value in ['EXCLUDED', 'POSSIBLE', 'OK', 'REVIEW'] else 'OTHER'

        print(hum_field, llm_field, priority_field)
        
        # Initialize counters
        priority_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        updated_count = 0

        # Process each item
        for item in tqdm(self.items, desc="Categorizing Priority", unit="item"):
            data = item.get('data', {})
            
            # Extract and standardize values
            hum_val_raw = data.get(hum_field, '')
            llm_val_raw = data.get(llm_field, '')
            hum_val = standardize_human(hum_val_raw)
            llm_val = standardize_llm(llm_val_raw)
            
            # Determine priority
            if hum_val == 'OK': priority = 'HIGH'
            elif llm_val == 'OK' or llm_val == 'POSSIBLE': priority = 'MEDIUM'
            else: priority = 'LOW'
            
            # Count priority
            priority_counts[priority] += 1
            
            # Debug print per item (optional)
            if verbose:
                print(f"\nItem: {data.get('title', 'Untitled')[:50]}...")
                print(f"  HUM: {hum_val_raw} → Standard: {hum_val}")
                print(f"  LLM: {llm_val_raw} → Standard: {llm_val}")
                print(f"  [HUM: {hum_val} | LLM: {llm_val}] → Priority: {priority}")
                print("="*80)
                
            # Update item if priority_field value differs
            current_priority = str(data.get(priority_field, '')).strip()
            if current_priority != priority:
                # Update Zotero via API
                update_success = self.update_item(
                    item_key=item['key'],
                    updates={priority_field: priority},
                    overwrite=overwrite,
                    verbose=False
                )
                # Update local data if API update succeeded
                if update_success:
                     item['data'][priority_field] = priority
                     updated_count += 1
        
        # Calculate distribution percentages
        total = len(self.items)
        high_pct = (priority_counts['HIGH'] / total) * 100
        medium_pct = (priority_counts['MEDIUM'] / total) * 100
        low_pct = (priority_counts['LOW'] / total) * 100
        
        # Final summary
        print(f"\n{'='*50}")
        print(f"OPERATION COMPLETE: Updated {updated_count}/{len(self.items)} items")
        print(f"Priority field '{priority_field}' now contains new categorizations")
        
        print(f"\n{'='*50}")
        print("PRIORITY DISTRIBUTION REPORT")
        print('-'*50)
        print(f"HIGH Priority:\t{priority_counts['HIGH']} items\t({high_pct:.1f}%)")
        print(f"MEDIUM Priority:\t{priority_counts['MEDIUM']} items\t({medium_pct:.1f}%)")
        print(f"LOW Priority:\t{priority_counts['LOW']} items\t({low_pct:.1f}%)")
        print('-'*50)
        print("="*50)
        
#%%

    def update_field_in_collection(self, source_field, target_field, overwrite=True, verbose=True):
        """
        Copy data from one field to another within the same Zotero collection.
        
        Args:
            source_field (str): Zotero field to copy data from
            target_field (str): Zotero field to copy data to
            overwrite (bool): Whether to overwrite existing target field values
            verbose (bool): Whether to print detailed progress
            
        Returns:
            dict: Summary of update results
        """
        results = {
            'total_items': len(self.items),
            'updated': 0,
            'skipped_empty_source': 0,
            'skipped_same_value': 0,
            'update_failed': 0
        }
    
        for item in tqdm(self.items, desc=f"Copying {source_field} → {target_field}"):
            item_key = item.get('key')
            item_data = item.get('data', {})
            item_title = item_data.get('title', 'Unknown Title')
            
            source_value = item_data.get(source_field, '')
            target_value = item_data.get(target_field, '')
            
            # Skip if source is empty
            if not source_value or str(source_value).strip() == '':
                results['skipped_empty_source'] += 1
                if verbose:
                    print(f"⚠️ Empty {source_field}: {item_title[:50]}...")
                continue
            
            # Skip if values are the same and we're not overwriting
            if not overwrite and target_value and str(target_value).strip() != '':
                results['skipped_same_value'] += 1
                if verbose:
                    print(f"⚠️ Target exists: {item_title[:50]}...")
                continue
            
            # Skip if values are identical
            if str(source_value) == str(target_value):
                results['skipped_same_value'] += 1
                if verbose:
                    print(f"⚠️ Same value: {item_title[:50]}...")
                continue
            
            # Update the item
            success = self.update_item(
                item_key=item_key,
                updates={target_field: str(source_value)},
                overwrite=overwrite,
                verbose=False
            )
            
            if success:
                results['updated'] += 1
                if verbose:
                    print(f"✅ {item_title[:50]}... -> {target_field}: '{source_value}'")
            else:
                results['update_failed'] += 1
                if verbose:
                    print(f"❌ Update failed: {item_title[:50]}...")
    
        # Print summary
        if verbose:
            print(f"\n📊 WITHIN-COLLECTION UPDATE SUMMARY:")
            print(f"   Source field: {source_field}")
            print(f"   Target field: {target_field}")
            print(f"   Total items: {results['total_items']}")
            print(f"   Successfully updated: {results['updated']}")
            print(f"   Empty source values: {results['skipped_empty_source']}")
            print(f"   Skipped (same/existing): {results['skipped_same_value']}")
            print(f"   Update failures: {results['update_failed']}")
    
        return results
    
    
    def update_field_from_csv(self, csv_file_path, 
                             source_column, 
                             target_field,
                             key_column='key',
                             title_column='title', 
                             doi_column='DOI',
                             overwrite=True,
                             overwrite_only_if_different=True,  # NEW: Only overwrite if value changes
                             verbose=True,
                             use_title_fallback=True,
                             use_doi_fallback=True):
        """
        Update any field in Zotero items from any column in a CSV file.
        
        Args:
            csv_file_path (str): Path to the CSV file
            source_column (str): CSV column name to read values from
            target_field (str): Zotero field name to update
            key_column (str): CSV column name for Zotero keys
            title_column (str): CSV column name for titles
            doi_column (str): CSV column name for DOIs
            overwrite (bool): Whether to overwrite existing values
            overwrite_only_if_different (bool): NEW - Only overwrite if new value is different
            verbose (bool): Whether to print detailed progress
            use_title_fallback (bool): Use title matching as fallback
            use_doi_fallback (bool): Use DOI matching as fallback
            
        Returns:
            dict: Summary of update results
        """
        import pandas as pd
        
        # ADJUSTMENT 1: Only load necessary columns to save memory
        columns_to_load = [key_column, source_column]
        if use_title_fallback:
            columns_to_load.append(title_column)
        if use_doi_fallback:
            columns_to_load.append(doi_column)
        
        # Remove duplicates while preserving order
        columns_to_load = list(dict.fromkeys(columns_to_load))
        
        # Load CSV file with only necessary columns
        try:
            df = pd.read_csv(csv_file_path, usecols=columns_to_load)
            if verbose:
                print(f"✅ Loaded CSV with {len(df)} rows")
                print(f"📊 Loaded columns: {columns_to_load}")
                print(f"📊 Available CSV columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"❌ Failed to load CSV: {e}")
            # Fallback: Try loading all columns and then filter
            try:
                if verbose:
                    print("🔄 Trying fallback: loading all columns...")
                df_full = pd.read_csv(csv_file_path)
                df = df_full[columns_to_load].copy()
                if verbose:
                    print(f"✅ Fallback successful - filtered to {len(columns_to_load)} columns")
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                return None
    
        # Validate columns exist
        if source_column not in df.columns:
            print(f"❌ Source column '{source_column}' not found in CSV")
            return None
        if key_column not in df.columns:
            print(f"❌ Key column '{key_column}' not found in CSV")
            return None
    
        # Create mappings
        key_to_value = dict(zip(df[key_column], df[source_column]))
        
        # Create title mapping if requested
        title_to_value = {}
        if use_title_fallback and title_column in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row.get(title_column)) and pd.notna(row.get(source_column)):
                    norm_title = self.zotero.normalize_title(str(row[title_column]))
                    title_to_value[norm_title] = row[source_column]
            if verbose:
                print(f"📝 Created {len(title_to_value)} title mappings")
    
        # Create DOI mapping if requested
        doi_to_value = {}
        if use_doi_fallback and doi_column in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row.get(doi_column)) and pd.notna(row.get(source_column)) and row[doi_column] != '':
                    norm_doi = str(row[doi_column]).lower().strip()
                    doi_to_value[norm_doi] = row[source_column]
            if verbose:
                print(f"🔗 Created {len(doi_to_value)} DOI mappings")
    
        # Initialize results
        results = {
            'total_items': len(self.items),
            'updated': 0,
            'matched_by_key': 0,
            'matched_by_title': 0,
            'matched_by_doi': 0,
            'no_match': 0,
            'update_failed': 0,
            'empty_value': 0,
            'skipped_same_value': 0  # NEW: Track items skipped due to same value
        }
    
        # Update items
        for item in tqdm(self.items, desc=f"Updating {target_field} from {source_column}"):
            item_key = item.get('key')
            item_data = item.get('data', {})
            item_title = item_data.get('title', 'Unknown Title')
            item_doi = item_data.get('DOI', '')
            
            source_value = None
            match_method = "none"
            
            # Method 1: Key matching
            if item_key in key_to_value:
                source_value = key_to_value[item_key]
                match_method = "key"
                results['matched_by_key'] += 1
            
            # Method 2: Title matching
            elif use_title_fallback and item_title and item_title != 'Unknown Title':
                norm_title = self.zotero.normalize_title(item_title)
                if norm_title in title_to_value:
                    source_value = title_to_value[norm_title]
                    match_method = "title"
                    results['matched_by_title'] += 1
            
            # Method 3: DOI matching
            elif use_doi_fallback and item_doi:
                norm_doi = str(item_doi).lower().strip()
                if norm_doi in doi_to_value:
                    source_value = doi_to_value[norm_doi]
                    match_method = "doi"
                    results['matched_by_doi'] += 1
            
            # ADJUSTMENT 2: Check if we should update based on overwrite settings
            should_update = False
            skip_reason = ""
            
            if source_value is not None and pd.notna(source_value) and source_value != '':
                current_value = item_data.get(target_field, '')
                source_value_str = str(source_value)
                current_value_str = str(current_value)
                
                # Determine if we should update
                if not overwrite:
                    # Only update if target field is empty
                    should_update = (current_value_str.strip() == '')
                    if not should_update:
                        skip_reason = "target not empty (overwrite=False)"
                elif overwrite_only_if_different:
                    # Only update if values are different
                    should_update = (source_value_str != current_value_str)
                    if not should_update:
                        skip_reason = "values are the same"
                else:
                    # Always update when overwrite=True and overwrite_only_if_different=False
                    should_update = True
                
                # Perform update if needed
                if should_update:
                    success = self.update_item(
                        item_key=item_key,
                        updates={target_field: source_value_str},
                        overwrite=overwrite,  # This controls the update_item behavior
                        verbose=False
                    )
                    if success:
                        results['updated'] += 1
                        if verbose:
                            print(f"✅ [{match_method}] {item_title[:50]}... -> {target_field}: '{source_value}'")
                    else:
                        results['update_failed'] += 1
                        if verbose:
                            print(f"❌ Update failed: {item_title[:50]}...")
                else:
                    # Track skipped items due to same value
                    if skip_reason == "values are the same":
                        results['skipped_same_value'] += 1
                    if verbose and skip_reason:
                        print(f"⏭️  Skipped ({skip_reason}): {item_title[:50]}...")
            else:
                if source_value is None:
                    results['no_match'] += 1
                    if verbose:
                        print(f"❌ No match: {item_title[:50]}...")
                else:
                    results['empty_value'] += 1
                    if verbose:
                        print(f"⚠️ Empty source value for: {item_title[:50]}...")
    
        # Print summary
        if verbose:
            print(f"\n📊 UPDATE SUMMARY:")
            print(f"   Field updated: {target_field}")
            print(f"   Source column: {source_column}")
            print(f"   Total items: {results['total_items']}")
            print(f"   Successfully updated: {results['updated']}")
            print(f"   Matched by key: {results['matched_by_key']}")
            print(f"   Matched by title: {results['matched_by_title']}")
            print(f"   Matched by DOI: {results['matched_by_doi']}")
            print(f"   No match found: {results['no_match']}")
            print(f"   Empty source values: {results['empty_value']}")
            print(f"   Update failures: {results['update_failed']}")
            print(f"   Skipped (same value): {results['skipped_same_value']}")
    
        return results


#%% CREATE CHILD COLLECTIONS (FOR LLMs COMPARAISON!)

    def create_child_collections(self, splits_count=3, samples_count=50, 
                                balance_field=None, random_state=None):
        """
        Create child collections (splits) with optional balancing by a field.
        
        Args:
            splits_count (int): Number of child collections to create
            samples_count (int): Items per child collection
            balance_field (str): Field to balance distributions across splits
            random_state (int): Seed for reproducible sampling
            
        Returns:
            list: ZoteroCollectionSubset objects representing child collections
        """
        # Initialize randomness and validate inputs
        random.seed(random_state)
        if splits_count <= 0 or samples_count <= 0:
            raise ValueError("splits_count and samples_count must be positive integers")
        
        # Prepare item indices for sampling
        indices = list(range(len(self.items)))
        available_indices = set(indices)
        
        # Group indices by balance field if specified
        field_groups = defaultdict(list)
        if balance_field and balance_field in self.df.columns:
            for idx, value in self.df[balance_field].items():
                field_groups[value].append(idx)
        else:
            field_groups['all'] = indices  # Treat as single group if no balancing
        
        child_collections = []
        for _ in range(splits_count):
            selected_indices = []
            
            # Balanced sampling logic
            if balance_field:
                # Calculate proportional allocation per group
                group_allocations = {}
                total_items = len(indices)
                for group, group_indices in field_groups.items():
                    proportion = len(group_indices) / total_items
                    group_allocations[group] = max(1, round(proportion * samples_count))
                
                # Adjust allocations to match sample size
                allocated_total = sum(group_allocations.values())
                allocation_diff = samples_count - allocated_total
                if allocation_diff != 0:
                    largest_group = max(field_groups, key=lambda k: len(field_groups[k]))
                    group_allocations[largest_group] += allocation_diff
                
                # Sample from each group
                for group, allocation in group_allocations.items():
                    available_in_group = list(set(field_groups[group]) & available_indices)
                    if not available_in_group:
                        continue
                    
                    n_to_sample = min(len(available_in_group), allocation)
                    sampled = random.sample(available_in_group, n_to_sample)
                    selected_indices.extend(sampled)
                    available_indices -= set(sampled)
            else:
                # Simple random sampling
                available_list = list(available_indices)
                if not available_list:
                    break
                    
                n_to_sample = min(len(available_list), samples_count)
                selected_indices = random.sample(available_list, n_to_sample)
                available_indices -= set(selected_indices)
            
            # Create child collection
            child_items = [self.items[i] for i in selected_indices]
            child_collection = ZoteroCollectionSubset(
                parent=self,
                items=child_items
            )
            child_collections.append(child_collection)
        
        return child_collections
    
#%% ZoteroCollectionSubset

class ZoteroCollectionSubset(ZoteroCollection):
    """Lightweight subset of a Zotero collection without reloading data"""
    def __init__(self, parent, items):
        """
        Initialize from parent collection and item subset
        
        Args:
            parent (ZoteroCollection): Original collection
            items (list): Subset of Zotero item dictionaries
        """
        self.zotero = parent.zotero
        self.collection_key = None  # Not a real Zotero collection
        self.items = items
        self.storage_folder = parent.storage_folder
        self.attachments = self._filter_attachments(parent.attachments)
        self.df = self.get_dataframe(verbose=False)
    
    def _filter_attachments(self, parent_attachments):
        """Filter attachments to only include items in this subset"""
        if not parent_attachments:
            return None
            
        item_keys = {item['key'] for item in self.items}
        return {
            key: paths for key, paths in parent_attachments.items()
            if key in item_keys
        }

