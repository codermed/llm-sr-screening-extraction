import re
from difflib import SequenceMatcher
import csv
from tqdm import tqdm
from pyzotero import zotero

class Zotero:
    def __init__(self, api_key, user_id, library_type="user"):
        """
        Initialize the ZoteroManager with API credentials.
        
        Args:
            api_key (str): Your Zotero API key.
            user_id (str): Your Zotero User ID or Group ID.
            library_type (str): "user" for personal libraries, "group" for group libraries.
        """
        self.zot = zotero.Zotero(user_id, library_type, api_key)

    def list_collections(self):
        """
        List all collections in the Zotero library along with their keys.
        
        Returns:
            dict: A dictionary mapping collection names to their keys.
        """
        collections = self.zot.collections()
        collection_dict = {}
        print("Collections:")
        for collection in collections:
            name = collection['data']['name']
            key = collection['key']
            collection_dict[name] = key
            print(f"- Name: {name}, Key: {key}")
        return collection_dict

    def get_collection_key_by_name(self, collection_name):
        """
        Retrieve the key(s) of a collection by its name.
        
        Args:
            collection_name (str): The name of the target collection.
        
        Returns:
            list: A list of keys for collections with the specified name.
                  Returns an empty list if no matching collection is found.
        """
        collections = self.zot.collections()
        matching_keys = []
        for collection in collections:
            name = collection['data']['name']
            key = collection['key']
            if name == collection_name:
                matching_keys.append(key)
        if not matching_keys:
            print(f"No collection found with the name '{collection_name}'.")
        else:
            print(f"Found {len(matching_keys)} collection(s) with the name '{collection_name}': {matching_keys}")
        return matching_keys
    
    def get_items_from_collection(self, collection_key):
        """
        Retrieve all items from a specific collection.
        
        Args:
            collection_key (str): The key of the target collection.
        
        Returns:
            list: A list of items in the collection.
        """
        print(f"Retrieving items from collection with key: {collection_key}...")
        items = self.zot.everything(self.zot.collection_items(collection_key))
        print(f"Found {len(items)} items in the collection.")
        return items

    def load_csv_assessments(self, csv_file):
        """
        Load verdicts from a CSV file and normalize titles.
        
        Args:
            csv_file (str): Path to the CSV file.
        
        Returns:
            dict: A dictionary mapping normalized titles to verdicts.
        """
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            assessments = {self.normalize_title(row['title']): row['verdict'] for row in reader}
        return assessments

    def update_zotero_items(self, items, assessments, field, verbose=True):
        """
        Update Zotero items with verdicts from the CSV file.
        
        Args:
            items (list): A list of Zotero items to process.
            assessments (dict): A dictionary mapping normalized titles to verdicts.
            verbose (bool): Whether to display detailed log messages for each updated item.
        
        Returns:
            int: The number of updated items.
        """
        updated_count = 0
    
        # Wrap the items list with tqdm to show a progress bar
        for item in tqdm(items, desc="Updating Zotero Items", unit="item"):
            # Extract the title of the Zotero item
            zotero_title = item.get('data', {}).get('title', '')
            if not zotero_title:
                continue  # Skip items without a title
    
            # Normalize the Zotero title
            normalized_zotero_title = self.normalize_title(zotero_title)
    
            # Check if the normalized title exists in the CSV assessments
            verdict = None
            if normalized_zotero_title in assessments:
                verdict = assessments[normalized_zotero_title]
            else:
                # Use fuzzy matching if exact match fails
                best_match = None
                best_similarity = 0.0
                for csv_title, csv_verdict in assessments.items():
                    sim = self.similarity(normalized_zotero_title, csv_title)
                    if sim > best_similarity and sim >= 0.8:  # Threshold for similarity
                        best_similarity = sim
                        best_match = csv_verdict
                if best_match:
                    verdict = best_match
    
            # If a verdict was found, update the Zotero item
            if verdict:
                # Update the 'extra' field with the verdict (or any other field you prefer)
                item['data'][field] = verdict  # You can use another field like 'tags' or 'collections'
                self.zot.update_item(item)
                updated_count += 1
    
                # Display log message only if verbose is True
                if verbose:
                    tqdm.write(f"Updated item '{zotero_title}' with verdict: {verdict}")  # Use tqdm.write to avoid breaking the progress bar
    
        return updated_count


    @staticmethod
    def normalize_title(title):
        """
        Normalize a title by removing LaTeX formatting, { }, and extra spaces.
        
        Args:
            title (str): The title to normalize.
        
        Returns:
            str: The normalized title.
        """
        if not title:
            return ""
        # Remove curly braces, LaTeX formatting, HTML tags, and extra spaces
        normalized = re.sub(r'[{}]', '', title)  # Remove {}
        normalized = re.sub(r'\\textit\{.*?\}', '', normalized)  # Remove \textit{}
        normalized = re.sub(r'\$_\{.*?\}\$', '', normalized)  # Remove $_{\textrm{...}}$
        normalized = re.sub(r'<[^>]+>', '', normalized)  # Remove HTML tags like <sub>, <i>
        normalized = re.sub(r'\s+', ' ', normalized).strip()  # Normalize whitespace
        return normalized.lower()  # Convert to lowercase for case-insensitive matching

    @staticmethod
    def similarity(a, b):
        """
        Calculate similarity between two strings using SequenceMatcher.
        
        Args:
            a (str): First string.
            b (str): Second string.
        
        Returns:
            float: Similarity ratio between 0 and 1.
        """
        return SequenceMatcher(None, a, b).ratio()
