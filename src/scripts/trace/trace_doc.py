import json
import time
import os
import concurrent.futures
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# Paths for Chrome and Chromedriver
chrome_path = os.getenv("CHROME_PATH", os.path.expanduser("~/browsers/chrome/chrome"))
chromedriver_path = os.getenv("CHROMEDRIVER_PATH", os.path.expanduser("~/browsers/chromedriver/chromedriver"))

# File paths
LINKS_FILE = "data/retrieval_data/mathlib_init_links.txt"
OUTPUT_JSON = "data/retrieval_data/lean_definitions.json"
# FAILED_PAGES_FILE = "data/retrieval_data/failed_pages.txt"
INPUT_FILE = "data/retrieval_data/lean_definitions.pkl"
FULL_OUTPUT_FILE = "data/retrieval_data/partitioned_theorems.pkl"
PREFERRED_OUTPUT_FILE = "data/retrieval_data/preferred_partitioned.pkl"

# Parallelism Settings
MAX_WORKERS = 32  # Uses all 8 CPU cores
REQUEST_DELAY = 0.5  # Reduced delay to increase speed
MAX_RETRIES = 3  # Number of retries per failed request


def extract_definitions_from_page(page_url):
    """
    Extracts all definitions from a given Mathlib4 documentation page.
    Returns None if request fails after MAX_RETRIES.
    """
    # Set up Selenium
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.binary_location = chrome_path

    service = Service(chromedriver_path)

    for attempt in range(MAX_RETRIES):
        
        try:
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.get(page_url)
            time.sleep(REQUEST_DELAY)  # Prevents overloading

            # Find all definition blocks
            definition_blocks = driver.find_elements(By.CLASS_NAME, "decl")

            definitions = []
            for block in definition_blocks:
                try:
                    name = block.get_attribute("id").strip()

                    # Extract type signature
                    try:
                        type_signature = block.find_element(By.CLASS_NAME, "decl_header").text.strip()
                    except:
                        try:
                            type_signature = block.find_element(By.CLASS_NAME, "decl_type").text.strip()
                        except:
                            type_signature = "No type signature found"

                    # Extract description (all <p> tags inside the div)
                    description_elements = block.find_elements(By.TAG_NAME, "p")
                    description = "\n".join(p.text.strip() for p in description_elements if p.text.strip())

                    # Store result
                    definitions.append({
                        "definition_name": name,
                        "type_signature": f"{type_signature}" ,
                        "description" : f"{description}" if description else ""
                    })
                except Exception:
                    continue  # Skip this definition if there's an issue

            driver.quit()
            return definitions

        except Exception as e:
            print(f"⚠ Error fetching {page_url} (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(REQUEST_DELAY * 2)  # Wait before retrying

    return None  # Return None after MAX_RETRIES


def scrape_all_definitions():
    """
    Reads all links and extracts definitions in parallel using ProcessPoolExecutor.
    Saves results incrementally and retries failed pages.
    """
    # Read all stored URLs
    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        page_links = [line.strip() for line in f.readlines()]

    # Load previous progress
    processed_urls = set()
    all_definitions = []


    print(f"Total pages to scrape: {len(page_links)}")
    print(f"Skipping {len(processed_urls)} already processed pages.")

    failed_pages = []

    # Run in parallel using 8 CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(extract_definitions_from_page, url): url for url in page_links if url not in processed_urls}

        for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(future_to_url), desc="Scraping Lean Docs"):
            url = future_to_url[future]
            try:
                definitions = future.result()
                if definitions:
                    all_definitions.extend(definitions)
                    # Save after each successful page
                    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                        json.dump(all_definitions, f, indent=4)
                else:
                    failed_pages.append(url)
            except Exception:
                failed_pages.append(url)

    # # Save failed pages for retry
    # if failed_pages:
    #     with open(FAILED_PAGES_FILE, "w", encoding="utf-8") as f:
    #         for url in failed_pages:
    #             f.write(url + "\n")

    #     print(f"⚠ {len(failed_pages)} pages failed. Saved for retry in {FAILED_PAGES_FILE}")

    print(f"Scraping complete! All results saved in {OUTPUT_JSON}")

import pickle
PKL_FILE = "data/retrieval_data/lean_definitions.pkl"
# CLEANED_JSON_FILE = "lean_definitions_cleaned.json"  # Optional (to store cleaned JSON)

def load_json_definitions(input_file = OUTPUT_JSON):
    """Load definitions from a JSON file and transform into a dictionary."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    definitions =  {entry["definition_name"]: entry for entry in data}
    with open(PKL_FILE, 'wb') as f:
        pickle.dump(definitions, f)
    return {entry["definition_name"]: entry for entry in data}






import pickle
from collections import defaultdict



PREFERRED_NAMESPACES = [
    "Nat", "Int", "Rat", "Real", "Complex", "ENat", "NNReal", "EReal", "Monoid",
    "CommMonoid", "Group", "CommGroup", "Ring", "CommRing", "Field", "Algebra",
    "Module", "Set", "Finset", "Fintype", "Multiset", "List", "Fin", "BigOperators",
    "Filter", "Polynomial", "SimpleGraph.Walk", "Equiv", "Embedding", "Injective",
    "Surjective", "Bijective", "Order", "Topology"
]

def load_theorems(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def split_namespace(full_name):
    parts = full_name.split(".")
    definition_name = parts[-1]
    namespace = ".".join(parts[:-1])
    return namespace, definition_name

def partition_by_namespace(theorem_dict, filter_to_preferred=False):
    ns_dict = defaultdict(dict)

    for full_name, definition in theorem_dict.items():
        ns, _ = split_namespace(full_name)
        if filter_to_preferred and not any(
            ns == pref or ns.startswith(pref + ".") for pref in PREFERRED_NAMESPACES
        ):
            continue
        ns_dict[ns][full_name] = definition
    return dict(ns_dict)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

if __name__ == "__main__":
    # Run the script

    scrape_all_definitions()

    load_json_definitions()
    original = load_theorems(INPUT_FILE)
    full_partition = partition_by_namespace(original, filter_to_preferred=False)
    preferred_partition = partition_by_namespace(original, filter_to_preferred=True)

    save_pickle(full_partition, FULL_OUTPUT_FILE)
    save_pickle(preferred_partition, PREFERRED_OUTPUT_FILE)

    print(f"Saved full partitioned dictionary to {FULL_OUTPUT_FILE} ({len(full_partition)} namespaces)")
    print(f"Saved preferred-only partitioned dictionary to {PREFERRED_OUTPUT_FILE} ({len(preferred_partition)} namespaces)")
