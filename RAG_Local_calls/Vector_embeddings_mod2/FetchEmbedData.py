import pandas as pd
import os
from sklearn.datasets import fetch_20newsgroups
import ssl
import certifi
import requests
from collections import Counter
import urllib.request
from sklearn.datasets._base import RemoteFileMetadata, _fetch_remote
from CreateVectorDB import CreateVectorDB
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer 
from CreateWeviate import create_weaviate_collection




# Configure SSL context with certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Custom download function using requests
def custom_download_20newsgroups(target_dir, cache_path, remote):
    try:
        # Use requests with certifi's CA bundle
        response = requests.get(remote.url, verify=certifi.where(), timeout=30)
        response.raise_for_status()  # Raise an error for bad status codes
        with open(cache_path, 'wb') as f:
            f.write(response.content)
        return cache_path
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")
        print("Trying with unverified SSL context (insecure)...")
        # Fallback: Disable SSL verification (not recommended)
        response = requests.get(remote.url, verify=False, timeout=30)
        response.raise_for_status()
        with open(cache_path, 'wb') as f:
            f.write(response.content)
        return cache_path

# Monkey-patch sklearn's _fetch_remote to use custom download
original_fetch_remote = _fetch_remote

def patched_fetch_remote(remote, dirname, verify_ssl=True):
    if verify_ssl:
        return original_fetch_remote(remote, dirname)
    else:
        # Use custom download function for unverified SSL
        cache_path = os.path.join(dirname, remote.filename)
        return custom_download_20newsgroups(dirname, cache_path, remote)


def FetchEmbedData():
    # Patch sklearn to use custom fetch
    from sklearn.datasets import _twenty_newsgroups
    _twenty_newsgroups._fetch_remote = patched_fetch_remote

    # Fetch the training dataset
    try:
        news = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
    except (urllib.error.URLError, requests.exceptions.RequestException) as e:
        print(f"Fetch failed: {e}")
        print("Retrying with unverified SSL (insecure)...")
        # Retry with unverified SSL
        _twenty_newsgroups._fetch_remote = lambda remote, dirname: patched_fetch_remote(remote, dirname, verify_ssl=False)
        news = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
    
    # create a DF from the data to make it easier to work with. 
    category_names = news.target_names
    
    df = pd.DataFrame({'text': news.data, 'category': news.target})

    # Add a 3rd column with the actual category name. This may be useful for metadata in the vector DB.
    

    for index, value in df['category'].items():
        df.at[index, 'category_name'] = category_names[value]
    

    
    # Load the model once, then pass it to the VectorDB creation function.
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    VectorDB = create_weaviate_collection(df)
    
    # # print(f"Database sample entry:\n{database[0]}")
    # # print(f"Database size: {len(database)}")
    print(VectorDB) 

if __name__ == "__main__":
    FetchEmbedData()


