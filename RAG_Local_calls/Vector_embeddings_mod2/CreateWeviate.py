from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import weaviate
import weaviate.classes as wvc
from weaviate.util import generate_uuid5
from weaviate.exceptions import WeaviateStartUpError, WeaviateBaseError
import pandas as pd
import numpy as np
import os
from Embedding import process_row  # Your helper function

app = Flask(__name__)

# Configuration
WEAVIATE_URL = "http://localhost:8080"
UPLOAD_FOLDER = "./Uploads"
ALLOWED_EXTENSIONS = {"csv"}
MODEL_NAME = "BAAI/bge-base-en-v1.5"
CLASS_NAME = "NewsEmbedding"
model = SentenceTransformer(MODEL_NAME)


def create_weaviate_collection(df, class_name=CLASS_NAME):
    """
    Create a Weaviate collection from a DataFrame with text, category, and category_name columns.
# 
    Args:
        df: Pandas DataFrame with columns 'text', 'category', 'category_name'.
        class_name: Name of the Weaviate collection.

    Returns:
        collection: Weaviate collection object.
        count: Number of entries added.
    """
    try:
        # Connect to Weaviate
        with weaviate.connect_to_local() as client:
            # Define schema
            #client.schema.delete_class(class_name)  # Clear existing class (optional)
            print("Creating schema...")
            schema = {
                "class": class_name,
                "properties": [
                    {"name": "row_index", "dataType": ["string"]},
                    {"name": "text", "dataType": ["text"]},
                    {"name": "category", "dataType": ["int"]},
                    {"name": "category_name", "dataType": ["text"]},
                ],
                "vectorizer": "none",  # Use precomputed embeddings
                "vectorIndexConfig": {"distance": "cosine"}
            }
            client.schema.create_class(schema)

            collection = client.collection.get(class_name)

            # Print DataFrame rows for debugging
            print("DataFrame Contents:")
            for index, row in df.iterrows():
                print(f"Row {index}:")
                print(row.to_dict())
                print("---")

            # Generate embeddings
            print("Generating embeddings...")
            data_objects = []
            count = 0
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(process_row, index, row, model) for index, row in df.iterrows()]
                print(f"Submitted {len(futures)} tasks to the executor.")
                for future in futures:
                    index, result_dict = future.result()
                    embedding = result_dict["embedding"]
                    # Use DataFrame row for metadata to ensure consistency
                    metadata = {
                        "row_index": str(index),
                        "text": df.at[index, "text"],
                        "category": int(df.at[index, "category"]),  # Ensure int type
                        "category_name": df.at[index, "category_name"]
                    }
                    data_object = wvc.DataObject(
                        properties=metadata,
                        vector=embedding.tolist(),
                        uuid=generate_uuid5(str(index))
                    )
                    data_objects.append(data_object)
                    count += 1
                    print(f"Processed row {count}")

            # Batch insert into Weaviate
            print("Inserting into Weaviate collection...")
            with client.batch.dynamic() as batch:
                for data_object in data_objects:
                    batch.add_data_object(data_object=data_object, collection=class_name)

            print(f"Created Weaviate collection '{class_name}' with {count} entries.")
            return collection, count

    except WeaviateStartUpError:
        raise Exception("Failed to connect to Weaviate. Ensure the server is running at http://localhost:8080.")
    except WeaviateBaseError as e:
        raise Exception(f"Weaviate error: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error: {e}")
