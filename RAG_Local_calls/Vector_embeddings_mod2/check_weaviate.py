import weaviate
from weaviate.exceptions import WeaviateStartUpError, WeaviateBaseError

try:
    # Connect to local Weaviate instance using context manager
    with weaviate.connect_to_local() as client:
        print(client.is_ready())  # Should print: True
except WeaviateStartUpError:
    print("Error: Could not connect to Weaviate. Ensure the Weaviate server is running at http://localhost:8080.")
except WeaviateBaseError as e:
    print(f"Weaviate error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")