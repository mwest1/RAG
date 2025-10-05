# test_weaviate.py
try:
    from weaviate.classes.config import Configure
    print("Import successful! Weaviate version:", __import__('weaviate').__version__)
except ImportError as e:
    print(f"Still failing: {e}")