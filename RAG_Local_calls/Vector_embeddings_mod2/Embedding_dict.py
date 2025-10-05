### this is helper function that creates the embedding using the
### BAAI/bge-base-en-v1.5 model

from sentence_transformers import SentenceTransformer



def process_row(index,row,model):
    """
    This is a helpler function that processes a single Datafram row and generatees a dense vector. 
    The model is passed in to ensure that the model is only loaded once. Without this, the processing is very slow as it is reloaded for each row.
    For any real application is is impractical as it was taking 5+ seconds per row. 
 
    Returns a tuple of (index, result_dict) to avoid race conditions when updating the database.
    """
    text = row['text']
    category = row['category']
    vector = model.encode(text)  # Generate embedding using pre-loaded model. 
    return index, {'text': text, 'category': category, 'vector': vector}


if __name__ == "__main__":
    process_row()