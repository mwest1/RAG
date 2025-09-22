### this is helper function that creates the embedding using the
### BAAI/bge-base-en-v1.5 model

from sentence_transformers import SentenceTransformer


def EmbedText(text):
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    
    return model.encode(text)


