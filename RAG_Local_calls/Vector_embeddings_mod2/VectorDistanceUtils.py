# this helper module includes functions to calculate the cosine and euclidean distances between two vectors
import numpy as np

def cosine_distance(vector_a, vector_b):
    # Convert lists to numpy arrays
    A = np.array(vector_a)
    B = np.array(vector_b)
    
    # Calculate dot product
    dot_product = np.dot(A, B)
    
    # Calculate norms
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    
    # Calculate cosine similarity
    cosine_similarity = dot_product / (norm_a * norm_b)
    
    # Calculate cosine distance
    cosine_dist = 1 - cosine_similarity
    
    return cosine_dist

def euclidean_distance(vector_a, vector_b):
    # Convert lists to numpy arrays
    A = np.array(vector_a)
    B = np.array(vector_b)
    
    # Calculate Euclidean distance
    euclidean_dist = np.linalg.norm(A - B)
    
    return euclidean_dist

