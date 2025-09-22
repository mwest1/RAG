# this function is used to generate the embeddings and create a simplified databse. 
# This database may be called to by other modules to find the most relevent data set(s) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from Embedding import EmbedText
from CreateEmbedding import CreateVectorDB
from CalculateDistance import cosine_distance
from CalculateDistance import euclidean_distance
from sklearn.datasets import fetch_20newsgroups


def RetrieveDocument(Prompt: str):
    # Example vectors from a vector DB
    vector_prompt = EmbedText(Prompt)
    #print(f"Vector for Prompt: {Prompt} is {vector_prompt}")

    VectorDB = CreateVectorDB()
    CosineDistanceDB = {}
    EuclideanDistanceDB = {}

    # Find the best prompt by creating a list=
    for key, value in VectorDB.items():
        # calculate the cosine distance and populate a DB (dictionary)
        CosineDistance = cosine_distance(vector_prompt, value)
        CosineDistanceDB[key] = CosineDistance
        # calculate the euclidean distance and populate a DB (dictionary)
        EuclideanDistance = euclidean_distance(vector_prompt, value)
        EuclideanDistanceDB[key] = EuclideanDistance
        # print(f"Text: {key}\nCosine Distance: {Cosinedistance}\nEuclidean Distance: {EuclideanDistance}\n")
     
     
    # print("Vector DB:", VectorDB)
    # print(CosineDistanceDB)
    # print(EuclideanDistanceDB)
    return VectorDB, CosineDistanceDB, EuclideanDistanceDB


def main():

    news = fetch_20newsgroups(subset='traibn', remove=('headers', 'footers', 'quotes'))



    # Prompt = input("What would you like help with today? ")
    # Results = RetrieveDocument(Prompt)
    # VectorDB = Results[0]
    # CosineDistanceDB = Results[1]
    # EuclideanDistanceDB = Results[2]
    # #print("Vector DB:", VectorDB)
    # print(CosineDistanceDB)
    # print(EuclideanDistanceDB)

    # # calculate the min distance for cosine 

    # min_key = min(CosineDistanceDB, key=CosineDistanceDB.get)
    # min_value = CosineDistanceDB[min_key]
    # print(f"The best document as per Cosine distance is  {min_key}, with distance: {min_value}")

    # # calculate the min distance for euclidean
    # min_key = min(EuclideanDistanceDB, key=EuclideanDistanceDB.get)
    # min_value = EuclideanDistanceDB[min_key]
    # print(f"The best document as per Euclidiean distance is  {min_key}, with distance: {min_value}")

    # # print the document in order from best to worst i.e min to max distance   
    # print("""--------------------------------------------""")
    # print(f"Documents in order of best to worst match for prompt: {Prompt}")
    # for key in sorted(CosineDistanceDB, key=CosineDistanceDB.get):
    #     print(f"Document: {key}, Cosine Distance: {CosineDistanceDB[key]}")
    
    # for key in sorted(EuclideanDistanceDB, key=EuclideanDistanceDB.get):
    #     print(f"Document: {key}, Euclidean Distance: {EuclideanDistanceDB[key]}")   




if __name__ == "__main__":
    main()
