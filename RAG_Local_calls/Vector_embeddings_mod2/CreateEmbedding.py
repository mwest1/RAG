# this function is used to generate the embeddings and create a simplified databse. 
# This database may be called to by other modules to find the most relevent data set(s) 

from Embedding import EmbedText
# from CalculateDistance import cosine_distance
# from CalculateDistance import euclidean_distance


def CreateVectorDB():
    
    TextList = ['The capital of France is Paris', 'Pizza must be cooked for 10 minutes in the oven', 'Chicken must be heated to 80c to be fully cooked']
    
    # Create "empty" DB 
    VectorDB = {}
    for text in TextList:
        vector = EmbedText(text)
        VectorDB[text] = vector

    print("Vector DB:", VectorDB)
    return VectorDB



#     # calculate the distance from the prompt to each of the text items in the dB

#     Cosinedistance_list = []
#     Euclideandistance_list = []    


# # 
#     Cosinedistance = cosine_distance(vector_a, vector_b)
#     print(f"Cosine Distance: {Cosinedistance}")

#     EuclideanDistance = euclidean_distance(vector_a, vector_b)
#     print(f"Euclidean Distance: {EuclideanDistance}")



if __name__ == "__main__":
    main()
