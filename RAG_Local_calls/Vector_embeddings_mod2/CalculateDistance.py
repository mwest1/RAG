from LLM_utils import generate_with_single_input
from VectorDistanceUtils import cosine_distance
from VectorDistanceUtils import euclidean_distance


def main():
    # Example vectors from a vector DB
    vector_a = [1.0, 2.0, 3.0]
    vector_b = [10000, 2321231, 312313]

    Cosinedistance = cosine_distance(vector_a, vector_b)
    print(f"Cosine Distance: {Cosinedistance}")

    EuclideanDistance = euclidean_distance(vector_a, vector_b)
    print(f"Euclidean Distance: {EuclideanDistance}")



if __name__ == "__main__":
    main()
