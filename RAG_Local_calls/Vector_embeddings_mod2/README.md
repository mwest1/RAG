This module is the first with dealing with Vectors which will lead into understanding of how Vector Databases are leveraged during a semantic searh. 

Initially the exercise is to perform a simple calculation of the "Distance" between vectors, including Cosine or Ecludian distance. 

Cosine distance calculates based on the angles between them. i.e the same angle means the the two vectors are similar with a value close to 0 meaning they are the same.

 Cosine distance = 1 - Cosine Similarity(𝐪,𝐝𝑖)=𝐪⋅𝐝𝑖‖𝐪‖‖𝐝𝑖‖


Euclidean distance - This calculates the straight line distance between two vectors in "space". Smaller values equal greater similarity.

Euclidean Distance(𝐪,𝐝𝑖)=∑𝑗=1𝑛(𝑞𝑗−𝑑𝑖𝑗)2


Documents are transformed into a Vector Database using an embedding model.

A Users query is then converted to Vector, then using one of the two methods above, the closest "document" to the query can be found. Typically the top n documents are found. The results of the semantic search may be combined with a keyword search then then finally put through a metadata filter at the last stage. The top N documents are then returned to the RAG system to augment the users prompt. 

================
Embedding model
================

The model embeddeds words or sentences into a fixed size vector. The model is typically trained on millions of samples and specialises in grouping related sentences. 

In this example the BAAI/bgp-base-en-V1.5 model will be used. This embeds entire sentences with 768 dimensions. 

The Embedding module creates the embedding, with the CreateEmbedding script taking two simply sentences 

It is important that when embedding data, that the model is only called once, with each piece of data then passed to it. If the model is loaded every time, this will result in a very low creation of dense vectors, rendering the solution infeasible unless large compute resources are available. This is counter to the idead of this project which is to make a create a fully functional LLM for certain use cases that can be run on a laptop. 

The first time the model is loaded it will be downloaded from a remote server. 



====================
retrieval metrics
====================

