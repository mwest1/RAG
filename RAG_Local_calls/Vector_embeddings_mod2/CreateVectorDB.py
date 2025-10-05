
from Embedding import process_row
from concurrent.futures import ThreadPoolExecutor
from weaviate.classes.config import Configure


"""

This is a helper function that iterates over a dataframe then calls the Embbedding helper function to create the embedding of each row. 

Multi-threading is used to spead up the process, which is controlled by the number of max_workers variable.

This Vector DB is then stored in a dictionary. 

"""

def CreateVectorDB(df,model):

    """
    Create a vector DB collection. 

    
    """


    """
    Create a vector database from DataFrame rows using multi-threading.
    """
    count = 0
    max_workers = 16 # Adjust based on system capabilities (e.g., CPU cores)
    
    print("Creating vector database...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for each row, call the process_row helper function to create the embedding vector
        futures = [executor.submit(process_row, index,row,model) for index, row in df.iterrows()]
        print(f"Submitted {len(futures)} tasks to the executor.")   
        # Collect results as they complete
      
        for future in futures:
            index, result_dict = future.result()
            database[index] = result_dict
            count += 1
            print(index)
            print(f"Processed row {count}")

    
    print(f"Created vector database with {len(database)} entries.")
    return database

