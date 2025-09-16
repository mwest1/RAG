from LLM_utils import generate_with_single_input
from Prompt_content import read_json_params

def main():
    Prompt = input("What would you like help with today? ")
    print(f"Generating response for prompt: {Prompt}")
    # dictionary to capture all the necessary parameters
    params = {
    "role": "user",
    "temperature": 0.7,  # Adjust for more creative responses
    "top_p": 0.9,
    "max_tokens": 1000,
    "model": "eramax/salesforce-iterative-llama3-8b-dpo-r:Q5_K_M"
    #"model": "llama3.2:1b"
}
    # read in the data from the DB. 
    try:
        RAG_VectorDB = read_json_params("Houses.json")
    except Exception as e:
        print(f"Error loading prompt augmentation: {str(e)}")
        PromptAugment = ""
    
    #print("Prompt Augmentation:{}", RAG_VectorDB)

    # Create an augmented prompt, initailly just appending the JSON content 

    AugmentedPrompt = f"""Use the following housing data to answer the users query. \n
    {RAG_VectorDB}\n\n
    User Query: {Prompt}
    Provide a detailed and informative response based on the data provided. Answer as if you were a professional real estate agent. 
"""

    params['prompt'] = Prompt
    params['prompt'] = AugmentedPrompt

    # repeat the test for each model that has been pulled down 
    modelList = ["eramax/salesforce-iterative-llama3-8b-dpo-r:Q5_K_M", "llama3.2:1b","llama3.1:latest"]
    for model in modelList:
        print(f"\n\nGenerating response using model: {model}")
        params['model'] = model
    ## unpack the params dicationry using the ** operator 
        response = generate_with_single_input(**params)
        #print("Response:", response)
        ResponseTime= response['response_metadata']['total_duration']/1000000000
        print(f"Response time for model {model}: {ResponseTime} seconds")
        print(response['content'])


if __name__ == "__main__":
    main()
