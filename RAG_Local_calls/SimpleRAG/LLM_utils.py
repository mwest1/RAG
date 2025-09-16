# LLM_utils.py
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def generate_with_single_input(params):

    role_map = {
        "user": HumanMessage,
        "system": SystemMessage,
        "assistant": AIMessage,
    }
    MessageClass = role_map.get(params['role'], HumanMessage)
    messages = [MessageClass(content=params['prompt'])]
    
    llm = ChatOllama(
        model=params['model'] if 'model' in params else "eramax/salesforce-iterative-llama3-8b-dpo-r:Q5_K_M",
        temperature=params['temperature'] if 'temperature' in params else 0.7,
        top_p=params['top_p'] if 'top_p' in params else 0.9,
        max_tokens=params['max_tokens'] if 'max_tokens' in params else 1000,
        frequency_penalty=params['frequency_penalty'] if 'frequency_penalty' in params else 0.0,
        presence_penalty=params['presence_penalty'] if 'presence_penalty' in params else 0.0,
    )
    
    try:
        response = llm.invoke(messages)
        response_dict = response.model_dump()
        response_role = "assistant" if isinstance(response, AIMessage) else response.__class__.__name__
        response_dict["Role"] = response_role
        return response_dict
    except Exception as e:
        return {"error": f"Failed to generate response: {str(e)}"}