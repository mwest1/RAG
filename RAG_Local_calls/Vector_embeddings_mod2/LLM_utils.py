# LLM_utils.py
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import Dict, Any  # Added for type hints

def generate_with_single_input(
    prompt: str,
    role: str = "user",
    top_p: float = 0.1,
    temperature: float = 0.1,
    max_tokens: int = 500,
    model: str = "llama3.1:8b",
    **kwargs: Any
) -> dict[str, Any]: 
    """
    Generates a response from an Ollama model using a single input prompt.

    Args:
        prompt (str): The input prompt for the model.
        role (str, optional): The role of the message ('user', 'system', 'assistant'). Defaults to 'user'.
        top_p (float, optional): Nucleus sampling parameter (0.0 to 1.0). Defaults to 0.1.
        temperature (float, optional): Randomness control (0.0 to 2.0). Defaults to 0.1.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 500.
        model (str, optional): The Ollama model name. Defaults to 'llama3.1:8b'.
        **kwargs: Additional arguments passed to ChatOllama.

    Returns:
        Dict[str, Any]: The model response as a dictionary, or an error message if failed.
    """
    ### Validate the inputs: 

    if not isinstance(prompt, str) or not prompt.strip():
            return {"error": "Prompt must be a non-empty string"}
    if role not in ["user", "system", "assistant"]:
            return {"error": f"Invalid role: {role}. Must be 'user', 'system', or 'assistant'"}
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            return {"error": "Temperature must be a number between 0.0 and 2.0"}
    if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
            return {"error": "Top_p must be a number between 0.0 and 1.0"}
    if not isinstance(max_tokens, int) or max_tokens <= 0:
            return {"error": "Max_tokens must be a positive integer"}

    role_map = {
        "user": HumanMessage,
        "system": SystemMessage,
        "assistant": AIMessage,
    }
    MessageClass = role_map.get(role, HumanMessage)
    messages = [MessageClass(content=prompt)]
    
    llm = ChatOllama(
        model=model,
        temperature=float(temperature),  # Ensure float
        top_p=float(top_p),              # Ensure float
        max_tokens=max_tokens,
        frequency_penalty=0.5,           # Consistent with earlier version
        presence_penalty=0.3,            # Consistent with earlier version
        **kwargs                         # Pass additional arguments
    )
    
    try:
        response = llm.invoke(messages)
        response_dict = response.model_dump()
        response_role = "assistant" if isinstance(response, AIMessage) else response.__class__.__name__
        response_dict["Role"] = response_role
        return response_dict
    except Exception as e:
        return {"error": f"Failed to generate response: {str(e)}"}