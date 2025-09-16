# this function is written to load in data from a database  or other source to facilliate the augmentation of a prompt 
import json

def read_json_params(file_path: str) -> dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return {"error": f"File '{file_path}' not found."}
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON in '{file_path}'."}
    except Exception as e:
        return {"error": f"Failed to read '{file_path}': {str(e)}"}

# Example JSON file (params.json)
# {
#     "prompt": "Explain AI.",
#     "role": "user",
#     "temperature": 0.7,
#     "top_p": 0.9,
#     "max_tokens": 1000
# }


