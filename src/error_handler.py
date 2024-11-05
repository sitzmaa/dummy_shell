import json

def load_error_handling():
    with open('data/error_handling.json') as f:
        return json.load(f)

error_handling = load_error_handling()

def handle_error(error_message):
    for key, value in error_handling.items():
        if key in error_message.lower():
            return value
    return "An unknown error occurred."
