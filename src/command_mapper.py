import json

def load_command_mapping():
    with open('data/command_mapping.json') as f:
        return json.load(f)

command_mapping = load_command_mapping()

def map_to_command(user_input):
    for key, value in command_mapping.items():
        if key in user_input.lower():
            return value
    return None
