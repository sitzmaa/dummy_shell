from command_mapper import map_to_command
from command_executor import execute_command
from gpt_response import get_gpt_response

def dummyshell():
    print("Welcome to dummyshell! Type your commands in natural language.")
    
    while True:
        user_input = input(">>> ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        command = map_to_command(user_input)
        
        if command:
            output = execute_command(command)
            print(output)
        else:
            gpt_response = get_gpt_response(user_input)
            print("GPT Response:", gpt_response)

if __name__ == "__main__":
    dummyshell()
