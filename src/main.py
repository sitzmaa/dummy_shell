import sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from command_mapper import map_to_command
from response_mapper import map_to_response
from command_executor import execute_command

# Load the command interpretation model
command_tokenizer = GPT2Tokenizer.from_pretrained('../models/command_model')
command_model = GPT2LMHeadModel.from_pretrained('../models/command_model')

# Load the response interpretation model
response_tokenizer = GPT2Tokenizer.from_pretrained('../models/error_model')
response_model = GPT2LMHeadModel.from_pretrained('../models/error_model')

def dummyshell(test_mode=False):
    print("Welcome to dummyshell! Type your commands in natural language.")
    
    while True:
        user_input = input(">>> ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        command = map_to_command(user_input, command_model, command_tokenizer)
        
        if test_mode:
            print("Generated command:", command)
        else:
            terminal_output = execute_command(command)
            response = map_to_response(terminal_output, response_model, response_tokenizer)
            print(response)

if __name__ == "__main__":
    test_mode = '--test' in sys.argv
    dummyshell(test_mode)
