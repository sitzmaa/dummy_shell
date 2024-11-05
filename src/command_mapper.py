import re

def map_to_command(user_input, model, tokenizer):
    # Replace file names with a placeholder
    user_input = re.sub(r'\b\w+\.txt\b', '<FILE>', user_input)
    
    inputs = tokenizer.encode(user_input, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50)
    command = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Replace the placeholder with the actual file name
    return re.sub('<FILE>', 'file.txt', command)
