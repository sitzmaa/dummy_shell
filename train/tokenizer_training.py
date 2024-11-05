from transformers import GPT2Tokenizer

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Save the tokenizer
tokenizer.save_pretrained('./models/gpt2_model')
