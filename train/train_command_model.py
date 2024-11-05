import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from data_loader import load_dataset


# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the pad token to the EOS token (or you can add a new pad token)
tokenizer.pad_token = tokenizer.eos_token

# Alternatively, add a new special pad token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Load and tokenize the dataset
dataset = load_dataset('../data/command_mapping.json')

def tokenize_function(examples):
    tokens = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='../models/command_model',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('../models/command_model')
tokenizer.save_pretrained('../models/command_model')
