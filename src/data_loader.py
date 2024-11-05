import json
from datasets import Dataset

def load_dataset(file_path):
    """
    Load training data from a JSON file and convert it to a Hugging Face Dataset.

    Args:
    file_path (str): Path to the JSON file containing training data.

    Returns:
    Dataset: Hugging Face Dataset object with the training data.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to Hugging Face Dataset format
    dataset = Dataset.from_dict({"text": [f"{item['input']} -> {item['output']}" for item in data]})
    
    return dataset
