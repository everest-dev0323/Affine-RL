import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load both models
model1 = AutoModelForCausalLM.from_pretrained("path/to/model1")
model2 = AutoModelForCausalLM.from_pretrained("path/to/model2")

# Get state dictionaries
state_dict1 = model1.state_dict()
state_dict2 = model2.state_dict()

# Create new state dict starting with model1 (backbone)
new_state_dict = state_dict1.copy()

# Replace MLP/FFN layers from model2
for key in state_dict2.keys():
    # Replace feed-forward network layers
    # Common patterns: 'mlp', 'feed_forward', 'ffn', 'fc1', 'fc2'
    if any(pattern in key for pattern in ['mlp', 'feed_forward', 'ffn']):
        new_state_dict[key] = state_dict2[key]
        print(f"Replaced: {key}")

# Load merged weights
merged_model = AutoModelForCausalLM.from_pretrained("path/to/model1")
merged_model.load_state_dict(new_state_dict)

# Save
merged_model.save_pretrained("path/to/merged_model")
tokenizer = AutoTokenizer.from_pretrained("path/to/model1")
tokenizer.save_pretrained("path/to/merged_model")