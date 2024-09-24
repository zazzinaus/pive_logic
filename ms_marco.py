import json
from datasets import load_dataset
from tqdm import tqdm
import random

# Load the dataset
print("Starting to load the dataset...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split='train')
print("Dataset loaded successfully!")

# Initialize a list to store the filtered data
filtered_data = []

# Loop through the dataset and filter the required passages
for sample in tqdm(ds):
    selected_passages = [sample['passages']['passage_text'][i] for i, is_selected in enumerate(sample['passages']['is_selected']) if is_selected == 1]
    
    # If there are selected passages, store the relevant data
    if selected_passages:
        filtered_sample = {
            'answers': sample['answers'],
            'selected_passages': selected_passages,
            'query': sample['query'],
            'query_id': sample['query_id'],
            'query_type': sample['query_type']
        }
        filtered_data.append(filtered_sample)

# Save the filtered data to a JSON file
with open("filtered_ms_marco.json", "w") as f:
    json.dump(filtered_data, f, indent=4)

print("Filtered data saved")

# Load the JSON file
with open("filtered_ms_marco.json", "r") as f:
    filtered_data = json.load(f)

# Check if the dataset contains at least 10,000 elements
num_samples = min(10000, len(filtered_data))

# Randomly sample 10,000 elements from the filtered data
sampled_data = random.sample(filtered_data, num_samples)

# Function to add a newline after each period
def add_newline_after_period(text):
    return text.replace(". ", ".\n")

# Modify "answers" and "selected_passages" 
for item in sampled_data:
    # Convert "answers" list to a string (no newline modification here)
    item["answers"] = " ".join(item["answers"])
    
    # Convert "selected_passages" list to a string and add newline after each period
    passages_string = " ".join(item["selected_passages"])
    item["selected_passages"] = add_newline_after_period(passages_string)

# Save the modified sampled data back to a new file
with open("ms_marco_10k.json", "w") as f:
    json.dump(sampled_data, f, indent=4)

# Print the result to verify
print("Data has been modified and saved as 'ms_marco_10K.json'")




# Load the modified JSON file
with open("ms_marco_10k.json", "r") as f:
    filtered_data = json.load(f)

# Ensure we have enough samples to split
if len(filtered_data) == 10000:
    # Split the dataset into two chunks of 5000 samples each
    chunk1 = filtered_data[:5000]
    chunk2 = filtered_data[5000:10000]
    
    # Save the first chunk of 5000 samples to 'chunk1.json'
    with open("chunk1.json", "w") as f1:
        json.dump(chunk1, f1, indent=4)
    print("First 5000 samples saved as 'chunk1.json'")

    # Save the second chunk of 5000 samples to 'chunk2.json'
    with open("chunk2.json", "w") as f2:
        json.dump(chunk2, f2, indent=4)
    print("Second 5000 samples saved as 'chunk2.json'")
else:
    print(f"Not enough samples to split. Found only {len(filtered_data)} samples.")