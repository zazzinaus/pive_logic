


# BATCHED
from unsloth import FastLanguageModel
import torch
import json
from datasets import Dataset
from transformers import TrainingArguments
from evaluate import load
import os
import time

os.environ["WANDB_DISABLED"] = "true"

# Load the exact_match and rouge metrics
exact_match_metric = load("exact_match")
rouge_metric = load("rouge")

# Model and tokenizer setup
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "fol_lora_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# Load the dataset
with open("chunk1_0_299_corrected_ms.json", 'r') as f:
    dataset = json.load(f)

dataset = Dataset.from_list(dataset)#.select(range(10)) # first 300 of chunk1  or chunk1_0_299 for inference

def extract_answer(response_text):
    """
    Extracts the text after '### Answer:' from the generated response.
    """
    answer_start = response_text.find("### Answer:")
    if answer_start != -1:
        return response_text[answer_start + len("### Answer:"):].strip()
    return "N/A"

def generate_batch_responses(examples):
    """
    Generate responses in batches.
    """
    prompts = [
        f"""###Instruction:
You are given a question and a selected passage that provides context. Provide a clear and concise answer to the question using only the information from the passage paired with the First-order Logic Translations.\n### Passage:
{passage}
### FOL Translation of passage: 
{gen_fol_premises}

### Question:
{query}

### FOL Translation of question:
{gen_fol_conclusion}

### Answer:
""" for passage, query, gen_fol_premises, gen_fol_conclusion in zip(examples['nl_context'], examples['nl_question'], examples['generated_fol_premises'], examples['generated_fol_conclusion'])
#for passage, query, gen_fol_premises, gen_fol_conclusion in zip(examples['selected_passages'], examples['query'], examples['generated_fol_premises'], example['generated_fol_conclusion'])
    ]
    
    # Tokenize the input prompts for the batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
    # Generate responses for the batch
    outputs = model.generate(
        **inputs, 
        max_new_tokens=128,  
        use_cache=True,
        temperature=0.1,     
        top_p=0.95,          
        do_sample=True       
    )
    
    # Decode and extract the generated answers
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_answers = [extract_answer(output) for output in decoded_outputs]
    
    # Add the generated answers to the batch
    examples['generated_answer'] = generated_answers
    
    return examples

# Batch process the dataset
batch_size = 8 
dataset = dataset.map(generate_batch_responses, batched=True, batch_size=batch_size)

# Prepare to save data into a JSON file
output_data = []
predictions = []
references = []

for example in dataset:
    predictions.append(example['generated_answer'])
    references.append(example['gold_answer'])
    
    output_data.append({
        "context": example['nl_context'],
        "query": example['nl_question'],
        "generated_answer": example['generated_answer'],
        "gold_answer": example['gold_answer']
    })

# Save generated answers and queries to a JSON file
with open("no_train_qa_fol_base.json", "w") as outfile:
    json.dump(output_data, outfile, indent=4)

print("Saved generated answers and queries to 'no_train_qa_fol_base.json'.")

# Flatten references
predictions_flattened = predictions
references_flattened = [item[0] if isinstance(item, list) else item for item in references]

# Compute exact match metric
results = exact_match_metric.compute(predictions=predictions_flattened, references=references_flattened, ignore_case=True, ignore_punctuation=True)
print(f"Exact Match Score: {results['exact_match']}")

# Compute ROUGE metric
rouge_results = rouge_metric.compute(predictions=predictions_flattened, references=references_flattened)
formatted_rouge_results = {key: round(float(value), 4) for key, value in rouge_results.items()}
print(f"ROUGE Scores: {formatted_rouge_results}")
 