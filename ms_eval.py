# # NOT BATCHED 
# from unsloth import FastLanguageModel
# import torch
# import json
# from datasets import Dataset
# from transformers import TrainingArguments
# from evaluate import load
# import os
# os.environ["WANDB_DISABLED"] = "true"

# # Load the exact_match metric
# exact_match_metric = load("exact_match")
# rouge_metric = load("rouge")


# max_seq_length = 2048
# dtype = None
# load_in_4bit = True
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="unsloth/Meta-Llama-3.1-8B",
#     max_seq_length=max_seq_length,
#     dtype=dtype,
#     load_in_4bit=load_in_4bit,
# )

# model = FastLanguageModel.get_peft_model(
#     model,
#     r=16,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     lora_alpha=16,
#     lora_dropout=0,
#     bias="none",
#     use_gradient_checkpointing="unsloth",
# )

# EOS_TOKEN = tokenizer.eos_token

# FastLanguageModel.for_inference(model)

# dataset = []
# with open("chunk1.json", 'r') as f:
#     dataset = json.load(f)

# dataset = Dataset.from_list(dataset)

# def extract_answer(response_text):
#     """
#     Extracts the text after '### Answer:' from the generated response.
#     """
#     answer_start = response_text.find("### Answer:")
#     if answer_start != -1:
#         return response_text[answer_start + len("### Answer:"):].strip()
#     return "N/A"

# def generate_response(example):
#     prompt = """###Instruction:
# You are given a question and a selected passage that provides context. Provide a clear and concise answer to the question using only the information from the passage.\n### Passage:
# {}

# ### Question:
# {}

# ### Answer:
# {}"""
    
#     input_text = prompt.format(
#         example['selected_passages'], 
#         example['query'], 
#         "",
#     )
    
#     inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
    
#     outputs = model.generate(
#         **inputs, 
#         max_new_tokens=128,  
#         use_cache=True,
#         temperature=0.1,     
#         top_p=0.95,          
#         do_sample=True       
#     )

#     decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
#     generated_answer = extract_answer(decoded_output[0])
    
#     # Store the extracted answer for evaluation
#     example["generated_answer"] = generated_answer
#     return example

# # Select the first 10 samples from the dataset
# dataset_subset = dataset.select(range(10))

# # Apply the inference function only to these 10 samples
# dataset_subset = dataset_subset.map(generate_response)


# # Prepare to save data into a JSON file
# output_data = []

# # Collect predictions and references, and save generated answers and queries
# predictions = []
# references = []

# for example in dataset_subset:
#     predictions.append(example['generated_answer'])
#     references.append(example['answers'])
    
#     # Collect data to save in the output JSON file
#     output_data.append({
#         "context": example['selected_passages'],
#         "query": example['query'],
#         "generated_answer": example['generated_answer'],
#         "gold_answer": example['answers']
#     })

# # Save generated answers and queries to a JSON file
# with open("ms_qa_simple.json", "w") as outfile:
#     json.dump(output_data, outfile, indent=4)

# print("Saved generated answers and queries to 'ms_qa_simple.json'.")



# # # Collect predictions and references
# # predictions = [example['generated_answer'] for example in dataset_subset]
# # references = [example['answers'][0] if isinstance(example['answers'], list) else example['answers'] for example in dataset_subset]  # assuming 'answers' contain ground-truth


# # Print types of predictions and references to ensure they're lists of strings
# print(f"Type of predictions: {type(predictions)}")  # Should be <class 'list'>
# print(f"Type of each element in predictions: {[type(pred) for pred in predictions]}")  # Should be <class 'str'>

# print(f"Type of references: {type(references)}")  # Should be <class 'list'>
# print(f"Type of each element in references: {[type(ref) for ref in references]}")  # Should be <class 'str'>

# #references = [example['answers'] for example in dataset_subset]  # assuming 'answers' contain ground-truth

# # Print both the generated answers and gold answers for review
# for i, example in enumerate(dataset_subset):
#     print(f"Sample {i+1}:")
#     print(f"Generated Answer: {example['generated_answer']}")
#     print(f"Gold Answer: {example['answers']}")
#     print("-" * 50)


# # Compute exact match metric
# results = exact_match_metric.compute(predictions=predictions, references=references, ignore_case=True, ignore_punctuation=True)

# # Print the evaluation result
# print(f"Exact Match Score: {results['exact_match']}")


# # Compute ROUGE metric
# rouge_results = rouge_metric.compute(predictions=predictions, references=references)
# # Format and print ROUGE results to 4 decimal places
# formatted_rouge_results = {key: round(float(value), 4) for key, value in rouge_results.items()}
# print(f"ROUGE Scores: {formatted_rouge_results}")







# BATCHED
from unsloth import FastLanguageModel
import torch
import json
from datasets import Dataset
from transformers import TrainingArguments
from evaluate import load
import os
import time
import argparse

os.environ["WANDB_DISABLED"] = "true"
# Argument parser setup
parser = argparse.ArgumentParser(description='Run FOL Evaluation with a chosen prompt type.')
parser.add_argument('--prompt_type', type=int, required=True, help='Choose the prompt type (0, 1, 2, 3)')
args = parser.parse_args()
# Load the exact_match and rouge metrics
exact_match_metric = load("exact_match")
rouge_metric = load("rouge")

# Model and tokenizer setup
max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

FastLanguageModel.for_inference(model)

def extract_answer(response_text, prompt_type):
    """
    Extracts the text after the nth occurrence of '### Answer:' based on prompt_type.
    - If prompt_type is 0, 1, or 2, it extracts the text after the nth occurrence.
    - If prompt_type is 3, it extracts the text after the third occurrence of '### Answer:' and 
      also the reasoning part, which is located after '### Reasoning:'.
      
    Returns:
    - A tuple (answer, reasoning) where reasoning is None for prompt types 0, 1, 2.
    """
    def find_nth_occurrence(text, substring, n):
        """Helper function to find the nth occurrence of a substring in a text."""
        pos = -1
        for _ in range(n):
            pos = text.find(substring, pos + 1)
            if pos == -1:
                break
        return pos
    
    if prompt_type in [0, 1, 2]:
        # Find the nth occurrence of '### Answer:' based on prompt_type
        answer_start = find_nth_occurrence(response_text, "### Answer:", prompt_type + 1)
        if answer_start != -1:
            return response_text[answer_start + len("### Answer:"):].strip(), None
        
    elif prompt_type == 3:
        # Extract the third occurrence of '### Answer:' and capture until '### Reasoning:'
        answer_start = find_nth_occurrence(response_text, "### Answer:", 3)
        if answer_start != -1:
            reasoning_start = response_text.find("### Reasoning:", answer_start + len("### Answer:"))
            if reasoning_start != -1:
                # Extract everything between '### Answer:' and '### Reasoning:'
                answer = response_text[answer_start + len("### Answer:"):reasoning_start].strip()
                reasoning = response_text[reasoning_start + len("### Reasoning:"):].strip()
                return answer, reasoning
            else:
                # If no '### Reasoning:' is found, return everything after the third '### Answer:'
                answer = response_text[answer_start + len("### Answer:"):].strip()
                return answer, None
    
    return "Wrong prompt passed or not enough occurrences", None




def create_prompt(nl_context, nl_question, prompt_type):
    """
    Create a prompt based on the type of system instruction (four types).
    
    Parameters:
    - nl_context: The natural language passage or context.
    - nl_question: The natural language question.
    - gen_fol_premises: FOL premises extracted from the response.
    - gen_fol_conclusion: FOL conclusion extracted from the response.
    - prompt_type: An integer to choose which prompt to use (0, 1, 2, 3).
    
    Returns:
    - prompt: A formatted prompt string or "incorrect value for the prompt" if the prompt_type is invalid.
    """
    if prompt_type == 0:
        # Original prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are given a question and a selected passage that provides context. Provide a clear and concise answer to the question using only the information from the passage.<|eot_id|><|start_header_id|>user<|end_header_id|>
### Passage:
{nl_context}

### Question:
{nl_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Answer:
"""
    elif prompt_type == 1:
        # Alternative prompt with step-by-step reasoning
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are given a question and a selected passage that provides context. Provide a clear and concise answer to the question using only the information from the passage.<|eot_id|><|start_header_id|>user<|end_header_id|>

Follow this example:

Example:

### Passage:
Lactic acid, also known as 2-hydroxypropanoic or milk acid, is a compound formed when glucose is broken down under certain conditions in a living creature or by some types of bacteria.
In a person, for example, it is an important part of producing energy for strenuous exercise and helps with certain liver functions.
One common use for lactic acid in a human body is the formation of glucose.
Moderate amounts of this acid can move through someone's blood stream and reach the liver, where it undergoes a process called gluconeogenesis to become glucose.


### Question:
what is lactic acid

### Answer: 
It is a compound formed when glucose is broken down under certain conditions in a living creature or by some types of bacteria.<|eot_id|><|start_header_id|>user<|end_header_id|>

### Passage: 
{nl_context}


### Question:
{nl_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Answer:
"""
    elif prompt_type == 2:
        # Third prompt: Focus on logical consistency
       prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are given a question and a selected passage that provides context. Provide a clear and concise answer to the question using only the information from the passage.<|eot_id|><|start_header_id|>user<|end_header_id|>

Follow these examples:
Example 1:

### Passage:
Lactic acid, also known as 2-hydroxypropanoic or milk acid, is a compound formed when glucose is broken down under certain conditions in a living creature or by some types of bacteria.
In a person, for example, it is an important part of producing energy for strenuous exercise and helps with certain liver functions.
One common use for lactic acid in a human body is the formation of glucose.
Moderate amounts of this acid can move through someone's blood stream and reach the liver, where it undergoes a process called gluconeogenesis to become glucose.

### Question:
what is lactic acid


### Answer: 
It is a compound formed when glucose is broken down under certain conditions in a living creature or by some types of bacteria.

Example 2:

### Passage:
Biotin is a B vitamin that is sometimes referred to as vitamin H or vitamin B7.
It is one of the eight vitamins in the vitamin B-complex.
The B vitamins, in general, help in promoting healthy nerves, skin, eyes, hair, liver and a healthy mouth.
Biotin, also known as vitamin H or coenzyme R, is a water-soluble B-vitamin (vitamin B7).
It is composed of a ureido (tetrahydroimidizalone) ring fused with a tetrahydrothiophene ring.
A valeric acid substituent is attached to one of the carbon atoms of the tetrahydrothiophene ring.
Biotin is necessary for cell growth, the production of fatty acids, and the metabolism of fats and amino acids.
Biotin assists in various metabolic reactions involving the transfer of carbon dioxide.
It may also be helpful in maintaining a steady blood sugar level.

### Question:
is biotin a b vitamin

### Answer: 
Yes<|eot_id|><|start_header_id|>user<|end_header_id|>

### Passage: 
{nl_context}

### Question:
{nl_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Answer:
"""
    elif prompt_type == 3:
        # Fourth prompt: Focus on critical reasoning
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are given a question and a selected passage that provides context. 
Provide a clear answer in natural language to the question. Always provide a step-by-step reasoning and  use the information from the passage paired with the First-order Logic Translations.

Answer only in this format:
### Answer:
correct answer in natural language

### Reasoning:
step by step reasoning which lead to the answer

Follow this example:

Example:

### Passage:
Lactic acid, also known as 2-hydroxypropanoic or milk acid, is a compound formed when glucose is broken down under certain conditions in a living creature or by some types of bacteria.
In a person, for example, it is an important part of producing energy for strenuous exercise and helps with certain liver functions.
One common use for lactic acid in a human body is the formation of glucose.
Moderate amounts of this acid can move through someone's blood stream and reach the liver, where it undergoes a process called gluconeogenesis to become glucose.

### Question:
what is lactic acid

### Reasoning:
The question asks, "What is lactic acid" seeking its definition.
The passage introduces lactic acid as 2-hydroxypropanoic acid, a compound formed when glucose is broken down.
It specifies that this breakdown happens under specific conditions, either in living creatures or by bacteria.
The passage also mentions the role of lactic acid in energy production, liver function, and gluconeogenesis, but these details are not directly needed to define lactic acid.
The answer is formed by focusing on the key point: "It is a compound formed when glucose is broken down under certain conditions in a living creature or by some types of bacteria."

### Answer: 
It is a compound formed when glucose is broken down under certain conditions in a living creature or by some types of bacteria.

### Passage: 
{nl_context}

### Question:
{nl_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Answer:
"""
    else:
        # Return error message if prompt_type is not valid
        return "incorrect value for the prompt"

    return prompt

def generate_batch_responses(examples, prompt_type):
    """
    Generate responses in batches with a choice between four prompt types.
    
    Parameters:
    - prompt_type: An integer to choose which prompt to use (0, 1, 2, or 3).
    """
    # Check if prompt_type is valid
    if prompt_type not in [0, 1, 2, 3]:
        # Exit early with an error message
        print("wrong prompt")
        return examples  # Return the original examples without processing further

    prompts = []
    gold_answers = examples['answers']  # Assuming gold answers are available in examples
    
    for nl_context, nl_question in zip(examples['selected_passages'], examples['query']):
        # Generate the prompt using the external function
        prompt = create_prompt(nl_context, nl_question, prompt_type)
        
        # If the prompt is invalid, return early with an error message
        if prompt == "incorrect value for the prompt":
            print("wrong prompt")
            return examples  # Return the original examples without processing further
        
        # Add the generated prompt to the list
        prompts.append(prompt)

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
    
    # Decode and extract the generated answers and reasoning
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print("\nFull model outputs:\n", decoded_outputs)
    
    # Use the updated extract_answer function to get both answers and reasoning
    generated_answers_with_reasoning = [extract_answer(output, prompt_type) for output in decoded_outputs]
    
    # Split the generated answers and reasoning into separate lists
    generated_answers = [answer for answer, reasoning in generated_answers_with_reasoning]
    
    # Add the generated answers to the batch
    examples['generated_answer'] = generated_answers

    # Add reasoning only if prompt_type == 3
    if prompt_type == 3:
        # Handle reasoning only for prompt_type == 3
        generated_reasonings = [reasoning if reasoning is not None else "No reasoning available" for answer, reasoning in generated_answers_with_reasoning]
        examples['generated_reasoning'] = generated_reasonings

    # Print outputs for debugging
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i + 1}:")
        print(prompt)
        print(f"Generated Answer: {generated_answers[i]}")
        if prompt_type == 3:
            print(f"Generated Reasoning: {examples['generated_reasoning'][i]}")
        print(f"Gold Answer: {gold_answers[i]}")
    
    return examples




# Load the dataset
with open("chunk1.json", 'r') as f:
    dataset = json.load(f)

dataset = Dataset.from_list(dataset).select(range(8))  # First 300 of chunk1 or chunk1_0_299 for inference

# Batch process the dataset
batch_size = 8 

# Access the prompt_type
chosen_prompt = args.prompt_type

# This will return "incorrect value for the prompt" and not proceed further
dataset = dataset.map(lambda examples: generate_batch_responses(examples, prompt_type=chosen_prompt), batched=True, batch_size=batch_size)

# Prepare to save data into a JSON file
output_data = []
predictions = []
references = []

for example in dataset:
    predictions.append(example['generated_answer'])
    references.append(example['answers'])

    # Prepare the output data structure
    output_item = {
        "context": example['selected_passages'],
        "query": example['query'],
        "generated_answer": example['generated_answer'],
        "gold_answer": example['answers']
    }

    # Add generated_reasoning only if it exists (i.e., for prompt_type == 3)
    if 'generated_reasoning' in example:
        output_item["generated_reasoning"] = example['generated_reasoning']

    # Append to output data
    output_data.append(output_item)

# Save generated answers and queries to a JSON file
output_filename = f"no_train_qa_simple_base_prompt_{chosen_prompt}.json"
with open(output_filename, "w") as outfile:
    json.dump(output_data, outfile, indent=4)

print(f"Saved generated answers and queries to '{output_filename}'.")

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
