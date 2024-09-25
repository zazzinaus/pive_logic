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

# from unsloth import FastLanguageModel
# import torch
# import json
# from datasets import Dataset
# from transformers import TrainingArguments
# from evaluate import load
# import os
# import time
# os.environ["WANDB_DISABLED"] = "true"

# # Load the exact_match and rouge metrics
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
#     use_gradient_checkpointing=True,  # Change if needed
# )

# #EOS_TOKEN = tokenizer.eos_token
# FastLanguageModel.for_inference(model)

# dataset = []
# with open("chunk1.json", 'r') as f:
#     dataset = json.load(f)

# dataset = Dataset.from_list(dataset).select(range(16))

# def extract_answer(response_text):
#     """
#     Extracts the text after '### Answer:' from the generated response.
#     """
#     answer_start = response_text.find("### Answer:")
#     if answer_start != -1:
#         return response_text[answer_start + len("### Answer:"):].strip()
#     return "N/A"

# def generate_batch_responses(examples):
#     # Print the size of the batch being processed
#     print(f"Processing batch with {len(examples['query'])} examples")
    
#     # The input batch will be a dictionary of lists, so we format the batch as a single list of input strings
#     prompts = [
#         f"""###Instruction:
# You are given a question and a selected passage that provides context. Provide a clear and concise answer to the question using only the information from the passage.\n### Passage:
# {passage}

# ### Question:
# {query}

# ### Answer:
# """ for passage, query in zip(examples['selected_passages'], examples['query'])
#     ]
    
#     # Print the first few prompts for debugging
#     print(f"Example Prompts:\n{prompts[:2]}")  # Print only the first 2 prompts for clarity
    
#     # Tokenize the batch of input prompts
#     inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
#     # Print the tokenized input details (first two inputs for clarity)
#     #print(f"Tokenized inputs (first 2):\n{inputs['input_ids'][:2]}")
    
#     # Generate responses for the batch
#     start_time = time.time()  # Start timing the generation
#     outputs = model.generate(
#         **inputs, 
#         max_new_tokens=128,  
#         use_cache=True,
#         temperature=0.1,     
#         top_p=0.95,          
#         do_sample=True       
#     )
#     end_time = time.time()  # End timing
    
#     # Print time taken for generation
#     print(f"Batch generation took {end_time - start_time:.2f} seconds")
    
#     # Decode and extract the generated answers
#     decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
#     # Print the model outputs (first 2 for brevity)
#     print(f"Model outputs (first 2):\n{decoded_outputs[:2]}")
    
#     # Apply extract_answer function to each output in the batch
#     generated_answers = [extract_answer(output) for output in decoded_outputs]
    
#     # Print the extracted answers for the first few examples
#     print(f"Extracted answers (first 2):\n{generated_answers[:2]}")
    
#     # Add the generated answers to the batch
#     examples['generated_answer'] = generated_answers
    
#     return examples

# # Apply batch processing on the dataset using the map function
# # This will run the generate_batch_responses function on each batch of examples
# batch_size = 8  # You can adjust the batch size as needed
# dataset = dataset.map(generate_batch_responses, batched=True, batch_size=batch_size)
# # Prepare to save data into a JSON file
# output_data = []
# predictions = []
# references = []

# for example in dataset:
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
# with open("ms_qa_simple_batched.json", "w") as outfile:
#     json.dump(output_data, outfile, indent=4)

# print("Saved generated answers and queries to 'ms_qa_simple_batched.json'.")

# # Print types of predictions and references
# print(f"Type of predictions: {type(predictions)}")  # Should be <class 'list'>
# print(f"Type of each element in predictions: {[type(pred) for pred in predictions]}")  # Should be <class 'str'>

# print(f"Type of references: {type(references)}")  # Should be <class 'list'>
# print(f"Type of each element in references: {[type(ref) for ref in references]}")  # Should be <class 'list'> or <class 'str'>

# # Compute exact match metric
# predictions_flattened = [item for sublist in predictions for item in sublist]  # Flatten nested list if necessary
# references_flattened = [item[0] if isinstance(item, list) else item for item in references]  # Handle nested references

# # Print flattened types to ensure they're lists of strings
# print(f"Type of flattened predictions: {type(predictions_flattened)}")
# print(f"Type of each element in flattened predictions: {[type(pred) for pred in predictions_flattened]}")

# print(f"Type of flattened references: {type(references_flattened)}")
# print(f"Type of each element in flattened references: {[type(ref) for ref in references_flattened]}")

# results = exact_match_metric.compute(predictions=predictions_flattened, references=references_flattened, ignore_case=True, ignore_punctuation=True)
# print(f"Exact Match Score: {results['exact_match']}")

# # Compute ROUGE metric
# rouge_results = rouge_metric.compute(predictions=predictions_flattened, references=references_flattened)
# formatted_rouge_results = {key: round(float(value), 4) for key, value in rouge_results.items()}
# print(f"ROUGE Scores: {formatted_rouge_results}")
