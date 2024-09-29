


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
import random
import numpy as np

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

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
    model_name = "fol_lora_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference




def extract_fol_parts(refined_response):
    """
    Extracts FOL premises and FOL conclusion from the refined_response.
    
    refined_response: The string containing FOL premises and conclusion.
    Returns:
        - gen_fol_premises: Everything between "FOL premises" and "FOL question"
        - gen_fol_conclusion: Everything after "FOL question"
    """

    # Default to "N/A" in case parts are not found
    gen_fol_premises = "N/A"
    gen_fol_conclusion = "N/A"

    premises_start = refined_response.find("FOL premises:")

    if premises_start != -1:
        question_start = refined_response.find("FOL question:", premises_start)
    
        if question_start != -1:
        # Extract FOL premises and conclusion
            gen_fol_premises = refined_response[premises_start + len("FOL premises:"):question_start].strip()
            gen_fol_conclusion = refined_response[question_start + len("FOL question:"):].strip()

    return gen_fol_premises, gen_fol_conclusion


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




def create_prompt(nl_context, nl_question, gen_fol_premises, gen_fol_conclusion, prompt_type):
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
You are given a question and a selected passage that provides context. 
Provide a clear and concise answer in natural language to the question using only the information from the passage paired with the First-order Logic Translations.<|eot_id|><|start_header_id|>user<|end_header_id|>

### Passage: 
{nl_context}

### FOL Translation of passage: 
{gen_fol_premises}

### Question:
{nl_question}

### FOL Translation of question:
{gen_fol_conclusion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Answer:
"""
    elif prompt_type == 1:
        # Alternative prompt with step-by-step reasoning
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are given a question and a selected passage that provides context. 
Provide a clear and concise answer in natural language to the question using the information from the passage paired with the First-order Logic Translations.

Follow this example:

Example:

### Passage:
Lactic acid, also known as 2-hydroxypropanoic or milk acid, is a compound formed when glucose is broken down under certain conditions in a living creature or by some types of bacteria.
In a person, for example, it is an important part of producing energy for strenuous exercise and helps with certain liver functions.
One common use for lactic acid in a human body is the formation of glucose.
Moderate amounts of this acid can move through someone's blood stream and reach the liver, where it undergoes a process called gluconeogenesis to become glucose.

### FOL Translation of passage: 
 ∀x (LacticAcid(x) ⊕ (Compound(x) ∧ FormedFromGlucose(x) ∧ FormedUnderConditions(x, LivingCreature(x) ∨ Bacteria(x))))
 ∀x (LacticAcid(x) ⊕ ProducedBy(x, EnergyProductionForStrenuousExercise))
 ∀x (LacticAcid(x) ⊕ ProducedBy(x, LiverFunction))\n∀x (LacticAcid(x) ⊕ FormedInto(x, Glucose))
 ∀x (LacticAcid(x) ⊕ Glucose(x) ⊕ Gluconeogenesis(x, Liver(x)))


### Question:
what is lactic acid

### FOL Translation of question:
LacticAcid(x)

### Answer: 
It is a compound formed when glucose is broken down under certain conditions in a living creature or by some types of bacteria.<|eot_id|><|start_header_id|>user<|end_header_id|>

### Passage: 
{nl_context}

### FOL Translation of passage: 
{gen_fol_premises}

### Question:
{nl_question}

### FOL Translation of question:
{gen_fol_conclusion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Answer:
"""
    elif prompt_type == 2:
        # Third prompt: Focus on logical consistency
       prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are given a question and a selected passage that provides context. 
Provide a clear and concise answer in natural language to the question using the information from the passage paired with the First-order Logic Translations.

Follow these examples:
Example 1:

### Passage:
Lactic acid, also known as 2-hydroxypropanoic or milk acid, is a compound formed when glucose is broken down under certain conditions in a living creature or by some types of bacteria.
In a person, for example, it is an important part of producing energy for strenuous exercise and helps with certain liver functions.
One common use for lactic acid in a human body is the formation of glucose.
Moderate amounts of this acid can move through someone's blood stream and reach the liver, where it undergoes a process called gluconeogenesis to become glucose.

### FOL Translation of passage: 
 ∀x (LacticAcid(x) ⊕ (Compound(x) ∧ FormedFromGlucose(x) ∧ FormedUnderConditions(x, LivingCreature(x) ∨ Bacteria(x))))
 ∀x (LacticAcid(x) ⊕ ProducedBy(x, EnergyProductionForStrenuousExercise))
 ∀x (LacticAcid(x) ⊕ ProducedBy(x, LiverFunction))\n∀x (LacticAcid(x) ⊕ FormedInto(x, Glucose))
 ∀x (LacticAcid(x) ⊕ Glucose(x) ⊕ Gluconeogenesis(x, Liver(x)))


### Question:
what is lactic acid

### FOL Translation of question:
LacticAcid(x)

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

### FOL Translation of passage: 
∀x (Biotin(x) ⊕ (B_Vitamin(x) ∨ VitaminH(x) ∨ VitaminB7(x)))\n∀x (B_Vitamin(x) ⊕ VitaminB_Complex(x))
∀x (VitaminB_Complex(x) ⊕ HealthyNerves(x) ∧ HealthySkin(x) ∧ HealthyEyes(x) ∧ HealthyHair(x) ∧ HealthyLiver(x) ∧ HealthyMouth(x))
∀x (Biotin(x) ⊕ Water_Soluble(x) ∧ B_Vitamin(x) ∧ Coenzyme_R(x))
∀x (Biotin(x) ⊕ Ureido_Ring(x) ∧ Tetrahydroimidizalone_Ring(x) ∧ Tetrahydrothiophene_Ring(x))
∀x (Biotin(x) ⊕ Valeric_Acid_Substituent(x))
∀x (Biotin(x) ⊕ Cell_Growth(x) ∧ Fatty_Acids_Production(x) ∧ Fats_Metabolism(x) ∧ Amino_Acids_Metabolism(x))
∀x (Biotin(x) ⊕ Carbon_Dioxide_Transfer(x))
∀x (Biotin(x) ⊕ Steady_Blood_Sugar_Level(x))

### Question:
is biotin a b vitamin

### FOL Translation of question:
B_Vitamin(x) ⊕ Biotin(x)

### Answer: 
Yes<|eot_id|><|start_header_id|>user<|end_header_id|>

### Passage: 
{nl_context}

### FOL Translation of passage: 
{gen_fol_premises}

### Question:
{nl_question}

### FOL Translation of question:
{gen_fol_conclusion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Answer:
"""
    elif prompt_type == 3:
        # Fourth prompt: Focus on critical reasoning
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are given a question and a selected passage that provides context. 
Provide a clear answer in natural language to the question. Always shoewa step-by-step reasoning in the proper space and use the information from the passage paired with the First-order Logic Translations.

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

### FOL Translation of passage: 
 ∀x (LacticAcid(x) ⊕ (Compound(x) ∧ FormedFromGlucose(x) ∧ FormedUnderConditions(x, LivingCreature(x) ∨ Bacteria(x))))
 ∀x (LacticAcid(x) ⊕ ProducedBy(x, EnergyProductionForStrenuousExercise))
 ∀x (LacticAcid(x) ⊕ ProducedBy(x, LiverFunction))\n∀x (LacticAcid(x) ⊕ FormedInto(x, Glucose))
 ∀x (LacticAcid(x) ⊕ Glucose(x) ⊕ Gluconeogenesis(x, Liver(x)))

### Question:
what is lactic acid

### FOL Translation of question:
LacticAcid(x)

### Answer: 
It is a compound formed when glucose is broken down under certain conditions in a living creature or by some types of bacteria.

### Reasoning:
The question asks for a definition of lactic acid.
The FOL translation of the question is LacticAcid(x), seeking the nature of lactic acid.
The passage describes lactic acid as a compound formed from glucose in living creatures or bacteria.
The relevant FOL statement is: ∀x (LacticAcid(x) ⊕ (Compound(x) ∧ FormedFromGlucose(x) ∧ FormedUnderConditions(x, LivingCreature(x) ∨ Bacteria(x)))).
This FOL expression confirms that lactic acid is a compound formed under certain conditions.
Other FOL statements are not directly related to the definition.
Therefore, the answer is: "It is a compound formed when glucose is broken down under certain conditions in a living creature or by some types of bacteria.<|eot_id|><|start_header_id|>user<|end_header_id|>

### Passage: 
{nl_context}

### FOL Translation of passage: 
{gen_fol_premises}

### Question:
{nl_question}

### FOL Translation of question:
{gen_fol_conclusion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

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
    gold_answers = examples['gold_answer']  # Assuming gold answers are available in examples
    
    
    for refined_response, nl_context, nl_question in zip(examples['refined_response'], examples['nl_context'], examples['nl_question']):
        # Extract FOL premises and conclusion from refined_response
        gen_fol_premises, gen_fol_conclusion = extract_fol_parts(refined_response)

        # Generate the prompt using the external function
        prompt = create_prompt(nl_context, nl_question, gen_fol_premises, gen_fol_conclusion, prompt_type)
        
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
    
    # Decode and extract the generated answers
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


    
    # generated_answers = [extract_answer(output, prompt_type) for output in decoded_outputs]
    
    # # Add the generated answers to the batch
    # examples['generated_answer'] = generated_answers

    # for i, prompt in enumerate(prompts):
    #     print(f"\nPrompt {i + 1}:")
    #     print(prompt)
    #     print(f"Generated Answer: {generated_answers[i]}")
    #     print(f"Gold Answer: {gold_answers[i]}")
    
    # return examples

# Load the dataset
with open("chunk1_0_299_corrected_ms.json", 'r') as f:
    dataset = json.load(f)

dataset = Dataset.from_list(dataset)#.select(range(8)) # first 300 of chunk1  or chunk1_0_299 for inference

# Batch process the dataset
batch_size = 8 

# Access the prompt_type
chosen_prompt = args.prompt_type

  # Invalid prompt type for testing

# This will return "incorrect value for the prompt" and not proceed further
dataset = dataset.map(lambda examples: generate_batch_responses(examples, prompt_type=chosen_prompt), batched=True, batch_size=batch_size)

# Prepare to save data into a JSON file
output_data = []
predictions = []
references = []

for example in dataset:
    predictions.append(example['generated_answer'])
    references.append(example['gold_answer'])
    
    output_item = {
        "context": example['nl_context'],
        "query": example['nl_question'],
        "fol_translation": example['refined_response'],
        "generated_answer": example['generated_answer'],
        "gold_answer": example['gold_answer']
    }
    if 'generated_reasoning' in example:
        output_item["generated_reasoning"] = example['generated_reasoning']

    # Append to output data
    output_data.append(output_item)

# Save generated answers and queries to a JSON file
output_filename = f"train_qa_fol_base_prompt_{chosen_prompt}.json"
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


# # Load the dataset
# with open("chunk1_0_299_corrected_ms.json", 'r') as f:
#     dataset = json.load(f)

# dataset = Dataset.from_list(dataset)#.select(range(10)) # first 300 of chunk1  or chunk1_0_299 for inference


# def extract_fol_parts(refined_response):
#     """
#     Extracts FOL premises and FOL conclusion from the refined_response.
    
#     refined_response: The string containing FOL premises and conclusion.
#     Returns:
#         - gen_fol_premises: Everything between "FOL premises" and "FOL question"
#         - gen_fol_conclusion: Everything after "FOL question"
#     """

#     # Default to "N/A" in case parts are not found
#     gen_fol_premises = "N/A"
#     gen_fol_conclusion = "N/A"

#     premises_start = refined_response.find("FOL premises:")

#     if premises_start != -1:
#         question_start = refined_response.find("FOL question:", premises_start)
    
#         if question_start != -1:
#         # Extract FOL premises and conclusion
#             gen_fol_premises = refined_response[premises_start + len("FOL premises:"):question_start].strip()
#             gen_fol_conclusion = refined_response[question_start + len("FOL question:"):].strip()

#     return gen_fol_premises, gen_fol_conclusion



# def extract_answer(response_text):
#     """
#     Extracts the text after '### Answer:' from the generated response.
#     """
#     answer_start = response_text.find("### Answer:")
#     if answer_start != -1:
#         return response_text[answer_start + len("### Answer:"):].strip()
#     return "N/A"


# def generate_batch_responses(examples):
#     """
#     Generate responses in batches.
#     """
#     prompts = []
#     for refined_response, nl_context, nl_question in zip(examples['refined_response'], examples['nl_context'], examples['nl_question']):
#         # Extract FOL premises and conclusion from refined_response
#         gen_fol_premises, gen_fol_conclusion = extract_fol_parts(refined_response)
        

#         prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are given a question and a selected passage that provides context. 
# Provide a clear and concise answer in natural language to the question using only the information from the passage paired with the First-order Logic Translations.<|eot_id|><|start_header_id|>user<|end_header_id|>

# ### Passage: 
# {nl_context}

# ### FOL Translation of passage: 
# {gen_fol_premises}

# ### Question:
# {nl_question}

# ### FOL Translation of question:
# {gen_fol_conclusion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# ### Answer:
# """
#         prompts.append(prompt)

#     # Tokenize the input prompts for the batch
#     inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
#     # Generate responses for the batch
#     outputs = model.generate(
#         **inputs, 
#         max_new_tokens=128,  
#         use_cache=True,
#         temperature=0.1,     
#         top_p=0.95,          
#         do_sample=True       
#     )
    
#     # Decode and extract the generated answers
#     decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     generated_answers = [extract_answer(output) for output in decoded_outputs]
    
#     # Add the generated answers to the batch
#     examples['generated_answer'] = generated_answers
    
#     return examples

# # Batch process the dataset
# batch_size = 8 
# dataset = dataset.map(generate_batch_responses, batched=True, batch_size=batch_size)

# # Prepare to save data into a JSON file
# output_data = []
# predictions = []
# references = []

# for example in dataset:
#     predictions.append(example['generated_answer'])
#     references.append(example['gold_answer'])
    
#     output_data.append({
#         "context": example['nl_context'],
#         "query": example['nl_question'],
#         "fol_translation": example['refined_response'],
#         "generated_answer": example['generated_answer'],
#         "gold_answer": example['gold_answer']
#     })

# # Save generated answers and queries to a JSON file
# with open("train_qa_fol_base.json", "w") as outfile:
#     json.dump(output_data, outfile, indent=4)

# print("Saved generated answers and queries to 'train_qa_fol_base.json'.")

# # Flatten references
# predictions_flattened = predictions
# references_flattened = [item[0] if isinstance(item, list) else item for item in references]

# # Compute exact match metric
# results = exact_match_metric.compute(predictions=predictions_flattened, references=references_flattened, ignore_case=True, ignore_punctuation=True)
# print(f"Exact Match Score: {results['exact_match']}")

# # Compute ROUGE metric
# rouge_results = rouge_metric.compute(predictions=predictions_flattened, references=references_flattened)
# formatted_rouge_results = {key: round(float(value), 4) for key, value in rouge_results.items()}
# print(f"ROUGE Scores: {formatted_rouge_results}")
 