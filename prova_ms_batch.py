import os
import json
import torch
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
from correction import FOLRefiner

# QLoRA configuration
lora_config = LoraConfig(
    r=16, 
    lora_alpha=16, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run NL to FOL Translation")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the input dataset (JSON file)')
    parser.add_argument('--output', type=str, required=True, help='Path to the output JSON file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--sample_limit', type=int, default=-1, help='Limit number of samples to process (set to -1 for no limit)')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for the dataset chunk')
    parser.add_argument('--end_index', type=int, default=-1, help='End index for the dataset chunk (set to -1 to process till the end)')
    parser.add_argument('--continuous', type=bool, default=True, help='Flag to indicate if continuous inference should be performed')
    
    return parser.parse_args()

def safe_execute_program(premises, conclusion):
    """Safely execute the logic program and return any invalid premises or conclusion."""
    program = FOL_Prover9_Program(premises, conclusion)
    
    if not program.flag:
        invalid_data = {
            "invalid_premises": program.invalid_premises,
            "invalid_conclusion": program.invalid_conclusion
        }
        return None, 'parsing error', '', invalid_data
    
    answer, error_message, unification_stack = program.execute_program()
    
    if answer is None:
        # Even on execution error, return an empty dictionary for invalid_data if it's not about parsing.
        return answer, 'execution error', error_message, {}

    return program.answer_mapping(answer), '', '', unification_stack


def inference_on_generated_fol(generated_premises, generated_conclusion):
    #print(f"\nExecuting FOL prover with premises:\n{generated_premises}\nand conclusion:\n{generated_conclusion}\n")
    
    # Always unpack 4 variables from safe_execute_program
    answer, flag, error_message, invalid_data = safe_execute_program(generated_premises, generated_conclusion)

    if flag == 'parsing error':
        # Print the parsing error and return default values for prover_answer, prover_status, and prover_error
        print(f"Parsing error: {invalid_data}")
        
        # Return None for prover_answer, 'Parsing error' for status, the error message, and invalid_data
        return None, 'Parsing error', error_message, invalid_data

    # Continue as usual when there is no parsing error
    answer_mapping = {
        'A': "True",
        'B': "False",
        'C': 'Uncertain'
    }
    prover_answer = answer_mapping.get(answer, None)  # Default to None if not found

    # Return prover_answer, flag, error_message, and invalid_data
    return prover_answer, flag, error_message, invalid_data


class BatchData:
    """Helper class to store and manage batch data."""
    def __init__(self, batch):
        self.contexts = [item['selected_passages'] for item in batch]
        self.questions = [item['query'] for item in batch]
        self.gold_answers = [item['answers'] for item in batch]
        self.example_ids = [item['query_id'] for item in batch]

class TranslatorEngine:
    def __init__(self, dataset, output_filename, batch_size, sample_limit=None, start_index=0, end_index=-1, continuous=True):
        self.dataset = dataset
        self.output_filename = output_filename
        self.batch_size = batch_size
        self.sample_limit = sample_limit
        self.start_index = start_index
        self.end_index = end_index
        self.continuous = continuous

        self.max_seq_length = 2048
        self.dtype = torch.float16
        self.load_in_4bit = True

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        print("Tokenizer loaded.")

        # Assign eos_token as pad_token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct", 
            device_map="auto",
            load_in_4bit=self.load_in_4bit
        )
        print("Model loaded.")

        # Prepare model for LoRA
        print("Preparing model for LoRA...")
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        print("LoRA configuration applied.")

        # Move model to the appropriate device
        print(f"Moving model to {'GPU' if torch.cuda.is_available() else 'CPU'}...")
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("Model ready.")
                # Instantiate FOLRefiner for continuous inference
        print("Instantiating FOLRefiner...")
        self.refiner = FOLRefiner(
            model_name="LorMolf/LogicLlama2-chat-direct",  # FOL correction model
            template_filename='good_folio'
        )
        print("FOLRefiner initialized.")


    def load_data(self):
        print(f"Loading dataset: {self.dataset}...")

        with open(self.dataset, "r") as f:
            dataset = json.load(f)

        if self.sample_limit > 0:
            dataset = dataset[:self.sample_limit]

        if self.end_index == -1:
            self.end_index = len(dataset)

        dataset_chunk = dataset[self.start_index:self.end_index]
        print(f"Loaded {len(dataset_chunk)} examples from chunk {self.start_index} to {self.end_index-1} from {self.dataset}.")
        return dataset_chunk

    def format_prompt(self, nl_premise, nl_conclusion):
        prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Translate the following natural language (NL) premises and conclusions to first-order logic (FOL) rules.
Only translate the sentences to FOL, do not include any other information. Only include the FOL rule in the answer box.
Do not add any extra symbols or words to the FOL rule. Do not include any kind of explanation or description in the answer box.
You can use only these symbols for the FOL statements: '⊕', '∨', '∧', '→', '↔', '∀', '∃', '¬' and nothing else. Do not use "=", "?", "≥", "≤", "-", "_", "$". Other symbols will lead to a parsing error.

Respond only in the format:
FOL premises: 
NL premises translated into FOL

FOL question: 
NL conclusion translated into FOL.

Follow this example:

NL premises:
Cornish Hens.
Cornish chickens are a large English breed with white, black or red feathers.
They are a heritage breed, meaning they've been bred for many years in a particular location with traits conducive to thriving in that environment.
They are now raised by backyard enthusiasts and small farm

NL question:
what is a cornish hen

FOL premises: 
∀x (CornishHen(x) ⊕ Chicken(x)) 
∀x (CornishHen(x) ⊕ (Large(x) ∧ English(x) ∧ (Feathers(x, White) ∨ Feathers(x, Black) ∨ Feathers(x, Red))))
∀x (CornishHen(x) ⊕ HeritageBreed(x))
∀x (CornishHen(x) ⊕ RaisedBy(BackyardEnthusiasts(x) ∨ RaisedBy(SmallFarm(x))))

FOL question: 
CornishHen(x)

You are a helpful assistant.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
NL premises:
{}
NL question:
{}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{}"""
        return prompt.format(nl_premise, nl_conclusion, "") + self.tokenizer.eos_token

    def extract_response(self, text):
        return text.split("assistant")[2].strip()

    def translate_batch(self, batch_data):
        print(f"\nProcessing batch, generating FOL translation...\n")

        # Step 1: Prepare batch prompt inputs
        batch_prompt_inputs = [
            self.format_prompt(premises, conclusion)
            for premises, conclusion in zip(batch_data.contexts, batch_data.questions)
        ]

        json_ms_marco = []
        finished = [False] * len(batch_data.contexts)  # Track which samples are finished (prover_answer == True)

        # Repeat the refinement process up to 3 times
        for iteration in range(3):
            print(f"\nRefinement iteration {iteration + 1} for the batch...")

            # Step 2: Tokenize and generate for the entire batch
            inputs = self.tokenizer(batch_prompt_inputs, return_tensors="pt", padding=True, truncation=True).to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=256, use_cache=True)
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Step 3: Process each example individually for "inference_on_generated_fol"
            prover_results = []  # To store results of inference per sample
            for idx, generated_text in enumerate(generated_texts):
                if finished[idx]:  # Skip examples that have already been proven
                    continue

                response = self.extract_response(generated_text)

                # Parse the generated FOL premises and conclusion
                if "FOL premises:" in response and "FOL question:" in response:
                    parts = response.split("FOL question:")
                    generated_premises = parts[0].split("FOL premises:")[1].strip()
                    generated_conclusion = parts[1].strip()
                else:
                    generated_premises = ""
                    generated_conclusion = ""

                prover_answer, prover_status, prover_error, invalid_data = None, None, None, None
                if self.continuous:
                    # Step 4: Run inference for this single example
                    print(f"Running inference on generated FOL for example {idx + 1}/{len(generated_texts)}...")
                    prover_answer, prover_status, prover_error, invalid_data = inference_on_generated_fol(
                        generated_premises, generated_conclusion)

                    # If Prover9 proves the FOL, mark this example as finished
                    if prover_answer == "True":
                        print(f"FOL proved by Prover9 for example {idx + 1}, stopping further refinement.")
                        finished[idx] = True  # Mark this example as done

                prover_results.append({
                    'prover_answer': prover_answer,
                    'prover_status': prover_status,
                    'prover_error': prover_error,
                    'invalid_premises': invalid_data.get("invalid_premises", None),
                    'invalid_conclusion': invalid_data.get("invalid_conclusion", None),
                    'generated_premises': generated_premises,
                    'generated_conclusion': generated_conclusion,
                    'response': response
                })

            # If all examples are finished (prover_answer == True), break out of the loop early
            if all(finished):
                print("All examples have been proven. Ending refinement early.")
                break

            # Step 5: Reconcile results for the FOLRefiner as a batch for the remaining examples
            for idx, prover_result in enumerate(prover_results):
                if finished[idx]:  # Skip already proven examples
                    continue

                print(f"Correcting FOL for example {idx + 1}/{len(prover_results)}...")
                corrected_fol = self.refiner.process_single_sample({
                    'generated_fol_premises': prover_result['generated_premises'],
                    'generated_fol_conclusion': prover_result['generated_conclusion'],
                    'nl_context': batch_data.contexts[idx],
                    'nl_question': batch_data.questions[idx],
                    'prover_answer': prover_result['prover_answer'],
                    'prover_status': prover_result['prover_status'],
                    'prover_error': prover_result['prover_error'],
                    'invalid_premises': prover_result['invalid_premises'],
                    'invalid_conclusion': prover_result['invalid_conclusion'],
                    'gold_answer': batch_data.gold_answers[idx]
                })

                # Update the prompt for the next iteration based on the corrections
                refinement_string = f"Results from Prover9 (round {iteration + 1}): {prover_result['prover_answer']}, {prover_result['prover_status']}, {prover_result['prover_error']},\nInvalid sentences {prover_result['invalid_premises']}\nFOLRefiner answer:\nFOL correction: {corrected_fol}\n\n"
                batch_prompt_inputs[idx] = batch_prompt_inputs[idx].split("<|start_header_id|>user<|end_header_id|>")[0] + refinement_string + "<|start_header_id|>user<|end_header_id|>" + batch_prompt_inputs[idx].split("<|start_header_id|>user<|end_header_id|>")[1]

        # Store the final output for each example in the batch
        for idx, prover_result in enumerate(prover_results):
            output_dict = {
                'generated_fol_premises': prover_result['generated_premises'],
                'generated_fol_conclusion': prover_result['generated_conclusion'],
                'gold_answer': batch_data.gold_answers[idx],
                'example_id': batch_data.example_ids[idx],
                'nl_context': batch_data.contexts[idx],
                'nl_question': batch_data.questions[idx],
                'prover_answer': prover_result['prover_answer'],
                'prover_status': prover_result['prover_status'],
                'prover_error': prover_result['prover_error'],
                'refined_response': prover_result['response'],
                'corrected_fol': corrected_fol if not finished[idx] else "Already proved"
            }

            json_ms_marco.append(output_dict)

        return json_ms_marco


    def translate(self):
        dataset = self.load_data()

        json_output = []
        iterations = len(dataset) // self.batch_size

        print(f'Starting translation from NL to FOL for {len(dataset)} examples...')

        for i in tqdm(range(iterations)):
            b_start = i * self.batch_size
            b_end = min((i + 1) * self.batch_size, len(dataset))
            batch = [dataset[j] for j in range(b_start, b_end)]
            batch_data = BatchData(batch)

            print(f"\nProcessing batch {i + 1}/{iterations}...")
            batch_fol_json = self.translate_batch(batch_data)
            json_output.extend(batch_fol_json)

            if i % 5 == 0:
                print("Saving intermediate results...")
                with open(self.output_filename, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, indent=4, ensure_ascii=False)

        print("Saving final results...")
        with open(self.output_filename, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4, ensure_ascii=False)

# Example usage:
if __name__ == '__main__':
    args = parse_arguments()

    engine = TranslatorEngine(
        dataset=args.dataset, 
        output_filename=args.output, 
        batch_size=args.batch_size, 
        sample_limit=args.sample_limit, 
        start_index=args.start_index, 
        end_index=args.end_index
    )
    
    engine.translate()
