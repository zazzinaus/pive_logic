import os
import shutil
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
import argparse
from vllm import LLM, SamplingParams
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
from vllm_correction import FOLRefiner
from transformers import AutoTokenizer



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

# QLoRA configuration remains the same
lora_config = LoraConfig(
    r=16, 
    lora_alpha=16, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

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
    print(f"\nExecuting FOL prover with premises:\n{generated_premises}\nand conclusion:\n{generated_conclusion}\n")
    
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



def print_gpu_memory_usage():
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_memory_max = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"GPU memory used: {gpu_memory_used:.2f} GB")
        print(f"GPU max memory allocated: {gpu_memory_max:.2f} GB")
    else:
        print("CUDA is not available, running on CPU.")

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

        # vLLM Engine Arguments
        #self.engine_args = EngineArgs(tensor_parallel_size=2, dtype="float16")
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256, n=1)
        
        # Print GPU memory after loading the model
        print("GPU memory usage before loading model:")
        print_gpu_memory_usage()

        self.tokenizer = AutoTokenizer.from_pretrained("hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
        # Load model using vLLM
        print("Loading model with vLLM...")
        self.model = LLM("hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                        max_model_len=2048, 
                        dtype="auto", 
                        quantization='awq'
                        )
        print("Model loaded with vLLM.")

        # Print GPU memory after loading the model
        print("GPU memory usage after loading model:")
        print_gpu_memory_usage()

        # FOL Refiner
        print("Instantiating FOLRefiner...")
        self.refiner = FOLRefiner(
            model_name="LorMolf/LogicLlama2-chat-direct",  # FOL correction model
            template_filename='good_folio'
        )
        print("FOLRefiner initialized.")

        # Check GPU status
        print("Checking GPU availability and clearing cache...")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()  # Clear the GPU cache
        else:
            print("No GPU found. Running on CPU.")

    def load_data(self):
        print(f"Loading dataset: {self.dataset}...")

        # Load the specified JSON dataset
        with open(self.dataset, "r") as f:
            dataset = json.load(f)

        # Apply sample limit if needed
        if self.sample_limit > 0:
            dataset = dataset[:self.sample_limit]

        # Apply the chunking based on start_index and end_index
        if self.end_index == -1:
            self.end_index = len(dataset)

        dataset_chunk = dataset[self.start_index:self.end_index]

        print(f"Loaded {len(dataset_chunk)} examples from chunk {self.start_index} to {self.end_index} from {self.dataset}.")
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

        batch_prompt_inputs = [
            self.format_prompt(premises, conclusion)
            for premises, conclusion in zip(batch_data.contexts, batch_data.questions)
        ]

        json_ms_marco = []

        for idx, prompt in enumerate(batch_prompt_inputs):
            print(f"\nTranslating NL to FOL for example {idx + 1}/{len(batch_data.example_ids)}...")
            
            # Print GPU memory usage before inference
            print("GPU memory usage before inference:")
            print_gpu_memory_usage()

            corrected_fol = None
            refined_response = None
            base_prompt = prompt
            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

            prompt_before_user = base_prompt.split("<|start_header_id|>user<|end_header_id|>")[0]
            prompt_after_user = "<|start_header_id|>user<|end_header_id|>" + base_prompt.split("<|start_header_id|>user<|end_header_id|>")[1]

            for i in range(3):
                #print(f"\nRefinement iteration {i + 1} for example {idx + 1}...")
                outputs = self.model.generate(**inputs, max_new_tokens=256, use_cache=True)
                generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                response = self.extract_response(generated_texts)
                #print(f"Generated FOL (iteration {i + 1}): {response}")

                if "FOL premises:" in response and "FOL question:" in response:
                    parts = response.split("FOL question:")
                    generated_premises = parts[0].split("FOL premises:")[1].strip()
                    generated_conclusion = parts[1].strip()
                else:
                    generated_premises = ""
                    generated_conclusion = ""
                
                print("Running inference on generated FOL...")
                prover_answer, prover_status, prover_error = None, None, None

                print(f"Continuous mode is: {self.continuous}")
                if self.continuous:
                    prover_answer, prover_status, prover_error, invalid_data = inference_on_generated_fol(
                        generated_premises, generated_conclusion)
                    
                    #print(f"Inference result: {prover_answer}, Errors: {prover_status} {prover_error} ")
                    invalid_data = invalid_data if isinstance(invalid_data, dict) else {}

                    print("Correcting FOL...")
                    corrected_fol = self.refiner.process_single_sample({
                        'generated_fol_premises': generated_premises,
                        'generated_fol_conclusion': generated_conclusion,
                        'nl_context': batch_data.contexts[idx],
                        'nl_question': batch_data.questions[idx],
                        'prover_answer': prover_answer,
                        'prover_status': prover_status,
                        'prover_error': prover_error,
                        'invalid_premises': invalid_data.get("invalid_premises", None),  # Include invalid premises
                        'invalid_conclusion': invalid_data.get("invalid_conclusion", None),  # Include invalid conclusion
                        #'fol_context': batch_data.fol_contexts[idx],
                        #'gold_conclusion': batch_data.fol_gold_questions[idx],
                        'gold_answer': batch_data.gold_answers[idx]
                    })
                    #print(f"Corrected FOL: {corrected_fol}")
                    
                    # Check if corrected FOL starts with "N/A" and break if it does
                    if prover_answer == "True":
                        print("FOL already proved by Prover9, breaking the refinement loop.")
                        break

                    refinement_string = f"Results from Prover9 (round {i + 1}): {prover_answer}, {prover_status}, {prover_error},\nInvalid sentences {invalid_data}\nFOLRefiner answer:\nFOL correction: {corrected_fol}\n\n"
                    #print(f"Refinement string (round {i + 1}):\n{refinement_string}")
                    prompt = prompt_before_user + refinement_string + prompt_after_user

                inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

            refined_response = response
            #print(f"Final refined response for example {idx + 1}: {refined_response}")

            # Print GPU memory usage after inference
            print("GPU memory usage after inference:")
            print_gpu_memory_usage()

            # Store the final output
            output_dict = {
                'generated_fol_premises': generated_premises,
                #'fol_context': batch_data.fol_contexts[idx],
                'generated_fol_conclusion': generated_conclusion,
                #'gold_conclusion': batch_data.fol_gold_questions[idx],
                'gold_answer': batch_data.gold_answers[idx],
                'example_id': batch_data.example_ids[idx],
                'nl_context': batch_data.contexts[idx],
                'nl_question': batch_data.questions[idx],
                #'context_id': batch_data.context_ids[idx],
                'prover_answer': prover_answer,
                'prover_status': prover_status,
                'prover_error': prover_error,
                #'is_correct': is_correct,
                'refined_response': refined_response,
                'corrected_fol': corrected_fol
            }

            json_ms_marco.append(output_dict)

        return json_ms_marco



    def translate(self):
        """Main translation method."""
        dataset = self.load_data()

        json_output = []
        iterations = len(dataset) // self.batch_size

        print(f'Starting translation from NL to FOL for {len(dataset)} examples...')

        for i in tqdm(range(iterations)):
            b_start = i * self.batch_size
            b_end = min((i + 1) * self.batch_size, len(dataset))
            batch = [dataset[j] for j in range(b_start, b_end)]
            batch_data = BatchData(batch)

            print(f"\nProcessing batch sample {i}/{iterations}...")
            batch_fol_json = self.translate_batch(batch_data)
            json_output.extend(batch_fol_json)

            if i % 5 == 0:
                print("Saving intermediate results...")
                with open(self.output_filename, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, indent=4, ensure_ascii=False)

        print("Saving final results...")
        with open(self.output_filename, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4, ensure_ascii=False)

        self.cleanup()

    def cleanup(self):
        """Clean up temporary files."""
        compiled_krb_dir = './compiled_krb'
        if os.path.exists(compiled_krb_dir):
            print('Removing compiled_krb directory...')
            shutil.rmtree(compiled_krb_dir)


# Example usage:
if __name__ == '__main__':
    # dataset = 'ms_marco_10k.json'
    # output = 'corrected_ms_marco_10k.json'
    # batch_size = 1
    # sample_limit = 2  # Process all samples, or set this to limit it to a number of samples
    # continuous = True
    # engine = TranslatorEngine(dataset, output, batch_size, sample_limit)
    # engine.translate()

    # Parse command-line arguments
    args = parse_arguments()

    # Create an instance of TranslatorEngine with arguments from the command line
    engine = TranslatorEngine(
        dataset=args.dataset, 
        output_filename=args.output, 
        batch_size=args.batch_size, 
        sample_limit=args.sample_limit, 
        start_index=args.start_index, 
        end_index=args.end_index, 
        continuous=args.continuous
    )
    
    # Run the translation process
    engine.translate()