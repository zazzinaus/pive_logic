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
    def __init__(self, dataset, output_filename, batch_size, sample_limit=None, continuous=True):
        self.dataset = dataset
        self.output_filename = output_filename
        self.batch_size = batch_size
        self.sample_limit = sample_limit
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
        with open(self.dataset, "r") as f:
            dataset = json.load(f)
        if self.sample_limit > 0:
            dataset = dataset[:self.sample_limit]
        print(f"Loaded {len(dataset)} examples from {self.dataset}.")
        return dataset
        
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

            inputs = prompt  # The raw prompt string is now used for vLLM

            for i in range(3):
                #print(f"\nRefinement iteration {i + 1} for example {idx + 1}...")

                # Use vLLM to generate the response
                outputs = self.model.generate([inputs], sampling_params=self.sampling_params)
                generated_texts = outputs[0].outputs[0].text
                response = self.extract_response(generated_texts)

                # Process the response (unchanged from original code)
                generated_premises, generated_conclusion = self.parse_generated_fol(response)
                prover_answer, prover_status, prover_error, invalid_data = self.run_fol_inference(generated_premises, generated_conclusion)

                if prover_answer == "True":
                    break

                # Correct FOL with FOLRefiner
                corrected_fol = self.refiner.process_single_sample({
                    'generated_fol_premises': generated_premises,
                    'generated_fol_conclusion': generated_conclusion,
                    'nl_context': batch_data.contexts[idx],
                    'nl_question': batch_data.questions[idx],
                    'prover_answer': prover_answer,
                    'prover_status': prover_status,
                    'prover_error': prover_error,
                    'invalid_premises': invalid_data.get("invalid_premises", None),
                    'invalid_conclusion': invalid_data.get("invalid_conclusion", None),
                    'gold_answer': batch_data.gold_answers[idx]
                })
                inputs = prompt + f"\nCorrected FOL:\n{corrected_fol}"

            # Store the results (unchanged)
            json_ms_marco.append(self.format_output(generated_premises, generated_conclusion, prover_answer, corrected_fol))

        return json_ms_marco

    def run_fol_inference(self, premises, conclusion):
        prover_answer, prover_status, prover_error, invalid_data = inference_on_generated_fol(premises, conclusion)
        return prover_answer, prover_status, prover_error, invalid_data

    def format_output(self, generated_premises, generated_conclusion, prover_answer, corrected_fol):
        # Output formatting remains the same
        return {
            'generated_fol_premises': generated_premises,
            'generated_fol_conclusion': generated_conclusion,
            'prover_answer': prover_answer,
            'corrected_fol': corrected_fol
        }

    def translate(self):
        dataset = self.load_data()
        json_output = []
        iterations = len(dataset) // self.batch_size
        for i in tqdm(range(iterations)):
            b_start = i * self.batch_size
            b_end = min((i + 1) * self.batch_size, len(dataset))
            batch = [dataset[j] for j in range(b_start, b_end)]
            batch_data = BatchData(batch)
            batch_fol_json = self.translate_batch(batch_data)
            json_output.extend(batch_fol_json)
            if i % 5 == 0:
                with open(self.output_filename, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, indent=4, ensure_ascii=False)
        with open(self.output_filename, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4, ensure_ascii=False)
        self.cleanup()

    def cleanup(self):
        compiled_krb_dir = './compiled_krb'
        if os.path.exists(compiled_krb_dir):
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
    args = parse_arguments()

    # Create an instance of TranslatorEngine with arguments from the command line
    engine = TranslatorEngine(
        dataset=args.dataset, 
        output_filename=args.output, 
        batch_size=args.batch_size, 
        sample_limit=args.sample_limit, 
        continuous=args.continuous
    )
    
    # Run the translation process
    engine.translate()