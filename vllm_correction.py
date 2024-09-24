from vllm import LLM, SamplingParams
from utils_back import query_model, get_template
import torch

class FOLRefiner:
    def __init__(self, model_name, template_filename, max_seq_length=2048):
        self.template_filename = template_filename
        
        # Use vLLM to load the model
        print("Loading model with vLLM for FOLRefiner...")
        self.refiner_model = LLM(model_name, max_model_len=max_seq_length)
        print("Model loaded with vLLM for FOLRefiner.")

    def preprocess_sample(self, example):
        prover_status = example.get('prover_status', '')  # Default to an empty string if not found
        prover_error = example.get('prover_error', '')  # Default to an empty string if not found
        #is_correct = example.get('is_correct', None)  # Default to None if not found

        #gold_fol = f"FOL premises: \n{example['fol_context']}\n FOL conclusion: \n{example['gold_conclusion']}"
        gen_fol = f"### FOL premises: \n{example['generated_fol_premises']}\n### FOL question: \n{example['generated_fol_conclusion']}"

        template = get_template(
            task='train', 
            template_filename=self.template_filename, 
            formatting_kwargs={
                'premises': example['nl_context'], 
                'question': example['nl_question'], 
                'generated_fol': gen_fol, 
                'prover_answer': example['prover_answer'],
                'parsing_error': example['prover_status'], 
                'exec_error': example['prover_error'],
                'invalid_premises': example['invalid_premises'],
                'invalid_conclusion': example['invalid_conclusion']
                #'is_correct': is_correct
            }
        )
        return template

    def correct_fol(self, input_text):
        # Check the input text before passing it to the model
        #print(f"Input to query_model: {input_text}")

        # Set vLLM sampling parameters
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256, n=1)

        # Use vLLM to generate the corrected FOL
        outputs = self.refiner_model.generate(input_text, sampling_params)
        response = outputs[0].outputs[0].text
        
        # Handle cases where the generated text might not contain the expected "### FOL Correction:" marker
        if "### FOL Correction:" in response:
            corrected_fol = response.split("### FOL Correction:\n")[2].strip()
        else:
            corrected_fol = "### FOL Correction: Correction Error"  # Fallback if no correction is found

        return corrected_fol
    
    def process_single_sample(self, sample):
        # Preprocess the data
        input_text = self.preprocess_sample(sample)
        
        # Correct the FOL using the refiner model
        corrected_fol = self.correct_fol(input_text)
        
        # Output the corrected FOL
        return corrected_fol


