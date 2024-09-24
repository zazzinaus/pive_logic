import os
import shutil
import json
from tqdm import tqdm
import torch
from unsloth import FastLanguageModel

class BatchData:
    """Helper class to store and manage batch data."""
    def __init__(self, batch):
        self.prompts = [item['text'] for item in batch]
        self.gold_answers = [item['answers'] for item in batch]
        self.example_ids = [item['query_id'] for item in batch]

class TranslatorEngine:
    def __init__(self, dataset, output_filename, batch_size, sample_limit=None):
        self.dataset = dataset  # Use the dataset that includes 'text' field
        self.output_filename = output_filename
        self.batch_size = batch_size
        self.sample_limit = sample_limit
        
        # Model configuration
        self.max_seq_length = 2048
        self.dtype = None
        self.load_in_4bit = True
        
        # Load model and tokenizer
        print("Loading the model and tokenizer...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Meta-Llama-3.1-8B",
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )
        
        # Apply PEFT (Parameter-Efficient Fine-Tuning)
        print("Applying PEFT...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Enable faster inference
        print("Enabling faster inference mode...")
        FastLanguageModel.for_inference(self.model)

        # Check GPU status
        print("Checking GPU availability and clearing cache...")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()  # Clear the GPU cache
        else:
            print("No GPU found. Running on CPU.")
        
    def load_data(self):
        """Loads the dataset into memory."""
        print(f"Loading dataset: {self.dataset}...")

        # Load the formatted dataset that contains the 'text' field for inference
        with open(self.dataset, "r") as f:
            dataset = json.load(f)
        
        # Apply sample limit if needed
        if self.sample_limit > 0:
            dataset = dataset[:self.sample_limit]
        
        print(f"Loaded {len(dataset)} examples from {self.dataset}.")
        return dataset

    def translate_batch(self, batch_data):
        """Processes a batch of prompts and retrieves answers."""
        batch_prompt_inputs = batch_data.prompts

        batch_answers = []
        for prompt in batch_prompt_inputs:
            print("\nGenerated Prompt:\n", prompt)
            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            outputs = self.model.generate(**inputs, max_new_tokens=256, use_cache=True)
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            response = generated_texts.strip()
            batch_answers.append(response)
            
            # Print the machine's output (response)
            print("\nMachine Output:\n", response)

        return batch_answers
    
    def translate(self):
        """Main translation method."""
        dataset = self.load_data()

        json_output = []
        iterations = len(dataset) // self.batch_size

        print(f'Starting inference for {len(dataset)} examples...')
        for i in tqdm(range(iterations + 1)):
            b_start = i * self.batch_size
            b_end = min((i + 1) * self.batch_size, len(dataset))
            
            # Create a batch from the dataset
            batch = [dataset[j] for j in range(b_start, b_end)]
            batch_data = BatchData(batch)

            # Generate answers for the batch
            batch_answers = self.translate_batch(batch_data)

            # Save the batch results
            for j in range(len(batch)):
                prompt = batch_data.prompts[j]
                gold_answer = batch_data.gold_answers[j]
                example_id = batch_data.example_ids[j]
                generated_answer = batch_answers[j]

                # Print the generated answer
                print(f"Prompt: {prompt}")
                print(f"Generated Answer: {generated_answer}")
                print(f"Gold Answer: {gold_answer}")

                # Save the output in a dictionary
                output_dict = {
                    'prompt': prompt,
                    'generated_answer': generated_answer,
                    'gold_answer': gold_answer,
                    'example_id': example_id
                }
                json_output.append(output_dict)

            # Periodic save (save every 5 batches)
            if i % 5 == 0:
                print(f"Saving progress at batch {i}...")
                with open(self.output_filename, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, indent=4, ensure_ascii=False)

        # Final save
        print("Saving final output...")
        with open(self.output_filename, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4, ensure_ascii=False)

        self.cleanup()

    def cleanup(self):
        """Clean up temporary files."""
        print("Cleaning up temporary files...")
        compiled_krb_dir = './compiled_krb'
        if os.path.exists(compiled_krb_dir):
            print('Removing compiled_krb directory...')
            shutil.rmtree(compiled_krb_dir)

# Example usage:
if __name__ == '__main__':
    dataset = 'format_inference_dataset.json'  # Ensure the 'text' field is present in this dataset
    output = 'fol_qa_output.json'
    batch_size = 1
    sample_limit = 10  # Set sample limit if needed

    engine = TranslatorEngine(dataset, output, batch_size, sample_limit)
    engine.translate()
