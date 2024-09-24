from vllm import LLM
import torch
from transformers import AutoTokenizer


# pip install vllm

def print_gpu_memory_usage(stage):
    """Helper function to print GPU memory usage."""
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB
    print(f"{stage} - GPU Memory Allocated: {allocated_memory:.2f} GB")
    print(f"{stage} - GPU Memory Reserved: {reserved_memory:.2f} GB")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("casperhansen/llama-3-8b-instruct-awq")  #("meta-llama/Meta-Llama-3.1-8B-Instruct")

print_gpu_memory_usage("Before model loading")
model = LLM(model="casperhansen/llama-3-8b-instruct-awq",\
             max_model_len=1024, quantization='awq', gpu_memory_utilization=0.9) #("meta-llama/Meta-Llama-3.1-8B-Instruct")

print_gpu_memory_usage("After model loading")

# Prepare the input message
messages = [{"role": "user", "content": "What is the capital of France?"}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print_gpu_memory_usage("Before generation")
# Generate the output
output = model.generate(formatted_prompt)
print(output)

print_gpu_memory_usage("After generation")

# Clear the CUDA cache
torch.cuda.empty_cache()
print_gpu_memory_usage("After clearing cache")
