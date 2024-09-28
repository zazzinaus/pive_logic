
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
import json
from datasets import Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import os
os.environ["WANDB_DISABLED"] = "true"

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None
load_in_4bit = True

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

corrected_dataset = []
with open("chunk2.json", 'r') as f:
    corrected_dataset = json.load(f)

# Create a prompt string by iterating through the corrected dataset
# Assume each data point in corrected_dataset has keys 'question' and 'answer'
formatted_dataset = []
for data in corrected_dataset:
    prompt_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are given a question and a selected passage that provides context. Provide a clear and concise answer to the question using only the information from the passage.<|eot_id|><|start_header_id|>user<|end_header_id|>
### Passage:
{data['selected_passages']}

### Question:
{data['query']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Answer: {data['answers']}<|end_of_text|>
"""
    data['text'] = prompt_text  # Add the prompt to a new 'text' field
    formatted_dataset.append(data)

output_json_path = "formatted_dataset.json"
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(formatted_dataset, f, indent=4, ensure_ascii=False)

print(f"Formatted dataset saved to {output_json_path}")
#print(formatted_dataset[0]['text'])



# Convert the list of dictionaries to a Huggingface Dataset
dataset = Dataset.from_pandas(pd.DataFrame(formatted_dataset))


#print(dataset[0])


prompt = """Provide a clear and concise answer to the question using only the information from the passage.
### Passage:
{}
### Question:
{}

### Answer: {}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    query = examples["query"]
    context       = examples["selected_passages"]
    outputs      = examples["answers"]
    texts = []
    for q, c, a in zip(query, context, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = prompt.format(q, c, a) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

corrected_dataset = []
with open("chunk2.json", 'r') as f:
    corrected_dataset = json.load(f)

corrected_dataset = Dataset.from_list(corrected_dataset)

dataset = corrected_dataset.map(formatting_prompts_func, batched = True,)

# response_template = "### Answer: "
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    #data_collator = collator,
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 5,
        #max_steps = 5, #  it will override any value given in num_train_epochs
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)



#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Go to https://github.com/unslothai/unsloth/wiki for advanced tips like
# (1) Saving to GGUF / merging to 16bit for vLLM
# (2) Continued training from a saved LoRA adapter
# (3) Adding an evaluation loop / OOMs
# (4) Customized chat templates

model.save_pretrained("base_lora_model") # Local saving
tokenizer.save_pretrained("base_lora_model")






