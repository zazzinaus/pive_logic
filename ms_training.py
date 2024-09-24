from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported
import torch
import json
import pandas as pd
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from datasets import Dataset
import os
import evaluate
from evaluate import load


exact_match_metric = load("exact_match")
rouge = load('rouge')
# predictions = ["hello there", "general kenobi"]
# references = ["hello there", "general kenobi"]
# results = rouge.compute(predictions=predictions, references=references)
#print(results)
os.environ["WANDB_DISABLED"] = "true"

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None
load_in_4bit = True


corrected_dataset = []
with open("chunk1_0_300_corrected_ms.json", 'r') as f:
    corrected_dataset = json.load(f)

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


def formatting_prompts_func(example):
    formatted_sample = []
    for i in range(len(example['nl_question'])):
            prompt_text = f"""<|begin_of_text|>{{
Use the natural language and the first-order logic translation of contexts and question to answer the queries.
Answer only in Natural Language with the response and nothing else.
You are a helpful assistant.
### Contexts: 
{example['nl_context'][i]}

### FOL contexts translation: 
{example['generated_fol_premises'][i]}

### Question: 
{example['nl_question'][i]}

### FOL question traslation: 
{example['generated_fol_conclusion']}

### Answer:
{example['gold_answer']}
}}"""
            # Print each prompt to see the formatted output
            #print(f"Formatted prompt {i+1}:")
            #print(prompt_text)
            formatted_sample.append(prompt_text)
    return formatted_sample

response_template = " ### Answer: "
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


#  Convert the list of dictionaries to a Huggingface Dataset
dataset = Dataset.from_pandas(pd.DataFrame(corrected_dataset))


# print(dataset[0])


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


trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 2, #  it will override any value given in num_train_epochs
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
    ),
)


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


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

# # Go to https://github.com/unslothai/unsloth/wiki for advanced tips like
# # (1) Saving to GGUF / merging to 16bit for vLLM
# # (2) Continued training from a saved LoRA adapter
# # (3) Adding an evaluation loop / OOMs
# # (4) Customized chat templates




# formatted_dataset = []
# for data in corrected_dataset:
#     prompt_text = f"""<|begin_of_text|>{{
# Use the natuaral language and the first-order logic translation of contexts and question to answer the queries.
# Answer only in Natural Language with the response and nothing else.
# You are a helpful assistant.
# ### Contexts: 
# {data['nl_context']}

# ### FOL contexts translation: 
# {data['generated_fol_premises']}

# ### Question: 
# {data['nl_question']}

# ### FOL question traslation: 
# {data['generated_fol_conclusion']}

# ### Answer:
# }}
#     """


#     data['text'] = prompt_text  # Add the prompt to a new 'text' field
#     formatted_dataset.append(data)

# output_json_path = "formatted_dataset.json"
# with open(output_json_path, 'w', encoding='utf-8') as f:
#     json.dump(formatted_dataset, f, indent=4, ensure_ascii=False)

#print(f"Formatted dataset saved to {output_json_path}")
#print(formatted_dataset[0]['text'])


# trainer = SFTTrainer(
#     model = model,
#     train_dataset = dataset,
#     dataset_text_field = "text",
#     max_seq_length = max_seq_length,
#     tokenizer = tokenizer,
#     args = TrainingArguments(
#         per_device_train_batch_size = 2,
#         gradient_accumulation_steps = 4,
#         warmup_steps = 10,
#         max_steps = 2, #  it will override any value given in num_train_epochs
#         fp16 = not is_bfloat16_supported(),
#         bf16 = is_bfloat16_supported(),
#         logging_steps = 1,
#         output_dir = "tmp",
#         optim = "adamw_8bit",
#         seed = 3407,
#     ),
# )