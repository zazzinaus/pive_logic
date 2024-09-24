from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported
import torch
import json
import pandas as pd
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from datasets import Dataset
import os
os.environ["WANDB_DISABLED"] = "true"

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None
load_in_4bit = True


corrected_dataset = []
with open("chunk1.json", 'r') as f:
    corrected_dataset = json.load(f)

dataset = Dataset.from_dict({"answers": [item["answers"] for item in corrected_dataset],
                             "selected_passages": [item["selected_passages"] for item in corrected_dataset],
                             "query": [item["query"] for item in corrected_dataset],
                             "query_id": [item["query_id"] for item in corrected_dataset],
                             "query_type": [item["query_type"] for item in corrected_dataset]})

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


def formatting_prompts_func(example):
    # Process each example as a whole, assuming that 'query', 'selected_passages', and 'answers' are strings
    text = f"### Query: {example['query']}\n" \
           f"### Selected Passages: {example['selected_passages']}\n" \
           f"### Answer:{example['answers']}"
    
    # Return the formatted text as a list (because the output should be a list of strings)
    return [text]

response_template = "### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

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
    #args=SFTConfig(output_dir="/outputs"),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 5, #  it will override any value given in num_train_epochs
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
    ),
)

trainer.train()
# def formatting_prompts_func(example):
#     prompt_text = f"""Answer the questions using the contexts provided \n### Contexts: 
# {example['selected_passages']}

# ### Question: 
# {example['query']}

# ### Answer:
# {example['answers']}

#     """
#         #print("sample: ", {prompt_text})
#     #     formatted_sample.append(prompt_text)
#     # return formatted_sample
#     example['text'] = prompt_text
#     return example

# response_template = "### Answer: "
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


# #     data['text'] = prompt_text  # Add the prompt to a new 'text' field
# #     formatted_dataset.append(data)

# # output_json_path = "formatted_dataset.json"
# # with open(output_json_path, 'w', encoding='utf-8') as f:
# #     json.dump(formatted_dataset, f, indent=4, ensure_ascii=False)

# # print(f"Formatted dataset saved to {output_json_path}")
# # #print(formatted_dataset[0]['text'])



# # Convert the list of dictionaries to a Huggingface Dataset
# dataset = Dataset.from_pandas(pd.DataFrame(corrected_dataset))
# #dataset = dataset.map(formatting_prompts_func)
# #print(dataset[0])


# # Do model patching and add fast LoRA weights
# model = FastLanguageModel.get_peft_model(
#     model,
#     r = 16,
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj"],
#     lora_alpha = 16,
#     lora_dropout = 0, # Supports any, but = 0 is optimized
#     bias = "none",    # Supports any, but = "none" is optimized
#     # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
#     use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
#     random_state = 3407,
#     max_seq_length = max_seq_length,
#     use_rslora = False,  # We support rank stabilized LoRA
#     loftq_config = None, # And LoftQ
# )


# trainer = SFTTrainer(
#     model = model,
#     train_dataset = corrected_dataset,
#     #dataset_text_field = "text",
#     formatting_func=formatting_prompts_func,
#     data_collator=collator,
#     max_seq_length = max_seq_length,
#     tokenizer = tokenizer,
#     args = TrainingArguments(
#         per_device_train_batch_size = 2,
#         gradient_accumulation_steps = 4,
#         warmup_steps = 10,
#         max_steps = 5, #  it will override any value given in num_train_epochs
#         fp16 = not is_bfloat16_supported(),
#         bf16 = is_bfloat16_supported(),
#         logging_steps = 1,
#         output_dir = "outputs",
#         optim = "adamw_8bit",
#         seed = 3407,
#     ),
# )


# gpu_stats = torch.cuda.get_device_properties(0)
# start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
# print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
# print(f"{start_gpu_memory} GB of memory reserved.")


# trainer_stats = trainer.train()


# used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
# used_percentage = round(used_memory         /max_memory*100, 3)
# lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
# print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
# print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
# print(f"Peak reserved memory = {used_memory} GB.")
# print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
# print(f"Peak reserved memory % of max memory = {used_percentage} %.")
# print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# # Go to https://github.com/unslothai/unsloth/wiki for advanced tips like
# # (1) Saving to GGUF / merging to 16bit for vLLM
# # (2) Continued training from a saved LoRA adapter
# # (3) Adding an evaluation loop / OOMs
# # (4) Customized chat templates

