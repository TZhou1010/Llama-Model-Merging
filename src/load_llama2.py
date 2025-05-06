# load_llama2.py

import os
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, login


# Config
base_model_name = "meta-llama/Llama-2-7b-hf"
lora_list = [
    'natural_questions', 'arc_challenge', 'arc_easy', 'story_cloze', 'piqa', 'copa',
    'hellaswag', 'definite_pronoun_resolution', 'wsc', 'cb', 'wnli', 'mnli_matched',
    'anli_r1', 'anli_r2', 'anli_r3', 'mnli_mismatched', 'snli', 'qnli', 'rte',
    'paws_wiki', 'glue_qqp', '', 'cosmos_qa', 'record', 'multirc', 'squad_v1',
    'squad_v2', 'openbookqa', 'bool_q', 'drop', 'sst2', 'yelp_polarity_reviews',
    'imdb_reviews', 'sentiment140', 'web_nlg_en', 'dart', 'e2e_nlg', 'common_gen',
    'wmt16_translate_deen', 'wmt16_translate_fien', 'wmt16_translate_roen',
    'wmt16_translate_ruen', 'wmt16_translate_tren', 'wmt16_translate_csen',
    'wmt14_enfr'
]
lora_repo_prefix = "Styxxxx/llama2_7b_lora-"
device = "cuda" if torch.cuda.is_available() else "cpu"
offload_folder = "./offload"

# Load Base Model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder=offload_folder,
    trust_remote_code=True
)

# Load LoRA deltas
print("Loading LoRA adapters...")
task_vectors = {}
for lora_name in lora_list:
    try:
        lora_file = hf_hub_download(
            repo_id=f"{lora_repo_prefix}{lora_name}",
            filename="adapter_model.safetensors",
            repo_type="model"
        )
        lora_state = load_file(lora_file)
        task_vectors[lora_name] = lora_state
        print(f"✅ Loaded {lora_name}")
    except Exception as e:
        print(f"❌ Failed {lora_name}: {e}")

print(f"Loaded {len(task_vectors)} LoRA adapters.")

# Save everything to use in training
torch.save({
    "base_model": base_model.state_dict(),
    "task_vectors": task_vectors
}, "prepared_llama2.pth")
