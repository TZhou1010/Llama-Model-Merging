# train_llama2.py

import os
import json
import random
import torch
import tqdm
import evaluate
import datetime
from transformers import AutoTokenizer, GenerationConfig
from model_llama2 import AdaMergingLlama2, softmax_entropy

# === Load prepared base and lora ===
checkpoint = torch.load("prepared_llama2.pth")
base_model_weights = checkpoint["base_model"]
task_vectors = checkpoint["task_vectors"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load base model structure ===
from transformers import AutoModelForCausalLM
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
base_model.load_state_dict(base_model_weights)

# === Build paramslist ===
paramslist = []
names = list(base_model_weights.keys())
paramslist.append(tuple(v.detach() for v in base_model_weights.values()))
for lora_state_dict in [task_vectors[i] for i in ["cb", "wnli", "wsc"]]:
    lora_params = {k: v.detach() for k, v in lora_state_dict.items()}
    paramslist.append(lora_params)

# === Setup Model ===
adamerging_model = AdaMergingLlama2(
    model_structure=base_model,
    paramslist=paramslist,
    names=names,
    device=device
).to(device)

optimizer = torch.optim.Adam(adamerging_model.collect_trainable_params(), lr=1e-3)

# === Setup tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# === Data ===
test_data_dir = "../data/data_test"
all_dataset_names = ["cb", "wnli", "wsc"]
all_inputs = []
for lora_name in all_dataset_names:
    filepath = os.path.join(test_data_dir, f"{lora_name}.jsonl")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                all_inputs.append(json.loads(line)["inputs"])

# === Setup Evaluation Config ===
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
gen_config = GenerationConfig(
    max_new_tokens=128,
    num_beams=1,
    do_sample=False,
    temperature=1.0,
    top_p=1.0,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# === Setup Logging ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_folder = f"../logs_llama/{timestamp}"
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, "training_log.txt")


def evaluate_dataset(dataset_name):
    dataset_path = os.path.join(test_data_dir, f"{dataset_name}.jsonl")
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_name} not found.")
        return None

    inputs = []
    targets = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            inputs.append(data['inputs'])
            targets.append(data['targets'])

    predictions = []
    adamerging_model.load_weights()
    adamerging_model.eval()

    for input_text in tqdm.tqdm(inputs, desc=f"Generating for {dataset_name}"):
        encoded = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            generated_ids = adamerging_model.model_structure.generate(**encoded, generation_config=gen_config)

        output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        predictions.append(output_text)

    adamerging_model.train()

    if dataset_name.lower() in {"common_gen", "dart", "e2e_nlg", "web_nlg_en"}:
        results = rouge.compute(predictions=predictions, references=targets, use_stemmer=True)
        return {"rouge1": results["rouge1"], "rouge2": results["rouge2"], "rougeL": results["rougeL"]}
    else:
        correct = sum([pred.strip() == tgt.strip() for pred, tgt in zip(predictions, targets)])
        return {"exact_match": correct / len(targets)}


# === Training Loop ===
epochs = 100
batch_size = 1

for epoch in range(epochs):
    random.shuffle(all_inputs)
    adamerging_model.load_weights()

    optimizer.zero_grad()

    total_loss = 0.0
    num_batches = 0

    for i in tqdm.tqdm(range(0, len(all_inputs), batch_size), desc=f"Epoch {epoch+1}"):
        batch_texts = all_inputs[i:i+batch_size]
        encoding = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        outputs = adamerging_model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )

        logits = outputs.logits
        loss = softmax_entropy(logits).mean()

        loss.backward()
        total_loss += loss.item()
        num_batches += 1

    optimizer.step()
    optimizer.zero_grad()

    avg_loss = total_loss / num_batches
    print(f"✅ Finished Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}\n")

    torch.cuda.empty_cache()

    # Evaluate every 100 epochs or at last
    if (epoch + 1) % 100 == 0 or (epoch + 1) == epochs:
        print(f"=== Evaluation after epoch {epoch+1} ===")
        with open(log_file, "a") as f:
            f.write(f"\n=== Evaluation after epoch {epoch+1} ===\n")

            for dataset in all_dataset_names:
                metrics = evaluate_dataset(dataset)
                if metrics:
                    metric_parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
                    metric_summary = ", ".join(metric_parts)
                    result_line = f"Dataset: {dataset} | {metric_summary}\n"
                    print(result_line)
                    f.write(result_line)
