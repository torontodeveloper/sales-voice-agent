"""
DPO Training Script
- 50 preference pairs generated in data/processed/dpo_generated_pair.jsonl
- DPOTrainer configured with beta=0.1, lr=5e-6, 1 epoch
- Blocked by trl/transformers/unsloth version conflict on Colab
- Run on dedicated environment: pip install unsloth trl==0.11.4 transformers==4.44.2
"""

from unsloth import PatchDPOTrainer
from datasets import load_dataset

PatchDPOTrainer()

from transformers import TrainingArguments
from unsloth import PatchDPOTrainer, FastLanguageModel

PatchDPOTrainer()
from trl import DPOTrainer

training_args = TrainingArguments(
    per_device_train_batch_size=1,  # safer
    gradient_accumulation_steps=8,  # keeps effective batch
    num_train_epochs=1,  # start small
    learning_rate=5e-6,
    logging_steps=10,
    output_dir="outputs",
    report_to="none",
    fp16=True,
)


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/drive/MyDrive/sales-voice-agent/llama_lora",
    max_seq_length=2048,
    load_in_4bit=True,
)
dpo_generated_dataset = load_dataset(
    "json", data_files="./data/processed/dpo_generated_pair.jsonl", split="train"
)
dpo_generated_dataset = dpo_generated_dataset.remove_columns(["id"])
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dpo_generated_dataset,  # make sure this is your cleaned dataset
    tokenizer=tokenizer,
    beta=0.1,
    max_length=1024,
    max_prompt_length=512,
)
