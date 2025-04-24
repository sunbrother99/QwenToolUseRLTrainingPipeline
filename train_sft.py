from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

model_name = "Qwen/Qwen-7B-Chat"
dataset = load_dataset("json", data_files="data/sft_data.jsonl")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    args=TrainingArguments(
        output_dir="output/sft",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=5e-5,
    )
)
trainer.train()
