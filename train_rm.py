from trl import RewardTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("json", data_files="data/rm_data.jsonl")

trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    args=TrainingArguments(output_dir="output/rm", num_train_epochs=3)
)
trainer.train()
