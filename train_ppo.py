# train_ppo.py
from trl import PPOTrainer, PPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ppo_config = PPOConfig(
    model_name="Qwen/Qwen-7B-Chat",
    reward_tokenizer="output/rm",
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=AutoModelForCausalLM.from_pretrained(ppo_config.model_name, trust_remote_code=True),
    tokenizer=AutoTokenizer.from_pretrained(ppo_config.model_name, trust_remote_code=True),
    dataset=load_dataset("json", data_files="data/ppo_data.jsonl")["train"]
)

ppo_trainer.train()
