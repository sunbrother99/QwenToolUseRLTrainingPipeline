# QwenToolUseRLTrainingPipeline
我们来构建一个企业内部 **模糊提问 → 多工具调用 → RL 微调** 的完整链条，包括：

- ✅ 数据格式设计（HuggingFace 风格）
- ✅ SFT 训练脚本（`train_sft.py`）
- ✅ Reward Model 训练脚本（`train_rm.py`）
- ✅ PPO 强化学习脚本（`train_ppo.py`）
- ✅ 评估指标：Tool-level + Chain-level 准确率
- ✅ 示例配置文件（`config.yaml`）

---

## 📁 项目目录结构建议

```
tool_use_rl_project/
├── data/
│   ├── sft_data.jsonl
│   ├── rm_data.jsonl
│   ├── ppo_data.jsonl
├── configs/
│   ├── config_sft.yaml
│   ├── config_rm.yaml
│   ├── config_ppo.yaml
├── train_sft.py
├── train_rm.py
├── train_ppo.py
├── evaluate.py
└── utils/
    ├── tool_schema.py  # 工具定义 + schema 验证
    ├── reward_metrics.py  # tool-level & chain-level 评估函数
```

---

## 📄 1. 数据格式说明（HuggingFace SFT 格式）

### 🔹 SFT 数据（`sft_data.jsonl`）

```json
{
  "instruction": "帮我查一下明天北京的天气",
  "input": "",
  "output": "{\"tool_call\": {\"tool_name\": \"queryWeather\", \"parameters\": {\"location\": \"北京\", \"date\": \"2025-04-24\"}}}"
}
```

---

### 🔹 Reward Model 数据（`rm_data.jsonl`）

```json
{
  "prompt": "帮我查一下明天北京的天气",
  "chosen": "queryWeather(location=北京, date=2025-04-24)",
  "rejected": "queryWeather(location=上海, date=2025-04-24)"
}
```

---

### 🔹 PPO 数据（`ppo_data.jsonl`）

```json
{
  "prompt": "查下现在杭州那边的气温是多少",
  "response": "queryWeather(location=杭州, date=now)"
}
```

---

## 🧠 2. SFT 训练脚本（`train_sft.py`）

基于 [TRL (Transformers Reinforcement Learning)](https://github.com/huggingface/trl)：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

model_name = "qwen/Qwen-7B-Chat"
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
```

---

## 🎯 3. Reward Model 训练脚本（`train_rm.py`）

```python
from trl import RewardTrainer
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("qwen/Qwen-7B", num_labels=1)
dataset = load_dataset("json", data_files="data/rm_data.jsonl")

trainer = RewardTrainer(
    model=model,
    train_dataset=dataset["train"],
    tokenizer=AutoTokenizer.from_pretrained("qwen/Qwen-7B"),
    args=TrainingArguments(output_dir="output/rm", num_train_epochs=3)
)
trainer.train()
```

---

## 💥 4. PPO 强化学习脚本（`train_ppo.py`）

```python
from trl import PPOTrainer, PPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ppo_config = PPOConfig(
    model_name="qwen/Qwen-7B-Chat",
    reward_tokenizer="output/rm",  # RM 路径
    log_with=None,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", trust_remote_code=True),
    tokenizer=AutoTokenizer.from_pretrained("qwen/Qwen-7B-Chat", trust_remote_code=True),
    dataset=load_dataset("json", data_files="data/ppo_data.jsonl")["train"]
)

ppo_trainer.train()
```

---

## 📊 5. 评估脚本（`evaluate.py`）

```python
from utils.tool_schema import validate_tool_output
from utils.reward_metrics import compute_tool_accuracy, compute_chain_success_rate

# 示例
results = [
    {"expected": {"tool": "queryWeather", "params": {"location": "北京"}},
     "generated": {"tool": "queryWeather", "params": {"location": "北京"}}},
    ...
]

print("Tool-Level Accuracy:", compute_tool_accuracy(results))
print("Chain-Level Success Rate:", compute_chain_success_rate(results))
```

---

## ⚙️ 6. 配置模板（示例：`configs/config_sft.yaml`）

```yaml
model_name: qwen/Qwen-7B-Chat
train_file: data/sft_data.jsonl
output_dir: output/sft
num_train_epochs: 3
learning_rate: 5e-5
batch_size: 2
```

---

如果你需要我打包这些脚本、生成完整代码模板、或者构造样本数据/奖励函数，请告诉我：

- ✅ 模型名称（如 Qwen-7B/DeepSeek-VL）
- ✅ 工具 schema（我可以用 OpenAPI 格式帮你自动转 prompt）
- ✅ reward 评估策略细节（比如：意图准确 + 工具名正确 + 参数合理 + 调用成功）

我可以直接帮你生成打通训练链路的完整代码仓库模板。是否继续？
