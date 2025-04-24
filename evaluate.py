# evaluate.py
from utils.tool_schema import validate_tool_output
from utils.reward_metrics import compute_tool_accuracy, compute_chain_success_rate

results = [
    {"expected": {"tool": "queryWeather", "params": {"city": "北京", "date": "2025-04-24"}},
     "generated": {"tool": "queryWeather", "params": {"city": "北京", "date": "2025-04-24"}}},
    # Add more result pairs...
]

print("Tool-Level Accuracy:", compute_tool_accuracy(results))
print("Chain-Level Success Rate:", compute_chain_success_rate(results))
