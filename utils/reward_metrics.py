def compute_tool_accuracy(results):
    correct = sum(validate_tool_output(r["generated"], r["expected"]) for r in results)
    return correct / len(results)

def compute_chain_success_rate(results):
    return compute_tool_accuracy(results)  # 暂时用同一指标表示链成功
