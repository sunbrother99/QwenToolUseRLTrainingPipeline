TOOL_SCHEMAS = {
    "queryWeather": {"params": ["city", "date"]},
    "queryLocation": {"params": ["city"]},
}

def validate_tool_output(generated, expected):
    if generated["tool"] != expected["tool"]:
        return False
    for param in TOOL_SCHEMAS[generated["tool"]]["params"]:
        if generated["params"].get(param) != expected["params"].get(param):
            return False
    return True
