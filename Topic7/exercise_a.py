import requests
import os
import json

URL = "https://asta-tools.allen.ai/mcp/v1"

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"]
}

payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {}
}

resp = requests.post(URL, headers=headers, json=payload)
text = resp.text

# extract the JSON after "data:"
for line in text.splitlines():
    if line.startswith("data:"):
        data = line.replace("data:", "").strip()
        resp_json = json.loads(data)
        break

tools = resp_json["result"]["tools"]

for tool in tools:
    print(f"\nTool: {tool['name']}")
    print(f"  Description: {tool['description']}")

    schema = tool["inputSchema"]
    props = schema.get("properties", {})
    required = schema.get("required", [])

    for name, meta in props.items():
        t = meta.get("type", "unknown")
        if name in required:
            print(f"  Required: {name} ({t})")
        else:
            print(f"  Optional: {name} ({t})")