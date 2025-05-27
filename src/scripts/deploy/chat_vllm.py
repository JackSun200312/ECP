import requests
import json

BASE_URL = "http://127.0.0.1:30000"
MODELS_URL = f"{BASE_URL}/v1/models"
COMPLETION_URL = f"{BASE_URL}/v1/completions"

# Step 1: Query the server for available models
response = requests.get(MODELS_URL)
model_list = response.json().get("data", [])


# Use the first model ID returned
model_id = model_list[0]["id"]
print(f"Using model: {model_id}")

# Step 2: Run a completion request
payload = {
    "model": model_id,
    "prompt": "Hi",
    "temperature": 0.0,
    "max_tokens": 512,
    "top_p": 1.0,
    "n": 1
}

response = requests.post(
    COMPLETION_URL,
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

print(response.json())
