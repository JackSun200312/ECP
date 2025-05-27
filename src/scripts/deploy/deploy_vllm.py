import subprocess
import time
import requests
import torch

# Configuration
MODEL_PATH = "/model-weights/DeepSeek-R1-Distill-Qwen-1.5B"
PORT = 30000
HOST = "0.0.0.0"
LOGFILE = "server.log"
API_URL = f"http://127.0.0.1:{PORT}/v1/models"
NUM_GPUS = torch.cuda.device_count()

def is_server_running():
    """Check if the vLLM OpenAI-compatible API server is already running."""
    try:
        r = requests.get(API_URL, timeout=2)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False

def start_vllm_server():
    """Start the vLLM server with OpenAI-compatible endpoints."""
    if is_server_running():
        print(f"vLLM server already running at {API_URL}")
        return

    print(f"Starting OpenAI-compatible vLLM server on {HOST}:{PORT} with {NUM_GPUS} GPU(s)...")

    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--host", HOST,
        "--port", str(PORT),
        "--tensor-parallel-size", str(NUM_GPUS),
        "--dtype", "bfloat16",
    ]

    with open(LOGFILE, "w") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)

    print(f"[✔] Server launched with PID {process.pid}")
    print(f"[📄] Logging to {LOGFILE}")


if __name__ == "__main__":
    start_vllm_server()
