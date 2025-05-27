import subprocess
import torch

logfile = "server.log"

# Model path (change as needed)
model_name = 'Your model name'
# Automatically detect number of available GPUs
num_gpus = torch.cuda.device_count()

with open(logfile, "w") as f:
    process = subprocess.Popen([
        "python", "-m", "sglang.launch_server",
        "--trust-remote-code",
        "--model-path", model_name,
        "--tensor-parallel-size", str(torch.cuda.device_count()),
        "--host", "0.0.0.0",
        "--port", "30000",
        '--dtype', 'bfloat16'
    ], stdout=f, stderr=subprocess.STDOUT)

print(f"[✔] Server launched with PID {process.pid}")
print(f"[📄] Logging output to {logfile}")
print(f"[🖥] Detected {num_gpus} GPU(s), set tensor-parallel-size to {num_gpus}")
