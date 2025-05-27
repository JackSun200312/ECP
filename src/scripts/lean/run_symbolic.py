# symbolic_runner.py
from pathlib import Path
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.ecp.utils import symbolic_prover

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--max_workers", type=int, default=1)
args = parser.parse_args()

input_path = Path(args.input_path)
output_path = Path(args.output_path)
dataset_name = input_path.stem  # e.g., constructivebench_gpt-4.1-nano-code
output_path.parent.mkdir(parents=True, exist_ok=True)
output_file = output_path / f"{dataset_name}.symbolic.jsonl"

with input_path.open("r", encoding="utf-8") as f:
    entries = [json.loads(line) | {"dataset": dataset_name} for line in f]
def process_entry(entry):
    name = entry["name"]
    header = entry["header"]
    statement = entry["formal_statement"]
    dataset = entry["dataset"]
    result = symbolic_prover(name, header, statement, dataset)
    return {"name": name, "result": result}

with ThreadPoolExecutor(max_workers=args.max_workers) as executor, output_file.open("w", encoding="utf-8") as f_out:
    futures = {executor.submit(process_entry, entry): entry["name"] for entry in entries}
    for future in tqdm(as_completed(futures), total=len(futures)):
        result_entry = future.result()
        f_out.write(json.dumps(result_entry) + "\n")
        f_out.flush()
