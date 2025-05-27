#!/usr/bin/env python3
import json
import argparse
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from prover.prover.lean.verifier import verify_lean4_file

def main():
    parser = argparse.ArgumentParser(
        description='Compile Lean code using lake repl in parallel threads.'
    )
    parser.add_argument(
        '--input_path', type=str, required=True,
        help='Path to input JSON file with code entries'
    )
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='Path to write output JSON with compilation results'
    )
    parser.add_argument(
        '--cpu', default=8, type=int,
        help='Number of parallel threads to use'
    )
    parser.add_argument(
        '--timeout', default=300, type=int,
        help='Timeout (seconds) for each verification call'
    )
    args = parser.parse_args()

    # Load your list of { "name": ..., "code": ... } entries
    with open(args.input_path, 'r') as f:
        codes = json.load(f)

    def compile_one(idx, entry):
        # re-use your existing verify_lean4_file function
        result = verify_lean4_file(
            code=entry["code"],
            timeout=args.timeout
        )
        return idx, result

    # Spin up a pool of lightweight threads
    with ThreadPoolExecutor(max_workers=args.cpu) as pool:
        futures = {
            pool.submit(compile_one, i, entry): i
            for i, entry in enumerate(codes)
        }
        for future in tqdm(as_completed(futures), total=len(futures), file=sys.stderr):
            i, result = future.result()
            codes[i]["compilation_result"] = result

    # Write back the full list with embedded results
    with open(args.output_path, 'w') as f:
        json.dump(codes, f, indent=4)

if __name__ == '__main__':
    main()