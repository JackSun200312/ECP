#!/usr/bin/env python3
import json, csv, argparse
from pathlib import Path

def summarize_run(run_dir: Path):
    partial = run_dir / "partial_results.jsonl"
    legal  = run_dir / "legal_check_results.jsonl"
    if not (partial.exists() and legal.exists()):
        return

    # 1) Answer accuracy
    total_ans = eq_true = 0
    with partial.open() as f:
        for line in f:
            data = json.loads(line)
            total_ans += 1
            if data.get("is_equivalent") in (True, "True"):
                eq_true += 1
    ans_pct = eq_true / total_ans * 100 if total_ans else 0.0

    print(f"Run: {run_dir.name}")
    print(f"  Answer Accuracy : {eq_true}/{total_ans} = {ans_pct:.2f}%")

    # 2) Load legal-check map
    legal_map = {}
    with legal.open() as f:
        for line in f:
            data = json.loads(line)
            name = data.get("name")
            legal_map[name] = data.get("is_legal") in (True, "True")

    # 3) Proving accuracy per CSV
    csv_files = list(run_dir.rglob("compilation_summarize.csv"))
    if not csv_files:
        print("  (no compilation_summarize.csv found)\n")
        return

    for csv_path in csv_files:
        total_rows = provable = 0
        with csv_path.open() as cf:
            reader = csv.DictReader(cf, delimiter="\t")
            for row in reader:
                total_rows += 1
                try:
                    corr = int(row.get("correct", 0))
                except ValueError:
                    corr = 0
                name = row.get("name")
                if corr > 0 and legal_map.get(name, False):
                    provable += 1

        pct = provable / total_rows * 100 if total_rows else 0.0
        rel = csv_path.relative_to(run_dir)
        print(f"  Proving Accuracy [{rel}]: {provable}/{total_rows} = {pct:.2f}%")

    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", choices=["constructivebench", "putnam", "test"], default="constructivebench")
    args = parser.parse_args()

    base = Path("output") / args.base
    for run in sorted(base.iterdir()):
        if run.is_dir():
            summarize_run(run)

if __name__ == "__main__":
    main()
