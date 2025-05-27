import os
import json
import concurrent.futures
from tqdm import tqdm
from src.ecp.agent import Coder, Conjecturer, Autoformalizer, EquivalenceChecker, FormalConjecturer
import logging
logging.basicConfig(level=logging.WARNING)
import torch
import argparse
from datetime import datetime
from pathlib import Path
from src.ecp.utils import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="ECP: Enumerate-Conjecture-Prove pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--mode",
        choices=["answer_gen", "autoformalize", 'proof_gen'],
        default="answer_gen",
        help=(
            "Which pipeline to run:\n"
            "answer_gen → run_answer_gen\n",
            'proof_gen → run_proof_gen'
            "autoformalize   → run_autoformalizer\n"
        )
    )
    parser.add_argument(
        "--problem_name",
        type=lambda s: "all" if s == "all" else s.split(","),
        default="all",
        help=(
            "Which problem(s) to process:\n"
            "  all               → every problem\n"
            "  name1,name2,...   → a comma-separated list of problem names\n"
        )
    )
    parser.add_argument("--conjecturer_model", type=str, default='deepseek-chat')
    parser.add_argument("--enumerator_model", type=str, default='deepseek-chat')
    parser.add_argument("--autoformalizer_model", type=str, default='deepseek-chat')
    parser.add_argument("--judge_model", type=str, default='deepseek-chat')
    parser.add_argument("--prover_model", type=str, default='Goedel-LM/Goedel-Prover-SFT')
    
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--coder_max_attempt", type=int, default=3)
    parser.add_argument("--autoformalizer_max_attempt", type=int, default=5)
    parser.add_argument(
        "--enable_enumerator",
        type=lambda x: x.lower() == 'true',
        default=True,
        help="Enable the coder loop (enumerator + conjecturer). Pass 'true' or 'false'."
    )
    
    parser.add_argument("--output_dir", type=str, default='default')
    parser.add_argument("--temp_formalization_dir", type=str, default='Formalization/Temp')
    parser.add_argument("--temp_python_dir", type=str, default='Temp/')
    parser.add_argument("--problem_path", choices=["constructivebench", "putnam", 'test'], default='constructivebench')
    parser.add_argument("--pass_at_n", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--use_embedding_search",
        type=lambda x: x.lower() == 'true',
        default=False
    )
    parser.add_argument(
        "--enable_llm_judge",
        type=lambda x: x.lower() == 'true',
        default=False
    )


    args = parser.parse_args()
    return args
def run_answer_gen(
    output_dir: str,
    problem_path: str,
    problem_name,
    use_coder: bool,
    enumerator_model: str,
    conjecturer_model: str,
    equivalence_checker_model: str,
    max_tokens: int,
    coder_max_attempt: int,
    autoformalizer_max_attempt: int,
    temp_formalization_dir: str,
    temp_python_dir: str,
    timeout: int,
    num_cpu: int,
    enable_llm_judge: bool = False,
    use_embedding_search : bool = True
):

    os.makedirs(temp_python_dir, exist_ok=True)
    accumulative_path = os.path.join(output_dir, "partial_results.jsonl")
    with open(problem_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_results = []
    processed_indices = set()
    if os.path.exists(accumulative_path):
        with open(accumulative_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_indices.add(entry["name"])
                    processed_results.append(entry)
                except json.JSONDecodeError:
                    continue

    if problem_name == 'all':
        entries = [
            e for e in data
            if e.get("is_formalized") == 'True'
            and e['name'] not in processed_indices
        ]
    elif isinstance(problem_name, list):
        entries = [
            e for e in data
            if e['name'] in problem_name and e['name'] not in processed_indices
        ]
    else:
        entries = []



    os.makedirs(temp_formalization_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'llm_history'), exist_ok=True)

    print(f"problems to process: {len(entries)} (Skipped {len(processed_indices)} already processed)")

    results = []

    def process_problem(entry):
        
        name = entry['name']
        answer = entry['formal_answer']
        header = entry['header']
        problem = f"{header}\n{entry['answer_part']}\n{entry['theorem_part']}"
        answer_type = entry['answer_type']
        formalization_path = os.path.join(temp_formalization_dir, f'{name}_{enumerator_model}.lean')

        if use_coder:
            python_path = os.path.join(temp_python_dir, f'{name}_{enumerator_model}.py')
            enumerator = Coder(model=enumerator_model, max_tokens=max_tokens, filename=python_path)
            program, enumerated_answer, attempts = enumerator.write_program_loop(
                problem, max_attempt=coder_max_attempt, timeout=timeout
            )
            with open(os.path.join(output_dir, "llm_history", f"enumerator_{name}.txt"), "w", encoding="utf-8") as f:
                f.write(str(enumerator.history))

            conjecturer = FormalConjecturer(
                model=conjecturer_model,
                max_tokens=max_tokens,
                filename=formalization_path,
                use_embedding_search=use_embedding_search
            )
            proposed_answer, successful = conjecturer.conjecture_answer_loop(
                problem, enumerated_answer, answer_type, max_attempt=autoformalizer_max_attempt
            )
        else:
            program = None
            enumerated_answer = None
            attempts = None
            conjecturer = FormalConjecturer(
                model=conjecturer_model,
                max_tokens=max_tokens,
                filename=formalization_path,
                use_embedding_search=use_embedding_search
            )
            proposed_answer, successful = conjecturer.conjecture_answer_loop(
                problem, "", answer_type
            )

        with open(os.path.join(output_dir, "llm_history", f"conjecturer_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(str(conjecturer.history))

        if successful:
            if enable_llm_judge:
                equivalence_checker = EquivalenceChecker(
                    model=equivalence_checker_model, max_tokens=max_tokens
                )
                is_equivalent = equivalence_checker.check_equivalence(
                    problem, answer, proposed_answer
                )
                explanation = equivalence_checker.history
                with open(os.path.join(output_dir, "llm_history", f"equivalence_checker_{name}.txt"), "w", encoding="utf-8") as f:
                    f.write(str(equivalence_checker.history))
            else:
                is_equivalent = formal_equivalence_checker(
                    name, header, answer_type, proposed_answer, answer
                )
                explanation = (
                    "equivalent by formal checking"
                    if is_equivalent == 'True'
                    else "non-equivalent by formal checking"
                )
        else:
            is_equivalent = "False"
            explanation = "Failed Lean check"
        if not is_legal_answer(entry['theorem_part'], answer_type, proposed_answer):
            is_equivalent = "False"
            explanation    = "Illegal answer: failed legal‐answer validation"
            
        result = {
            "name": name,
            "actual_answer": answer,
            "proposed_answer": proposed_answer,
            "is_equivalent": is_equivalent,
            "explanation": explanation,
            "enumerated_answer": enumerated_answer,
            "program": program,
            "attempts": attempts,
        }
        with open(accumulative_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    if num_cpu == 1:
        for entry in tqdm(entries):
            results.append(process_problem(entry))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpu) as executor:
            futures = {executor.submit(process_problem, e): e for e in entries}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                results.append(future.result())

    results += processed_results
    total_checked = sum(1 for r in results if r.get("is_equivalent") in [True, 'True'])
    accuracy = total_checked / len(results) if results else 0
    print(f"Accuracy: {total_checked}/{len(results)} = {accuracy:.2%}")

    base_name = os.path.splitext(os.path.basename(problem_path))[0]
    eval_path = f"data/eval/{base_name}_{os.path.basename(output_dir)}.jsonl"
    generate_formalization_with_candidate_answers(results, data, eval_path)
    write_legal_check_results(output_dir, results, data)
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def run_autoformalizer(
    output_dir: str,
    problem_path: str,
    problem_name,
    judge_model: str,
    autoformalizer_model: str,
    max_tokens: int,
    enable_llm_judge: bool,
    temp_formalization_dir: str,
    max_attempt: int,
    num_cpu: int,
    use_embedding_search : bool = True
):
    # Load data
    with open(problem_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Prepare output dirs
    os.makedirs(os.path.join(output_dir, 'llm_history'), exist_ok=True)
    os.makedirs(temp_formalization_dir, exist_ok=True)
    partial_results_path = os.path.join(output_dir, "partial_results.jsonl")

    # Load already processed names
    processed_names = set()
    partial_results = []
    if os.path.exists(partial_results_path):
        with open(partial_results_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_names.add(entry.get("name"))
                    partial_results.append(entry)
                except json.JSONDecodeError:
                    continue

    # Select entries by name
    if problem_name == 'all':
        entries = [
            e for e in data
            if e['name'] not in processed_names
        ]
    elif problem_name == 'remaining':
        entries = [
            e for e in data
            if e.get('is_formalized', 'False') != 'True'
               and e['name'] not in processed_names
        ]
        
    elif isinstance(problem_name, list):
        entries = [
            e for e in data
            if e['name'] in problem_name
               and e['name'] not in processed_names
        ]
    else:
        entries = []

    print(f"Total problems to be processed: {len(entries)}")

    # Start from any prior partial results
    results = partial_results.copy()

    def process_problem(entry):
        name = entry['name']
        formalization_path = os.path.join(
            temp_formalization_dir,
            f"{name}_{autoformalizer_model}.lean"
        )

        try:
            autoformalizer = Autoformalizer(
                filename=formalization_path,
                model=autoformalizer_model,
                judge_model=judge_model,
                max_tokens=max_tokens,
                enable_judge=enable_llm_judge,
                use_embedding_search=use_embedding_search
            )
            
            formalization, passed_lean, passed_llm, attempts_log = \
                autoformalizer.autoformalize_loop(
                    entry['problem'],
                    entry['answer'],
                    name,
                    max_attempt=max_attempt
                )

            # Save LLM history
            hist_path = os.path.join(
                output_dir,
                "llm_history",
                f"autoformalizer_{name}.txt"
            )
            with open(hist_path, "w", encoding="utf-8") as f:
                f.write(str(autoformalizer.history))

            # Post-process the .lean file: insert noncomputable abbrev
            with open(formalization_path, 'r', encoding='utf-8') as f:
                text = f.read()
            text = text.replace('\nabbrev', '\nnoncomputable abbrev')

            # Standardize theorem & answer names
            ans_info = process_answer(text)
            answer_name = ans_info.get('name', '')
            theorem_name = extract_theorem_name(text)
            target_ans = f"{name}_answer" if not name[0].isdigit() else f"P{name}_answer"
            target_thm = f"{name}" if not name[0].isdigit() else f"P{name}"
            if answer_name:
                text = re.sub(rf"\b{re.escape(answer_name)}\b", target_ans, text)
            if theorem_name:
                text = re.sub(rf"\b{re.escape(theorem_name)}\b", target_thm, text)

            # Check formatting & legality
            formatted_ok = (
                passed_lean and passed_llm
                and sanity_check_for_full_formalization(text) == ''
            )
            legal_ok = is_legal_answer(
                entry['theorem_part'],
                entry['answer_type'],
                ans_info.get('answer', '')
            )
            is_formatted = 'True' if (formatted_ok and legal_ok) else 'False'

            # Build result dict
            result = {
                "name": name,
                "passed_lean_check": str(passed_lean),
                "passed_llm_check": str(passed_llm),
                "is_formatted": is_formatted,
                "formalization": text,
                "header": entry.get('header'),
                "answer_part": entry.get('answer_part'),
                "theorem_part": entry.get('theorem_part'),
                "theorem_part_with_answer": entry.get('theorem_part_with_answer'),
                "formal_answer": entry.get('formal_answer'),
                "answer_type": entry.get('answer_type'),
                "problem": entry['problem'],
                "answer": entry['answer'],
                "attempts": attempts_log,
            }

            # If fully formatted, enrich with extra fields
            if is_formatted == 'True':
                extra = process_entry(entry)
                result.update(extra)

        except Exception as e:
            result = {
                "name": name,
                "error": str(e),
                "is_formatted": "False"
            }

        # Append to partial results log
        with open(partial_results_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        return result

    # Execute processing
    if num_cpu == 1:
        for entry in tqdm(entries):
            res = process_problem(entry)
            results.append(res)
            print(f"Processed problem {res.get('name', 'N/A')}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpu) as executor:
            futures = {executor.submit(process_problem, e): e for e in entries}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                res = future.result()
                results.append(res)
                print(f"Processed problem {res.get('name', 'N/A')}")

    
    name_to_entry = { e["name"]: e for e in data }

    # now you only do a single pass over results:
    updated = 0
    for r in results:
        if (
            r.get("passed_lean_check") == "True"
            and r.get("passed_llm_check") == "True"
            and r.get("is_formatted") == "True"
        ):
            e = name_to_entry.get(r["name"])
            if not e:
                continue
            # perform your updates directly
            e["formalization"]              = r["formalization"]
            e["is_formalized"]              = "True"
            e["header"]                     = r["header"]
            e["answer_part"]                = r["answer_part"]
            e["theorem_part"]               = r["theorem_part"]
            e["theorem_part_with_answer"]   = r["theorem_part_with_answer"]
            e["formal_answer"]              = r["formal_answer"]
            e["answer_type"]                = r["answer_type"]
            updated += 1

    # Write back updated problems file
    with open(problem_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Updated {updated} entries in {problem_path} with formalizations.")

    # Export Lean files if needed
    write_formalizations_to_files(problem_path, get_constructive_theorem=True)

    
def run_proof_gen(
    input_path: str,
    model_path: str,
    output_dir: str,
    n: int,
    gpu: int,
    cpu: int,
):
    """
    Run the prover pipeline by calling:
    1. step1_inference
    2. step2_compile
    3. step3_summarize_compile
    """
    import subprocess

    os.makedirs(output_dir, exist_ok=True)

    if model_path == 'symbolic':
        inference_cmd = [
        "python", "-m", "src.scripts.lean.run_symbolic",
        "--input_path", input_path,
        "--output_path", output_dir,
        "--max_workers", str(cpu),
        ]
        subprocess.run(inference_cmd, check=True)
    else:
            
        # Step 1: Inference
        print(f"Pass at {n}")
        print("Generating Proof...")
        inference_cmd = [
            "python", "-m", "prover.eval.step1_inference",
            "--input_path", input_path,
            "--model_path", model_path,
            "--output_dir", output_dir,
            "--split", 'valid',
            "--n", str(n),
            "--gpu", str(gpu),
        ]
        subprocess.run(inference_cmd, check=True)

        # Step 2: Compile
        input_file = os.path.join(output_dir, "to_inference_codes.json")
        compile_output_path = os.path.join(output_dir, "code_compilation.json")
        print("Verifying Proof...")
        compile_cmd = [
            "python", "-m", "prover.eval.step2_compile",
            "--input_path", input_file,
            "--output_path", compile_output_path,
            "--cpu", str(cpu),
        ]
        subprocess.run(compile_cmd, check=True)

        # Step 3: Summarize
        summarize_output_path = os.path.join(output_dir, "compilation_summarize.json")
        summarize_cmd = [
            "python", "-m", "prover.eval.step3_summarize_compile",
            "--input_path", compile_output_path,
            "--output_path", summarize_output_path,
            "--field", 'complete',
        ]
        subprocess.run(summarize_cmd, check=True)

        print("Prover pipeline completed.")

# Update the main entry point
if __name__ == "__main__":
    args = parse_args()
    
    num_cpu = os.cpu_count()
    num_gpu = args.gpu
    auto_set_cuda_visible_devices()
    if args.mode == 'answer_gen':
        problem_path = f'data/dataset/{args.problem_path}.json'
        if args.enable_enumerator:

            output_dir = f'output/{args.problem_path}/{args.conjecturer_model}-code'
        else:
            output_dir = f'output/{args.problem_path}/{args.conjecturer_model}'
    elif args.mode == 'autoformalize':
        problem_path = f'data/dataset/{args.problem_path}.json'
        output_dir = f'output/{problem_path}/{args.model}-formalization'
    elif args.mode == 'proof_gen':

        if args.enable_enumerator:
            problem_path = f'data/eval/{args.problem_path}_{args.conjecturer_model}-code.jsonl'
        else:
            problem_path = f'data/eval/{args.problem_path}_{args.conjecturer_model}.jsonl'
        prover_model_name = args.prover_model.split('/')
        if len(prover_model_name)>1:
            prover_model_name = prover_model_name[-1]
        else:
            prover_model_name = prover_model_name[0]
        if args.enable_enumerator:

            output_dir = f'output/{args.problem_path}/{args.conjecturer_model}-code/{prover_model_name}'
        else:
            output_dir = f'output/{args.problem_path}/{args.conjecturer_model}/{prover_model_name}'
    if args.mode == ("answer_gen"):
        run_answer_gen(
            output_dir=output_dir,
            problem_path=problem_path,
            problem_name=args.problem_name,
            use_coder=args.enable_enumerator,
            enumerator_model=args.enumerator_model,
            conjecturer_model=args.conjecturer_model,
            equivalence_checker_model=args.judge_model,
            max_tokens=args.max_tokens,
            coder_max_attempt=args.coder_max_attempt,
            autoformalizer_max_attempt=args.autoformalizer_max_attempt,
            temp_formalization_dir=args.temp_formalization_dir,
            temp_python_dir=args.temp_python_dir,
            timeout=args.timeout,
            num_cpu=num_cpu,
            use_embedding_search = args.use_embedding_search
        )

    elif args.mode == ("autoformalize"):
        run_autoformalizer(
            output_dir=output_dir,
            problem_path=problem_path,
            problem_name=args.problem_name,
            judge_model=args.judge_model,
            autoformalizer_model=args.autoformalizer_model,
            max_tokens=args.max_tokens,
            enable_llm_judge=args.enable_llm_judge,
            temp_formalization_dir=args.temp_formalization_dir,
            max_attempt=args.autoformalizer_max_attempt,
            num_cpu=args.num_cpu,
            use_embedding_search = args.use_embedding_search
        )

    elif args.mode == "proof_gen":
        run_proof_gen(
            input_path=problem_path,
            model_path=args.prover_model,
            output_dir=output_dir,

            n=args.pass_at_n,
            gpu=num_gpu,
            cpu=num_cpu,
        )
