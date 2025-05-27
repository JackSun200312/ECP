
import os
import re
import subprocess
import sys
import json
import threading
import signal
import jsonlines
from pathlib import Path
from src.ecp.retrieve_definition import search_theorem

from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
def replace_last_occurrence(s: str, old: str, new: str) -> str:
    """Replace the last occurrence of `old` with `new` in string `s`."""
    index = s.rfind(old)
    if index == -1:
        return s  # `old` not found, return the original string
    return s[:index] + new + s[index + len(old):]

def run_python_program(code_string: str, timeout, temp_filename):
    """Run a python program with timeout"""
    if not code_string:
        return False, "Error: Program is empty"
    with open(temp_filename, "w") as temp_file:
        temp_file.write(code_string)

    result_container = {"stdout": [], "stderr": [], "process": None}
    stop_event = threading.Event()

    def run_subprocess():
        try:
            process = subprocess.Popen(
                [sys.executable, temp_filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            result_container["process"] = process

            while True:
                line = process.stdout.readline()
                if line == "" and process.poll() is not None:
                    break
                if stop_event.is_set():
                    break
                result_container["stdout"].append(line.strip())

            process.stdout.close()
            process.wait()
        except Exception as e:
            result_container["stderr"].append(str(e))

    thread = threading.Thread(target=run_subprocess)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        stop_event.set()
        process = result_container.get("process")
        if process and process.poll() is None:
            os.kill(process.pid, signal.SIGKILL)
        thread.join()
        result_container["stdout"].append("timeout triggered")

    stdout = "\n".join(result_container["stdout"])
    stderr = "\n".join(result_container["stderr"])

    if len(stdout) > 2000:
        stdout = stdout[:2000] + "\nOutput cuts off here."

    if stderr:
        return False, f"Error: Program exited with errors:\n{stderr}"
    elif not stdout.strip():
        return False, "Error: Program outputs empty result"
    else:
        return True, stdout

def has_repeating_substring(s, repetitions=10):
    """
    Check if string s contains any substring that is repeated consecutively 
    at least 'repetitions' times.
    
    Parameters:
      s : str
         The string to test.
      repetitions : int
         The minimum number of consecutive repetitions required.
    
    Returns:
      bool : True if such a repeating substring is found, False otherwise.
    """
    # Build the regex pattern using a capturing group.
    # (. +?) captures any substring non-greedily.
    # \1{repetitions-1,} ensures that the captured substring repeats at least (repetitions-1) more times.
    if len(s) >= 10000:
        return True
    s = re.sub(r' {2,}', ' ', s)

    s = re.sub(r'\n{2,}', '\n', s)
    pattern = re.compile(r'(.+?)\1{' + str(repetitions - 1) + r',}')
    
    # Search the string for the pattern.
    match = pattern.search(s)
    return bool(match)

def match_equals(a: str, b: str, text: str) -> bool:
    pattern1 = rf"{re.escape(a)}\s*=\s*{re.escape(b)}"
    pattern2 = rf"{re.escape(b)}\s*=\s*{re.escape(a)}"
    return bool(re.search(pattern1, text)) or bool(re.search(pattern2, text))

def sanity_check_for_full_formalization(lean_source, check_answer = True): 
    error = ''
    abbrev_pattern = r"\n(?:noncomputable\s+)?abbrev "
    abbrev_count = len(re.findall(abbrev_pattern, lean_source))
    answer_content = process_answer(lean_source)
    answer_name = answer_content['name']
    answer_value = answer_content['answer']
    
    if check_answer:
        parsed = process_answer(lean_source)
        answer_value, answer_name = parsed["answer"], parsed["name"]
        if not answer_value:
            error += (
                "Caution: Answer cannot be parsed. Begin with "
                "noncomputable abbrev <name>_answer : <type> := …"
            )
        if lean_source.count(answer_name) <= 1:
            error += (
                "Caution: <name>_answer not referenced inside the theorem."
            )
        
        if answer_value != '' and (match_equals(answer_name, answer_value, lean_source) or match_equals(answer_value,answer_name,lean_source)):
            error += f"In theorem part, you referenced the answer name and its answer value by {answer_value} = {answer_name}, which should never happen. You should mention answer name {answer_name} somewhere in theorem, and you should not mention the actual answer {answer_value}"
        
    if has_repeating_substring(lean_source):
        error += "This formalization has repeating substrings of more than 10 times. Please re-formalize it."
    if len(lean_source) >= 10000:
        error += "This formalization is too long"
    if abbrev_count != 1:
        error += "You did not use noncomputable abbrev to state the answer of the problem, or you have used more than once. You should use noncomputable abbrev exactly once to state answer."
    if lean_source.count("\ntheorem") != 1:
        error += "You did not use theorem to formalize the problem statement, or you have used more than once. You should use theorem exactly once to state problem statement."
    if lean_source.count(answer_name) < 2:
        error += "You did not mention the answer you defined in your problem statement. Mention your answer in problem statement using appropriate connectives."
    if lean_source.count("import ") < 1:
        error += "You did not mention import packages. You should import Mathlib at beginning of the formalization."

    if '\n' not in lean_source:
        error += 'Your formalization contains only one line. You should use \n properly to change lines.'

    # if 'open' not in lean_source:
    #     error += 'Your formalization misses open <namespace> statements, e.g. open Nat.'
    return error
    

def sanity_check_for_conjectured_answer(lean_source): 
    error = ''
    # if "abbrev " in lean_source:
    #     error += "You should only include the content of proposed answer, and you should not include headers like abbrev <answer_name> : <type> :=."
    
    if "theorem " in lean_source:
        error += "You should only include the content of proposed answer, and you should not include theorem headers like theorem <theorem_name> : <type> := beyond the answer part."
    
    return error
    
def parse_lean_errors(file_name: str, error_message: str, lean_source: str, use_embedding_search = True):
    error_dict = []
    # Remove specific warnings using regex
    error_message = re.sub(rf"{re.escape(file_name)}:\d+:\d+: use the command set_option checkBinderAnnotations false to disable the check", "", error_message)
    error_message = re.sub(rf"{re.escape(file_name)}:\d+:\d+: warning: declaration uses 'sorry'", "", error_message)
    
    # Regex pattern to match errors (ignoring 'sorry' warnings)
    error_pattern = re.compile(rf"{re.escape(file_name)}:(\d+):(\d+): error: (.+?)(?=\n{re.escape(file_name)}:\d+:\d+: error:|\Z)", re.DOTALL)
    
    for i in error_pattern.finditer(error_message):
        line, column, message = i.groups()
        key = f"{line}:{column}"

        error_dict.append((key, message.strip()))
    error_dict = [(i[0], i[1]) for i in error_dict if 'warning:' not in i[1]]
    # Extract the relevant definitions
    source_lines = lean_source.split('\n')
    extracted_errors = []

    for i in error_dict:
        line, column = map(int, i[0].split(':'))
        try:
            line_content = source_lines[line - 1]
        except IndexError:
            # If the line number is out of range, log a default message.
            line_content = f"[Line {line} not found in source]"
        try:
            problematic_word = line_content[column:min(len(line_content), column+20)]
        except Exception:
            problematic_word = "[Could not extract problematic word]"
                
        extracted_errors.append({
            "key": i[0],
            "problematic_word": problematic_word,
            "line": line_content,
            "error": i[1]
        })
    processed_error = ''
    counter = 0
    for i in extracted_errors:
        counter += 1
        line, column = key.split(':')
        key = i['key']
        problematic_word = i['problematic_word']
        line_content = i['line']
        error = i['error']
        
        # Special handling for 'invalid field' and 'does not contain' errors
        retrieved_theorem_txt = ""
        if "invalid field" in error and "does not contain" in error:
            match = re.search(r"invalid field '(\w+)', the environment does not contain '(.*?)'", error)
            if match:
                full_identifier = match.group(2)  # Get the full missing identifier
                retrieved_theorems = search_theorem(full_identifier, use_embedding_search = use_embedding_search)

                for num, definition in enumerate(retrieved_theorems):
                    retrieved_theorem_txt += f'Definition {num+1}: {definition["type_signature"]}. Description: {definition["description"]}\n' if definition["description"] else f'Definition {num+1}: {definition["type_signature"]}.\n'
        if "unknown constant" in error:
            match = re.search(r"unknown constant '(.*?)'", error)
            if match:
                full_identifier = match.group(1)  # Get the full missing identifier
                retrieved_theorems = search_theorem(full_identifier, use_embedding_search = use_embedding_search)

                for num, definition in enumerate(retrieved_theorems):
                    retrieved_theorem_txt += f'Definition {num+1}: {definition["type_signature"]}. Description: {definition["description"]}\n' if definition["description"] else f'Definition {num+1}: {definition["type_signature"]}.\n'

        
        # Special handling for type mismatches
        elif "has type" in error:
            error += "\nPossible Fix: Check if the expected type aligns with the provided type."
        elif "ambiguous, possible interpretations" in error:
            error += "\nPossible Fix: Specify the namespace in the definition."
                # Special handling for binder annotation errors
        elif "invalid binder annotation" in error:

            retrieved_theorems = search_theorem(problematic_word, use_embedding_search = use_embedding_search)
            for num, definition in enumerate(retrieved_theorems):
                retrieved_theorem_txt += f'Definition {num+1}: {definition["type_signature"]}. Description: {definition["description"]}\n' if definition["description"] else f'Definition {num+1}: {definition["type_signature"]}.\n'
            error += "\nPossible Fix: Ensure the correct class instance is used."
        
        processed_error += f'Error {counter}: At line {line} column {column}, which is at "{problematic_word}" in "{line_content}", there is an error: {error}.\n'

        if retrieved_theorem_txt:
            processed_error += f'Here are ground-truth definitions retrieved from Lean documentation for you: {retrieved_theorem_txt}\n'
    return processed_error

def run_lean(lean_file: Path = Path("Formalization/IMO2023SL/C1.lean"), sanity_check = True, check_answer = True, use_embedding_search = True):
    lean_file = str(lean_file)  # ← convert Path to string

    # Run the command in the specified working directory
    project_dir = lean_file.split('/')[0]
    file_dir = '/'.join(lean_file.split('/')[1:])

    result = subprocess.run(
        ["lake", "env", "lean", file_dir],  # Just check the file
        cwd=project_dir,
        capture_output=True,
        text=True
    )
    
    # Read the lean source file
    with open(lean_file, "r", encoding="utf-8") as f:
        lean_source = f.read()
    errors = ''
    if sanity_check:
        errors += sanity_check_for_full_formalization(lean_source, check_answer)
    errors += parse_lean_errors(file_dir, result.stdout, lean_source, use_embedding_search = use_embedding_search)
    return errors


def run_lean_extract_goal(lean_file: Path = Path("Formalization/IMO2023SL/C1.lean")):
    lean_file = str(lean_file)  # ← convert Path to string

    # Run the command in the specified working directory
    project_dir = lean_file.split('/')[0]
    file_dir = '/'.join(lean_file.split('/')[1:])

    result = subprocess.run(
        ["lake", "env", "lean", file_dir],  # Just check the file
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    result = result.stdout.strip()
    match = re.search(r"error: unsolved goals\n(.+)", result, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def process_answer(lean_code: str):

    direct_pattern = re.compile(
        r"noncomputable\s+abbrev\s+([a-zA-Z0-9_]+)\s*:\s*([^=]+?)\s*:=\s*([\s\S]+?)\n*(?=(noncomputable|theorem|abbrev|def|/--|--|structure|inductive)\b)",
        re.DOTALL
    )


    match = direct_pattern.search(lean_code)
    if match:
        name, typ, solution = map(str.strip, match.groups()[:3])
        new_abbrev = f"noncomputable abbrev {name} : {typ} := sorry\n"
        new_code = lean_code.replace(match.group(0), new_abbrev)
        return {"name": name, "answer": solution, "type": typ, "new_code": new_code}
    else:

        return {"name": "", "answer": "", "type": "", "new_code": lean_code}




def insert_answer_comment(answer: str, text: str) -> str:
    text = text.replace("sorry", answer, 1)
    return text
    

def parse_answer_in_delimiter(text):
    match = re.search(r'<<<(.*?)>>>', text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        
        return answer
    match = re.search(r'boxed{(.*?)}', text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return answer
    return "None"

def parse_program_and_answer_in_delimiter(text):

    match = re.search(r'<<<(.*?)>>>', text, re.DOTALL)

    answer, program = None, None
    if match:
        answer = match.group(1).strip()
        
    match = re.search(r'```python\s*\n*(.*?)```', text, re.DOTALL)
    if match:
        program = match.group(1).strip()
        
    return answer, program

def parse_program_in_delimiter(text):

    match = re.search(r'```python\s*\n*(.*?)```', text, re.DOTALL)
    if match:
        program = match.group(1).strip()
        return program
    else:
        return text
def parse_lean_in_delimiter(text):

    match = re.search(r'```lean\s*\n*(.*?)```', text, re.DOTALL)
    if match:
        program = match.group(1).strip()
        return program
    match = re.search(r'<<<(.*?)>>>', text, re.DOTALL)
    if match:
        program = match.group(1).strip()
        return program
    return text


def standardize_formalization(formalization):
    formalization = re.sub(r'/--[\s\S]*?-/', '', formalization)
    formalization = re.sub(r'--[\s\S]*?\n', '\n', formalization) # Avoid removing necessary \n
    formalization = re.sub(r'(?::=\n*\s*\n*(by\s*)?\s*sorry\s*)', ':= by sorry\n', formalization)
    return formalization


def extract_theorem_type_annotation(formalization: str) -> str:
    def skip_whitespace(s, pos):
        while pos < len(s) and s[pos].isspace():
            pos += 1
        return pos

    def skip_balanced(s, pos):
        if pos >= len(s) or s[pos] not in "({[":
            return pos
        pairs = {'(': ')', '{': '}', '[': ']'}
        open_char = s[pos]
        close_char = pairs[open_char]
        level = 0
        while pos < len(s):
            if s[pos] == open_char:
                level += 1
            elif s[pos] == close_char:
                level -= 1
                if level == 0:
                    pos += 1
                    break
            pos += 1
        return pos
    theorem_index = formalization.find("\ntheorem ")
    if theorem_index == -1:
        return ""
    header = formalization[theorem_index:]
    pos = skip_whitespace(header, len("\ntheorem"))
    while pos < len(header) and not header[pos].isspace() and header[pos] != "(":
        pos += 1
    pos = skip_whitespace(header, pos)
    while pos < len(header) and header[pos] == "(":
        pos = skip_balanced(header, pos)
        pos = skip_whitespace(header, pos)
    if pos >= len(header) or header[pos] != ":":
        return ""
    pos += 1
    pos = skip_whitespace(header, pos)
    return replace_last_occurrence(header[pos:].strip(), ':= by sorry', '')


def extract_theorem_name(formalization: str) -> str:
    if formalization:
        direct_pattern = re.compile(
            r"theorem\s+([a-zA-Z0-9_]+)\s*[\s,()\[\]{}]",
            re.DOTALL
        )
        
        match = direct_pattern.search(formalization)
        if match:
            name = match.groups()[0]
            return name
        else:
            return ""
    else:
        return ''

def extract_preamble(formalization): 
    if "\nnoncomputable abbrev " in formalization:
        idx = formalization.index("\nnoncomputable abbrev ")
    elif "\nabbrev " in formalization:
        idx = formalization.index("\nabbrev ")
    else:
        idx = None
    preamble = formalization[:idx] if idx is not None else ""
    after_preamble = formalization[idx:] if idx is not None else formalization
    return preamble, after_preamble

def find_index(text: str, target: str):
    # Try whole-word match first
    whole_word_pattern = r"\b" + re.escape(target) + r"\b"
    match = re.search(whole_word_pattern, text)
    
    if match:
        return match.start()
    
    # Fallback: exact string match without word boundaries
    fallback_pattern = re.escape(target)
    match = re.search(fallback_pattern, text)

    if match:
        return match.start()
    else:
        return -1  # not found
    
def replace_target_with_worry(text: str, target: str) -> str:
    # Try whole-word match first
    pattern = r"\b" + re.escape(target) + r"\b"
    match = re.search(pattern, text)
    
    if match:
        start, end = match.span()
        return text[:start] + "sorry" + text[end:]  # Remove match
    else:
        # Fall back to simple replacement of first occurrence
        return text.replace(target, "sorry", 1)
    
    
def process_entry(entry):
    """
    Process a single entry by standardizing the formalization, extracting the answer,
    and generating target theorem parts. This version drops the 'informal_prefix' key.
    """
    # Standardize the formalization.
    formalization = standardize_formalization(entry.get("formalization", ""))
    extracted = process_answer(formalization)
    answer_type = extracted['type']
    answer_name = extracted.get('name', "")
    answer_content = extracted.get('answer', '')

    # Count the number of abbrevs (if needed)
    abbrev_pattern = r"\n(?:noncomputable\s+)?abbrev "
    abbrev_count = len(re.findall(abbrev_pattern, formalization))
    preamble, after_preamble = extract_preamble(formalization)
    # Split the formalization into preamble and the rest.


    is_formatted = sanity_check_for_full_formalization(formalization) == ''

    # Split further into answer part and theorem part.
    try:
        theorem_idx = after_preamble.index("\ntheorem")
        
        end_of_answer_idx = find_index(after_preamble, answer_content) + len(answer_content)
        answer_part = after_preamble[:end_of_answer_idx]
        
        answer_part = replace_target_with_worry(answer_part, answer_content)
        answer_part = re.sub(' {2,}', " ", answer_part) 
        answer_part = answer_part.replace('\n','')
        answer_part = answer_part.replace(':=  sorry', ':= sorry')
        
        theorem_part = after_preamble[end_of_answer_idx:].strip('\n')
    except:
        answer_part = after_preamble
        theorem_part = ""

    if is_formatted:
        theorem_part_with_answer = theorem_part.replace(answer_name, f"({answer_content} : {answer_type})")

    else:
        theorem_part_with_answer = ""

    # Build the combined entry.
    new_entry = {
        "name": entry["name"],
        # Drop the informal prefix (if present) as per the merge requirement.
        'problem':entry['problem'],
        "formalization": formalization,
        "header": preamble,
        "answer_part": answer_part,
        "theorem_part": theorem_part,
        "theorem_part_with_answer": theorem_part_with_answer,
        "formal_answer": answer_content,
        "answer_type": answer_type,
        "is_formatted": "True" if is_formatted else "False",
        
    }
    return new_entry


def process_file(input_path: Path, output_path: Path):
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    output_data = []
    for entry in data:
        if entry.get('is_formalized') == 'True':
            processed = process_entry(entry) 
            if processed is not None:
                output_data.append(processed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

# TODO: Remove
# def process_entry_for_lean_message(entry: dict) -> dict:
    
#     name = entry["name"]
#     header = entry["header"]
#     with_answer = entry.get("theorem_part_with_answer", "")
#     temp_dir = Path("Formalization/Temp")
#     temp_dir.mkdir(parents=True, exist_ok=True)

#     # Write with-answer version
#     file_with = temp_dir / f"{name}_with.lean"
#     text = replace_last_occurrence(f"{header}\n\n{with_answer}", 'by sorry\n', "by ")
#     file_with.write_text(text, encoding="utf-8")
#     msg_with = run_lean_extract_goal(file_with)


#     # Add messages back to entry
#     entry["message_with"] = msg_with
#     return entry


def process_entry_for_theorem_part_without_answer(entry: dict) -> dict:
    if entry.get("theorem_part_without_answer" , '') != '':
        return entry
    name = entry["name"]
    header = entry["header"]
    theorem_part = entry.get("theorem_part", "")
    answer_part, theorem_part = entry.get('answer_part'), entry.get('theorem_part')
    temp_dir = Path("Formalization/Temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Write with-answer version
    file_with = temp_dir / f"{name}_without.lean"
    theorem_part = replace_last_occurrence(theorem_part, 'by sorry', 'by \nprint_fol\nsorry')

    with open(str(file_with), 'w') as f:
        f.write(f"import utils.interactive\n{header}\n{answer_part}\n{theorem_part}")
    msg_without = run_print_fol(file_with)
    theorem_part_without_answer = f"theorem {extract_theorem_name(theorem_part)} : ∃ {name}_answer : {entry.get('answer_type')}, {msg_without}"
    if "Error during Lean execution." in msg_without:

        theorem_part_without_answer = f"Error during Lean execution"


    # Add messages back to entry
    entry["theorem_part_without_answer"] = theorem_part_without_answer
    return entry


def compute_theorem_part_without_answer(json_path: Path, num_workers: int = 64):
    # Load input

    if isinstance(json_path, str):
        json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Use threads instead of processes
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        updated_data = list(tqdm(executor.map(process_entry_for_theorem_part_without_answer, data), total=len(data)))

    # Save updated data
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)


def write_formalizations_to_files(formalized_json_path: Path, output_with_answer = Path("data/eval/constructivebench.jsonl") , get_constructive_theorem = False):

    with open(formalized_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    # Define output paths

    # Prepare output files
    with jsonlines.open(output_with_answer, mode="w") as writer_with:
        
        for entry in data:
            
            # Skip entries with missing required fields
            if not all([
                entry.get("name"),
                entry.get("problem"),
                entry.get("header"),
                entry.get("theorem_part_with_answer"),
                entry.get('answer_type')
            ]):
                continue

            # Write with answer
            writer_with.write({
                "name": entry["name"],
                "split": "valid",
                "informal_prefix": f'/--\n{entry["problem"]}\n-/',
                "formal_statement": strip_sorry_suffix(entry["theorem_part_with_answer"]),
                "header": entry["header"],
                'answer_type': entry.get('answer_type')
            })


    if get_constructive_theorem:     
        compute_theorem_part_without_answer(
            json_path=formalized_json_path
        )

def generate_formalization_with_candidate_answers(summary, data, output_dir):
    """
    Reads the summary JSON and the full dataset JSON, processes each entry to extract
    and format the theorem part with answer, and writes a .lean file per entry.
    """
    # Load summary and full data

    # Map full data entries by name
    data_map = {entry['name']: entry for entry in data}

    # Ensure output directory exists
    output = []

    for entry in summary:
        name = entry.get('name')
        if name not in data_map:
            continue
        data_entry = data_map[name]
        preamble = data_entry['header']
        theorem_part = data_entry["theorem_part"]
        answer_name = f'{name}_answer'
        answer_type = data_entry['answer_type']
        answer_content = entry['proposed_answer']
        
        theorem_part_with_answer = theorem_part.replace(answer_name, f"({answer_content} : {answer_type})")
        entry['lean_content_for_eval'] = f"{preamble}\n{strip_sorry_suffix(theorem_part_with_answer)}"
        output_entry = {
            "name" : name,
            "split": "valid",
            "informal_prefix": f"/--\n{data_map[name]['problem']}\n-/",
            'formal_statement': f"\n{strip_sorry_suffix(theorem_part_with_answer)}",
            'header' : preamble, 
            'answer_type': answer_type
            
        }
        output.append(output_entry)
    with jsonlines.open(output_dir, mode="w") as writer_with:
        for entry in output:
            writer_with.write(entry)

def strip_sorry_suffix(s: str) -> str:
    return re.sub(r':=\s*by\s+sorry\s*$', ':= by ', s)

def run_print_fol(lean_file: str) -> str:
    """
    Runs `lake env lean --message-json` on the given Lean file and
    extracts the first line of output that begins with '∀'.
    """
    lean_file = str(lean_file)
    try:
        # Split into project directory and file path
        parts = lean_file.split('/', 1)
        if len(parts) == 1:
            project_dir = "."
            file_dir = parts[0]
        else:
            project_dir, file_dir = parts

        # Invoke Lean within the Lake environment, requesting JSON messages
        result = subprocess.run(
            ["lake", "env", "lean",  file_dir],
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )

        result = result.stdout.replace("The '∑ x in s, f x' notation is deprecated: please use '∑ x ∈ s, f x' instead:\n", "")
        start = result.index("warning: declaration uses 'sorry'\n") + len("warning: declaration uses 'sorry'\n")
        end = result.index('\nTemp', 1)   
        return result[start:end]

    except subprocess.CalledProcessError as e:

        return "Error during Lean execution."
    
def formal_equivalence_checker(name, header, answer_type, answer_1, answer_2) -> dict:
    temp_dir = Path("Formalization/Temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Write with-answer version
    filename = temp_dir / f"{name}.lean"
    with open(filename, 'w') as f:
        f.write(f"import utils.interactive\n{header}\ntheorem check_equality : ({answer_1} : {answer_type}) = ({answer_2} : {answer_type}) := by \ntry_solvers\nsorry")
    # Run the command in the specified working directory

    project_dir = str(filename).split('/')[0]
    file_dir = '/'.join(str(filename).split('/')[1:])

    result = subprocess.run(
        ["lake", "env", "lean", file_dir],  # Just check the file
        cwd=project_dir,
        capture_output=True,
        text=True,
    )

    result = result.stdout.strip()
    # match = re.search(r"error: unsolved goals\n(.+)", result, flags=re.DOTALL)
    if "True" in result:
        return "True"
    return "False"

def symbolic_prover(name, header, statement, dataset) -> str:
    temp_dir = Path("Formalization/Temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    safe_dataset = dataset.replace("/", "_")
    filename = temp_dir / f"{safe_dataset}_{name}.lean"

    with open(filename, 'w') as f:
        f.write(f"import utils.interactive\n{header}\n{statement}\ntry_solvers\nsorry")

    project_dir = str(filename).split('/')[0]
    file_dir = '/'.join(str(filename).split('/')[1:])

    result = subprocess.run(
        ["lake", "env", "lean", file_dir],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )

    result = result.stdout.strip()
    return "True" if "True" in result else "False"

def is_legal_answer(theorem_part, answer_type, proposed_answer):
    if answer_type == 'Prop':
        if proposed_answer.strip() not in {"True", "False"}:
            return False
    if len(answer_type) > 4 and "Prop" in answer_type:
        if "True" not in proposed_answer and "False" not in proposed_answer:
            return False
    if len(proposed_answer) > 20 and proposed_answer in theorem_part:  # Prevent problem statement echo
        return False
    return True

# Function to generate the legal check results file
def write_legal_check_results(output_dir: str, results: list, formal_entries: dict):
    legal_results_path = Path(output_dir) / "legal_check_results.jsonl"
    legal_results = []
    formal_entries = {i.get("name"):i for i in formal_entries}
    for entry in results:
        name = entry.get("name")
        proposed_answer = entry.get("proposed_answer")
        answer_type = entry.get("answer_type") or formal_entries.get(name, {}).get("answer_type", "")
        theorem_part = formal_entries.get(name, {}).get("theorem_part", "")
        if name and proposed_answer and answer_type:
            is_legal = str(is_legal_answer(theorem_part, answer_type, proposed_answer))
            legal_results.append({
                "name": name,
                "answer_type": answer_type,
                "proposed_answer": proposed_answer,
                "is_legal": is_legal
            })

    with open(legal_results_path, "w", encoding="utf-8") as f:
        for result in legal_results:
            f.write(json.dumps(result) + "\n")

    return legal_results

def auto_set_cuda_visible_devices():
    try:
        # Count the number of GPUs
        output = subprocess.check_output(["nvidia-smi", "-L"], encoding="utf-8")
        num_gpus = output.count("GPU ")
        # Set CUDA_VISIBLE_DEVICES to all GPU indices
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
        print(f"Set CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")
    except Exception as e:
        print(f"Could not set CUDA_VISIBLE_DEVICES automatically: {e}")