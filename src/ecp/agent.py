import os

from appl import ppl, gen, SystemMessage
# from src.nonempty_sols import non_empty
from src.ecp.utils import *
from src.ecp.retrieve_definition import *
from typing import Tuple, List, Optional

import datetime

class TimeoutException(Exception):
    pass

# We keep the Agent class for reference but do not use its __init__ in subclasses.
class Agent:
    """Base class: keeps prompts, model info and an interaction log."""
    def __init__(self, model: str = "gpt-4o", max_tokens: int = 500):
        self.model = model
        self.max_tokens = max_tokens
        self.history =  ""
        


class Coder(Agent):
    """Builds and refines a Python enumeration program."""

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 500, enable_judge: bool = False, filename = "Temp/temp.py"):
        super().__init__(model, max_tokens)
        self.enable_judge = enable_judge
        with open("data/prompts/coder.txt", encoding="utf-8") as f:
            self.system_prompt = f.read()
        with open("data/prompts/coder_refiner.txt", encoding="utf-8") as f:
            self.system_prompt_refiner = f.read()
        self.filename = filename


    @ppl
    def write_program(self, problem: str) -> str:
        SystemMessage(self.system_prompt)
        problem
        reply = str(gen(self.model))
        self.history += f"[write_program]\n{reply}\n"
        return parse_program_in_delimiter(reply)

    @ppl
    def refine_program(self, problem: str, program: str, enumerated_answers: str) -> Tuple[bool, str]:
        SystemMessage(self.system_prompt_refiner)
        prompt = (
            f"Problem: {problem}\nProgram:\n{program}\n"
            f"Enumerated Answers:\n{enumerated_answers}"
        )
        prompt
        reply = str(gen(self.model, max_tokens=self.max_tokens))
        self.history += f"[refine_program]\n{reply}\n"
        accepted, new_program = parse_program_and_answer_in_delimiter(reply)
        return (accepted == "True"), new_program

    def write_program_loop(
        self, problem: str, max_attempt: int, timeout: int
    ) -> Tuple[str, str, List[dict]]:
        """
        Returns:
          program        : final program text
          program_output     : its last program_output output
          attempts       : list of dicts with keys
                           {compile_ok, judge_ok, program, program_output}
        """
        attempts = []

        # initial
        program = self.write_program(problem)
        ok, program_output = run_python_program(program, timeout, self.filename)
        judge_ok = False if self.enable_judge else True
        attempts.append({
            "compile_ok": ok,
            "judge_ok": judge_ok,
            "program": program,
            "program_output": program_output,
        })
        best_program, best_enum = (program, program_output) if ok else (None, "")

        for _ in range(1, max_attempt):
            if ok and self.enable_judge:
                refiner_feedback, candidate = self.refine_program(problem, program, program_output)
                if not refiner_feedback:
                    program = candidate
                else:
                    judge_ok = refiner_feedback
            elif ok and not self.enable_judge:
                # no judge desired → stop
                break
            else:
                # compile failed → just refine program
                _, candidate = self.refine_program(problem, program, program_output)
                program = candidate

            ok, program_output = run_python_program(program, timeout, self.filename)
            if ok and best_program is None:
                best_program, best_enum = program, program_output

            attempts.append({
                "compile_ok": ok,
                "judge_ok": judge_ok,
                "program": program,
                "program_output": program_output,
            })

            if ok and (judge_ok or not self.enable_judge):
                break

        # choose final
        if ok and (judge_ok):
            return program, program_output, attempts
        if best_program is not None:
            return best_program, best_enum, attempts
        return program, program_output, attempts


class Conjecturer(Agent):
    """Proposes a textual answer for the problem."""
    def __init__(self, model: str = "gpt-4o", max_tokens: int = 500):
        super().__init__(model, max_tokens)
        with open("data/prompts/conjecturer.txt", encoding="utf-8") as f:
            self.system_prompt = f.read()
        with open("data/prompts/conjecturer_without_coder.txt", encoding="utf-8") as f:
            self.system_prompt_without_coder = f.read()

    @ppl
    def conjecture_answer(
        self, problem: str, enumerated_answers: Optional[str] = None
    ) -> str:
        system_prompt = (
            self.system_prompt
            if enumerated_answers is not None
            else self.system_prompt_without_coder
        )
        SystemMessage(system_prompt)
        prompt = (
            f"Problem: {problem}\nEnumerated Answers: {enumerated_answers}"
            if enumerated_answers is not None
            else f"Problem: {problem}"
        )
        prompt
        reply = str(gen(self.model))
        self.history += f"[conjecture_answer] {prompt}\n{reply}"
        return parse_answer_in_delimiter(reply)


class Autoformalizer(Agent):
    """
    Turn a natural‑language problem (plus optional answer) into Lean.

    Public methods keep their original names; only the parameter lists
    change to receive the `problem` string directly.
    """

    def __init__(
        self,
        filename: str,
        model: str = "gpt-4o",
        judge_model: str = "gpt-4o",
        max_tokens: int = 500,
        enable_judge: bool = True,
        use_embedding_search = True,
    ):
        super().__init__(model, max_tokens)
        self.judge_model = judge_model
        self.enable_judge = enable_judge
        self.filename = filename
        self.use_embedding_search = use_embedding_search

        with open("data/prompts/autoformalizer.txt", encoding="utf-8") as f:
            self.system_prompt = f.read()
        with open("data/prompts/autoformalizer_refiner.txt", encoding="utf-8") as f:
            self.system_prompt_refiner = f.read()
        with open("data/prompts/autoformalizer_judge.txt", encoding="utf-8") as f:
            self.system_prompt_feedback = f.read()
        with open("data/prompts/autoformalizer_answer.txt", encoding="utf-8") as f:
            self.system_prompt_answer = f.read()
        with open("data/prompts/autoformalizer_answer_refiner.txt", encoding="utf-8") as f:
            self.system_prompt_answer_refiner = f.read()

        os.makedirs(os.path.dirname(filename), exist_ok=True)


    
    @ppl
    def autoformalize(
        self,
        problem: str,
        answer: Optional[str] = None,
        problem_name: str = "statement",
    ) -> Tuple[str, dict]:
        """
        Generate initial Lean code + metadata for logging.
        """
        if answer is None:
            prompt = (
                f"Problem Statement: {problem}. "
                f"Problem Name: {problem_name}"
            )
        else:
            prompt = (
                f"Problem Statement: {problem}. "
                f"Answer: {answer}. "
                f"Problem Name: {problem_name}"
            )

        SystemMessage(self.system_prompt)
        prompt
        reply = str(gen(self.model))
        self.history += f"[autoformalize] {prompt}\n{reply}\n"

        lean_code = parse_lean_in_delimiter(reply)
        lean_code = lean_code.replace('\nabbrev', '\nnoncomputable abbrev')
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": self.model,
            "prompt": prompt,
            "raw_reply": reply,
            "lean_code": lean_code,
            "attempt_type": "initial",
        }
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(lean_code)
        return lean_code, metadata

    def execute_formalization(self, check_answer: bool = True) -> Tuple[bool, str, dict]:
        parsed_err = run_lean(self.filename, check_answer=check_answer, use_embedding_search=self.use_embedding_search)
        error_free = parsed_err == ""

        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "lean_compilation_success": error_free,
            "lean_error_message": parsed_err,
            "attempt_type": "lean_compile",
        }
        return error_free, parsed_err, metadata

    @ppl
    def get_feedback(self, problem_prompt: str, lean_code: str) -> Tuple[bool, str, dict]:
        SystemMessage(self.system_prompt_feedback)
        prompt = f"{problem_prompt}\nFormalization: {lean_code}"
        prompt
        reply = str(gen(self.judge_model, max_tokens=self.max_tokens))
        self.history += f"[judge] {prompt}\n{reply}\n"

        judge_passed = parse_answer_in_delimiter(reply) == "True"
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": self.judge_model,
            "prompt": prompt,
            "raw_reply": reply,
            "judgement_passed": judge_passed,
            "attempt_type": "judge",
        }
        return judge_passed, reply, metadata

    @ppl
    def refine_formalization(self, problem_prompt: str, lean_code: str, error: str, feedback: str) -> Tuple[str, dict]:
        SystemMessage(self.system_prompt_refiner)
        prompt = (
            f"Problem Statement: {problem_prompt}\n"
            f"Current Formalization: {lean_code}\n"
            f"Lean Error Message: {error}\n"
            f"LLM Feedback: {feedback}"
        )
        prompt
        reply = str(gen(self.model))
        self.history += f"[refine] {prompt}\n{reply}\n"

        new_code = parse_lean_in_delimiter(reply)
        new_code = new_code.replace('\nabbrev', '\nnoncomputable abbrev')
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": self.model,
            "prompt": prompt,
            "raw_reply": reply,
            "lean_code": new_code,
            "attempt_type": "refine",
        }

        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(new_code)
        return new_code, metadata

    
    def autoformalize_loop(
        self,
        problem: str,
        answer: Optional[str] = None,
        problem_name: str = "statement",
        max_attempt: int = 5,
    ) -> Tuple[str, bool, bool, List[dict]]:
        """
        Full loop: generate → compile → (optional) judge → refine.

        Returns
        -------
        lean_code           : final Lean text
        compiler_error_free : bool
        judge_passed        : bool
        attempts_log        : list of metadata dicts for each attempt
        """
        
        attempts_log = []
        
        # initial generation
        problem_prompt = (
            f"Problem Statement: {problem}. "
            f"Answer: {answer}. Problem Name: {problem_name}"
            if answer is not None
            else f"Problem Statement: {problem}. Problem Name: {problem_name}"
        )
        if not self.enable_judge:
            judge_ok = True
        compilable = False
        attempt = 0
        best_compiling = ''
        err = ''
        feedback = ''
        while attempt < max_attempt and (not compilable or not judge_ok):

            if attempt == 0:
                lean_code, autoformalize_meta = self.autoformalize(problem, answer, problem_name)
            else: 
                lean_code, autoformalize_meta = self.refine_formalization(
                    problem_prompt, lean_code, err, feedback
                )
            attempts_log.append(autoformalize_meta)
            compilable, err, lean_compile_meta = self.execute_formalization()
            attempts_log.append(lean_compile_meta)
            if compilable:
                best_compiling = lean_code
                if self.enable_judge:
                    judge_ok, feedback, judge_meta = self.get_feedback(problem_prompt, lean_code)
                    attempts_log.append(judge_meta)
                
            attempt += 1

        return lean_code, compilable, judge_ok, attempts_log

    
    def autoformalize_answer_loop(
        self, 
        formalization:str, 
        formal_statement: str,
        formal_answer_type: str, 
        informal_answer:str,
        max_attempt:int = 5
    ) -> Tuple[str, bool, bool]:
        """
        Full loop: generate → compile → (optional) judge → refine.

        Returns
        -------
        lean_code           : final Lean text
        compiler_error_free : bool
        judge_passed        : bool
        """
        # initial generation
        problem_prompt = (
            f"Formal Problem Statement: {formal_statement}. "
            f"Answer Type: {formal_answer_type} "
            f"Informal Answer: {informal_answer}"
            )
        
        proposed_answer = self.autoformalize_answer(
            formalization, formal_statement, formal_answer_type, informal_answer
        )
        ok, err = self.execute_formalization()


        for _ in range(1, max_attempt):
            if not ok:
                proposed_answer = self.refine_answer(
                    formalization, formal_statement, formal_answer_type, informal_answer,
                    proposed_answer, err
                )
                ok, err = self.execute_formalization()

        return proposed_answer, ok
        
class EquivalenceChecker(Agent):
    """Checks if proposed answer matches ground truth."""
    def __init__(self, model: str = "gpt-4o", max_tokens: int = 500):
        super().__init__(model, max_tokens)
        with open("data/prompts/equivalence_checker.txt", encoding="utf-8") as f:
            self.system_prompt = f.read()

    @ppl
    def check_equivalence(
        self, problem: str, ground_truth_answer: str, proposed_answer: str
    ) -> bool:
        SystemMessage(self.system_prompt)
        prompt = (
            f"Problem: {problem}\nGround Truth Answer: {ground_truth_answer}\n"
            f"Proposed Answer: {proposed_answer}"
        )
        prompt
        reply = str(gen(self.model))
        self.history += f"[check_equivalence] {prompt}\n{reply}"
        return parse_answer_in_delimiter(reply)

class FormalConjecturer(Agent):
    """Injects a Lean answer into an existing formal statement."""
    def __init__(self, filename: str, model: str = "gpt-4o", max_tokens: int = 500, use_embedding_search = True):
        super().__init__(model, max_tokens)
        self.filename = filename
        self.use_embedding_search = use_embedding_search
        with open("data/prompts/conjecturer_formal.txt", encoding="utf-8") as f:
            self.system_prompt = f.read()
        with open("data/prompts/conjecturer_formal_refiner.txt", encoding="utf-8") as f:
            self.system_prompt_refiner = f.read()

    @ppl
    def conjecture_answer(
        self, formal_statement: str, enumerated_answers: str, expected_answer_type: str
    ) -> str:
        SystemMessage(self.system_prompt)
        prompt = (
            f"Formal Problem Statement: {formal_statement}. "
            f"Enumerated answers: {enumerated_answers}. "
            f"Expected Answer Type: {expected_answer_type}"
        )
        prompt
        reply = str(gen(self.model))
        self.history += f"[conjecture_answer] {prompt}\n{reply}"
        answer = parse_lean_in_delimiter(reply)
        if "abbrev" in answer and ":=" in answer:
            idx = answer.index(":=")
            answer = answer[idx+2:]
        return answer 

    @ppl
    def refine_answer(
        self,
        formal_statement: str,
        current_answer: str,
        error: str,
        enumerated_answers: str,
        expected_answer_type: str,
    ) -> str:
        SystemMessage(self.system_prompt_refiner)
        prompt = (
            f"Formal Problem Statement: {formal_statement}. "
            f"Current Proposed Answer: {current_answer}. "
            f"Lean Error Message: {error}. "
            f"Enumerated Answers: {enumerated_answers}. "
            f"Expected Answer Type: {expected_answer_type}"
        )
        prompt
        reply = str(gen(self.model))
        self.history += f"[refine_answer] {prompt}\n{reply}"
        answer = parse_lean_in_delimiter(reply)
        if "abbrev" in answer and ":=" in answer:
            idx = answer.index(":=")
            answer = answer[idx+2:]
        return answer 
    def execute_formalization(self, answer) -> Tuple[bool, str, dict]:
        parsed_err = run_lean(self.filename, sanity_check = False,check_answer=False, use_embedding_search=self.use_embedding_search)
        parsed_err += sanity_check_for_conjectured_answer(answer)
        error_free = parsed_err == ""

        # metadata = {
        #     "timestamp": datetime.datetime.now().isoformat(),
        #     "lean_compilation_success": error_free,
        #     "lean_error_message": parsed_err,
        #     "attempt_type": "lean_compile",
        # }
        return error_free, parsed_err


    def conjecture_answer_loop(
        self,
        formal_statement: str,
        enumerated_answers: str,
        expected_answer_type: str,
        max_attempt: int = 5,
    ) -> Tuple[str, bool]:
        answer = self.conjecture_answer(
            formal_statement, enumerated_answers, expected_answer_type
        )

        code = insert_answer_comment(answer, formal_statement)
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(code)

        ok, err = self.execute_formalization(answer)

        for _ in range(1, max_attempt):
            if ok:
                break
            answer = self.refine_answer(
                formal_statement,
                answer,
                err,
                enumerated_answers,
                expected_answer_type,
            )
            code = insert_answer_comment(answer, formal_statement)
            with open(self.filename, "w", encoding="utf-8") as f:
                f.write(code)
            ok, err = self.execute_formalization(answer)

        return answer, ok
    
    
@ppl
def get_model_knowledge_cutoff(model: str = "deepseek-chat", max_tokens: int = 50) -> str:
    from appl import SystemMessage, gen
    SystemMessage("You are a helpful assistant.")
    prompt = "What is your knowledge cutoff date?"
    prompt
    reply = str(gen(model, max_tokens=max_tokens))
    return reply