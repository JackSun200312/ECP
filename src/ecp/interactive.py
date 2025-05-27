import tempfile
from typing import Optional
from src.ecp.agent import Coder, Conjecturer, Autoformalizer, EquivalenceChecker, FormalConjecturer


def run_coder_interactive(problem: str,
               model: str = 'gpt-4o-mini',
               max_tokens: int = 1000,
               max_attempt: int = 1,
               timeout: int = 60) -> dict:
    """
    Runs only the Coder: generates a program and enumerated answers for the given problem.

    Returns a dict with:
      - 'program': the generated code
      - 'enumerated_answers': the output from running the code
      - 'history': LLM interaction history
    """
    coder = Coder(model=model, max_tokens=max_tokens)
    program, enumerated = coder.write_program_loop(problem, max_attempt=max_attempt, timeout=timeout)
    return {
        'program': program,
        'enumerated_answers': enumerated,
        'history': coder.history,
    }


def run_coder_conjecturer_interactive(problem: str,
                                      enumerator_model: str = 'gpt-4o',
                                      conjecturer_model: str = 'gpt-4o',
                                      max_tokens: int = 1000,
                                      coder_max_attempt: int = 1,
                                      timeout: int = 60) -> dict:
    """
    Runs the full coder + conjecturer pipeline on a single problem.

    Returns a dict with:
      - 'program': generated code
      - 'enumerated_answers': output of the code
      - 'program_history': coder LLM history
      - 'proposed_answer': conjectured answer
      - 'conjecturer_history': conjecturer LLM history
    """
    coder = Coder(model=enumerator_model, max_tokens=max_tokens)
    program, enumerated = coder.write_program_loop(problem, max_attempt=coder_max_attempt, timeout=timeout)

    conjecturer = Conjecturer(model=conjecturer_model, max_tokens=max_tokens)
    proposed = conjecturer.conjecture_answer(problem, enumerated)

    return {
        'program': program,
        'enumerated_answers': enumerated,
        'program_history': coder.history,
        'proposed_answer': proposed,
        'conjecturer_history': conjecturer.history,
    }


def run_equivalence_interactive(problem: str,
                                 ground_truth: str,
                                 proposed: str,
                                 model: str = 'gpt-4o',
                                 max_tokens: int = 1000) -> dict:
    """
    Checks equivalence between a proposed answer and ground truth.

    Returns a dict with:
      - 'is_equivalent': "True" or "False"
      - 'explanation': LLM's full response
      - 'history': LLM interaction history
    """
    checker = EquivalenceChecker(model=model, max_tokens=max_tokens)
    is_eq = checker.check_equivalence(problem, ground_truth, proposed)
    return {
        'is_equivalent': is_eq,
        'explanation': checker.explanation,
        'history': checker.history,
    }


def run_autoformalizer_interactive(problem: str,                                   
                                   answer: Optional[str] = None,
                                   model: str = 'gpt-4o',
                                   judge_model: str = 'gpt-4o',
                                   max_tokens: int = 1000,
                                   enable_judge: bool = True,
                                   max_attempt: int = 5) -> dict:
    """
    Autoformalizes a single problem (and answer) into Lean.

    Returns a dict with:
      - 'formalization': generated Lean code
      - 'lean_success': bool, whether it compiles
      - 'llm_success': bool, whether LLM judge accepted (when enabled)
      - 'history': LLM interaction history
    """
    # Create a temporary file for the Lean output
    tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.lean', delete=False)
    filename = tmp.name
    tmp.close()

    auto = Autoformalizer(
        filename=filename,
        model=model,
        judge_model=judge_model,
        max_tokens=max_tokens,
        enable_judge=enable_judge,
    )

    formalization, lean_ok, llm_ok = auto.autoformalize_loop(
        problem,
        answer,
        max_attempt=max_attempt,
    )

    return {
        'formalization': formalization,
        'lean_success': lean_ok,
        'llm_success': llm_ok,
        'history': auto.history,
    }


def run_formal_coder_conjecturer_interactive(formal_statement: str,
                                             expected_answer_type: str,
                                             use_coder: bool = True,
                                             enumerator_model: str = 'gpt-4o',
                                             conjecturer_model: str = 'gpt-4o',
                                             max_tokens: int = 1000,
                                             coder_max_attempt: int = 1,
                                             max_attempt: int = 5,
                                             timeout: int = 60) -> dict:
    """
    For a given formal Lean statement and expected answer type, optionally run the enumerator,
    then conjecture and insert the final answer into the Lean code.

    Returns a dict with:
      - 'proposed_answer': the Lean-syntax answer
      - 'success': bool, whether Lean compilation passed
      - 'formalization': full Lean file with answer inserted
      - 'history': LLM interaction history
    """
    tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.lean', delete=False)
    filename = tmp.name
    tmp.close()

    if use_coder:
        coder = Coder(model=enumerator_model, max_tokens=max_tokens)
        _, enumerated = coder.write_program_loop(formal_statement, max_attempt=coder_max_attempt, timeout=timeout)
    else:
        enumerated = ""

    conjecturer = FormalConjecturer(filename=filename, model=conjecturer_model, max_tokens=max_tokens)
    proposed, success = conjecturer.conjecture_answer_loop(formal_statement, enumerated, expected_answer_type, max_attempt=max_attempt)
    with open(filename, 'r') as f:
        full = f.read()

    return {
        'proposed_answer': proposed,
        'success': success,
        'formalization': full,
        'history': conjecturer.history,
    }
