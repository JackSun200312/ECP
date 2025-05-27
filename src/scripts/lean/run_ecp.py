import argparse
from src.ecp.interactive import run_coder_interactive, run_coder_conjecturer_interactive

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Coder + Conjecturer on a given state string.")
    parser.add_argument("--state", required=True, help="The input state string")
    parser.add_argument("--function", required=True, help="The function to be called")

    args = parser.parse_args()
    if args.function == 'conjecturer':
        result = run_coder_conjecturer_interactive(
            args.state,
            max_tokens=1000
        )

        # print("Program:\n", result["program"], flush=True)
        print("\nEnumerated:\n", result["enumerated_answers"], flush=True)
        print("\nConjecture:\n", result["proposed_answer"], flush=True)
        
    if args.function == 'coder':
        result = run_coder_interactive(
            args.state,
            max_tokens=1000
        )

        print("Program:\n", result["program"], flush=True)
        print("\nEnumerated:\n", result["enumerated_answers"], flush=True)
        # print('hello', flush=True)