import subprocess

def run_lean_and_capture_state(lean_file: str) -> str:
    try:
        project_dir = lean_file.split('/')[0]
        file_dir = '/'.join(lean_file.split('/')[1:])
        result = subprocess.run(
            ["lake","env","lean", file_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=project_dir,
            text=True,
            check=True
        )
        # Print the entire output (including logs from `eliminate_quantifier`)
        print("Lean Output:\n", result.stdout)

        # Extract the part with your proof state
        # You can modify this depending on what logInfo prints
        for line in result.stdout.splitlines():
            if "Proof state" in line or "⊢" in line:
                return line.strip()

        return "Proof state not found in output."

    except subprocess.CalledProcessError as e:
        print("Lean execution failed:\n", e.stdout)
        return "Error during Lean execution."

# Example usage
if __name__ == "__main__":
    state = run_lean_and_capture_state("Formalization/test/test.lean")
    print("\nExtracted State:\n", state)
