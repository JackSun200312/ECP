from lean_dojo import *
import pickle

def extract_mathlib_definitions(repo_url, commit_hash, output_file):
    repo = LeanGitRepo(repo_url, commit_hash)
    traced_repo = trace(repo)
    definitions = {}


    traced_files = [i for i in traced_repo.traced_files if str(i.lean_file.path)]

    print(len(traced_files))
    for traced_file in traced_files:
       
        try:
            for definition in traced_file.get_premise_definitions():
                full_name = definition['full_name']
                code = definition['code']
                # kind = definition['kind']
                definitions[full_name] = code
        except :
            print(f'{traced_file} has error')
    
    with open(output_file, 'wb') as f:
        pickle.dump(definitions, f)
    print(f"Saved {len(definitions)} definitions to {output_file}")

def load_definitions(input_file):
    with open(input_file, 'rb') as f:
        definitions = pickle.load(f)
    return definitions

if __name__ == "__main__":
    repo_url = "https://github.com/leanprover/lean4"
    commit_hash = "v4.18.0-rc1"
    output_file = "v4_18_lean_definitions.pkl"
    
    extract_mathlib_definitions(repo_url, commit_hash, output_file)
    # extract_mathlib_definitions("https://github.com/yangky11/lean4-example", "7f7d71379312f9e8098ce7341d8ea1cd449ec0e9", output_file)
    
    # Example usage of retrieving definitions
    definitions = load_definitions(output_file)
    for name, code in list(definitions.items())[:5]:  # Print first 5 definitions
        print(f"{name}: {code}\n")


