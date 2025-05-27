import requests
import json

# Define API URL
API_URL = "http://127.0.0.1:30000/v1/chat/completions"

# # Define the system prompt and query
# system_prompt = (
# """I have a difficult high-school competition-level math problem. Your task is to formalize the problem statement in Lean 4 (v4.18.0-rc1), and leave the proof empty by "sorry" keyword.\n\nYour should follow the guidelines:\n1. State "import Mathlib" at the beginning not import anything else. Then, clearly state the opened namespaces.\n2. If the problem requires construction of an answer, write in this format: abbrev <name>_solution : <type> := <solution expressed in lean>, then mention it in the theorem statement. In case the answer is a set, write it as {answer 1,... answer n} if finite, or {answer : type | property}. You should use bidirectional "iff" connective to indicate the uniqueness of answer set. \n3. Use the theorem name I gave to you, and your theorem should be: theorem <name> <your formalization here> := by sorry\n4. In lean 4.18.0-rc1, most namespaces begin with capital letter (e.g. nat -> Nat), and the lambda calculas use => instead of , (e.g. λ x, f x should become λ x => f x). \n5. The output should be at most 500 tokens. Do not repeat the natural language statement in your lean file, i.e. you do not need to include /-- <natural language description> -/ in your formalization.\n6. Never try to prove the theorem you wrote! Leave it by sorry keyword."""
# )

# query = """Problem Statement: Problem Statement: What is the smallest positive integer $x$ for which $x^2+x+41$ is not a prime?. Answer: 40. Problem Name: omnimath508"""

system_prompt = ("Write a program that enumerates the answer for following problem")

query = "Write a program that enumerates the zero for following function: f(x) = x^3-2x+1=0. Round to 2 decimals. Print the solutions."


# Define the request payload
payload = {
    "model": "local_model",  # Adjust model name as necessary
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ],
    "temperature": 0.0,
    "max_tokens": 1000
}

# Send the request
response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

# Print the response
print(response.json())
