## Enumerate–Conjecture–Prove: Formally Solving Answer-Construction Problem in Math Competitions
We provide a brief introduction to set up the code for Enumerate-Conjecture-Prove (ECP) paper.

## Requirement
- Python 3.10
- CUDA >= 11.8
- Git LFS
- Lean Proof Assistant
- (Optional) Chrome and Chrome Drive

## Environment Setup
First, ensure git-lfs is correctly initialized. (Skip it if you have already set up.)
```
git lfs install
```
Then, run the following command and follow instruction to setup Lean. (Skip it if you have already set up.)
```
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source ~/.bashrc
```
Check if it is correctly installed by 
```
lean --version
```

To pull the repository, run
```
git clone --recursive https://github.com/AnonymousAuthors91283711/ECP
cd ECP
```

To set up the required Python environment, we recommend using venv as follows.
```
python -m venv imosolver
pip install -r requirements.txt
```
Build Lean environment for both newest version and v4.9.0-rc1 for prover model (Which takes around 30 minutes):
```
cd Formalization
lake update
lake build Main
cd ..
cd prover/mathlib4
lake build
cd ../..
```
Finally, set up your LLM APIs in your shell file. Alternatively, you can manually set them in appl.yaml
```
echo 'export OPENAI_API_KEY="your_openai_key_here"' >> ~/.bashrc
echo 'export DEEPSEEK_API_KEY="your_deepseek_key_here"' >> ~/.bashrc
```

## File Structure
```
ECP/
├── prover/                         # Adapted from Goedel-Prover which contains utilities for running prover models. 
├── src/
│   └── ecp/                        
│       └── agent.py                # Multi-agents framework for Enumerate, Conjecture, Prove (ECP)
│       └── main.py                 # Main entry point for experiments.
        └── utils.py                # Utilities
│   └── scripts/                    # Other scripts
│       ├── dataset/
│       ├── deploy/
│       ├── finetune/
│       ├── lean/
│       └── trace/

```




<!-- To download and extract Google Chrome
```bash
wget https://dl.google.com/linux/chrome/deb/pool/main/g/google-chrome-stable/google-chrome-stable_134.0.6998.88-1_amd64.deb
mkdir -p ~/chrome
dpkg -x google-chrome-stable_134.0.6998.88-1_amd64.deb ~/chrome
```
To download and extract Chrome driver
```bash
wget https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/134.0.6998.88/linux64/chromedriver-linux64.zip
unzip chromedriver-linux64.zip -d ~/chrome
chmod +x ~/chrome/chromedriver-linux64/chromedriver
```
Update PATH
```bash
echo 'export PATH=$HOME/chrome/opt/google/chrome:$HOME/chrome/chromedriver-linux64:$PATH' >> ~/.bashrc
source ~/.bashrc
```
Verify Installation
```bash
~/chrome/opt/google/chrome/google-chrome --version
chromedriver --version
``` -->
## Datasets

We provide ConstructiveBench dataset described in the paper:

### ConstructiveBench

Located at: `data/dataset/constructivebench.json`  
This dataset contains curated Olympiad-style problems with metadata and aligned Lean formalizations.  
Each entry includes:
- Problem statement
- Category (e.g., Algebra, Combinatorics)
- Formal answer in Lean
- Full formal theorem
- Answer-construction alignment parts (header, answer, theorem with and without answer)

Example:
```json
{
  "name": "IMO2011SLC4",
  "category": "Combinatorics",
  "source": "IMO/2011",
  "problem": "...",
  "answer": "The greatest such number k is 3",
  "formalization": "...",
  "...": "..."
}
```
### PutnamBench (Answer-Construction subset)

Located at: data/dataset/putnam.json
This contains a selected subset of answer-construction problems from the PutnamBench dataset, specifically chosen to test the ECP pipeline's generalization to diverse university-level problems.

## Example Runs

The file `src/ecp/main.py` provides a unified interface for three pipelines:

- `answer_gen`: full ECP pipeline (enumerate → conjecture → verify)
- `autoformalize`: generate formalizations from informal problems and answers
- `proof_gen`: use a formal prover to generate complete Lean proofs

### 1. Input Dataset

The `--problem_path` argument specifies the dataset to process. The main option is:
- `constructivebench` (recommended). For testing, use `test` for one case of Constructivebench.

### 2. Choosing the Pipeline

You can set the `--mode` flag to one of the following:

- `answer_gen`: Run the full ECP pipeline. This mode first enumerates a Python program, then conjectures an answer, and finally verifies it.
- `autoformalize`: Translate informal problems and answers into fully formatted Lean code.
- `proof_gen`: Run proof generation using a formal prover like Goedel-Prover.

### 3. Key Flags

- `--enable_enumerator`: If `True`, runs the full ECP pipeline (enumerator + conjecturer). If `False`, skips program enumeration and uses only LLM-based conjecturing (like Chain-of-Thought).  
  - **True = ECP**, **False = CoT**
- `--problem_name`: `"all"` (default) to process all entries; or a comma-separated list of problem names to run.

---

### Example Commands
#### A. Run autoformalization (Optional)

```bash
python src/ecp/main.py \
    --mode autoformalize \
    --problem_path constructivebench
```
#### B. Run the answer-generation for ECP pipeline (Enumerate → Conjecture)

```bash
python src/ecp/main.py \
    --mode answer_gen \
    --problem_path constructivebench \
    --enable_enumerator true
```

> Output will be stored in:  
> `output/data/dataset/constructivebench.json/deepseek-chat-code/`
(To run the CoT-baseline, simply set --enable_enumerator false)


#### C. Run the proof-generation for ECP pipeline (Prove)

After generating formalizations and conjectures (e.g. via answer_gen), run:

```bash
python src/ecp/main.py \
    --mode proof_gen \
    --problem_path constructivebench
```

> Proof generation uses Goedel-Prover by default. You can set deepseek-ai/DeepSeek-Prover-V2-7B, AI-MO/Kimina-Prover-Preview-Distill-7B for more experiments. 

---

### Default Models & Parameters

- `--enumerator_model`: `deepseek-chat`
- `--conjecturer_model`: `deepseek-chat`
- `--prover_model`: `Goedel-LM/Goedel-Prover-SFT`
- `--max_tokens`: `500`
- `--timeout`: `60` (in seconds)
- `--pass_at_n`: `32` (for Pass@n metric in proof generation)
- `--gpu`: `1` (for number of GPUs in proof generation)
- `--use_embedding_search`: `False` (for using embedding-based lean retrieval. Set to `True` only if you have GPU resource.)

You can override these options as needed. Check `src/ecp/main.py` for the full list of arguments.

## Summarize
After experiment, the answer / proving accuracies for dataset (constructivebench, putnam, test) can be summarized by 
```
python summarize_result.py --base <dataset>
```

## Reference

This work builds on the [Goedel-Prover frontend model](https://github.com/Goedel-LM/Goedel-Prover) repository.