INPUT_PATH=datasets/minif2f.jsonl
MODEL_PATH=Goedel-LM/Goedel-Prover-SFT
OUTPUT_DIR=results/minif2f/Godel-Prover-SFT
SPLIT=test
N=32
CPU=128 #32
GPU=2
FIELD=complete
while getopts ":i:m:o:s:n:c:g:" opt; do
  case $opt in
    i) INPUT_PATH="$OPTARG"
    ;;
    m) MODEL_PATH="$OPTARG"
    ;;
    o) OUTPUT_DIR="$OPTARG"
    ;;
    s) SPLIT="$OPTARG"
    ;;
    n) N="$OPTARG"
    ;;
    c) CPU="$OPTARG"
    ;;
    g) GPU="$OPTARG"
    ;;
  esac
done
# python -m eval.step1_inference --input_path ${INPUT_PATH}  --model_path ${MODEL_PATH}  --output_dir $OUTPUT_DIR --split $SPLIT --n $N --gpu $GPU

INPUT_FILE=${OUTPUT_DIR}/to_inference_codes.json
COMPILE_OUTPUT_PATH=${OUTPUT_DIR}/code_compilation.json
python -m eval.step2_compile --input_path $INPUT_FILE --output_path $COMPILE_OUTPUT_PATH --cpu $CPU


SUMMARIZE_OUTPUT_PATH=${OUTPUT_DIR}/compilation_summarize.json
python -m eval.step3_summarize_compile --input_path $COMPILE_OUTPUT_PATH --output_path $SUMMARIZE_OUTPUT_PATH --field ${FIELD}