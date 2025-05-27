import torch
import os
import shutil
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from trl import SFTTrainer, DPOTrainer, PPOTrainer, PPOConfig

# ---------------- Step 1: Define Checkpoint Paths ----------------
slurm_job_id = os.getenv("SLURM_JOB_ID", "default_job")
user = os.getenv("USER", "default_user")

# Training checkpoint directory (MooseFS storage, auto-deletes in a week)
checkpoint_dir = f"/checkpoint/{user}/{slurm_job_id}"

# Final model directory (inside checkpoint storage to avoid home directory quota issues)
final_model_dir = f"{checkpoint_dir}/final_model"

# Create checkpoint directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

# ---------------- Step 2: Load Dataset ----------------
dataset_path = "data/finetune/data.jsonl"
dataset = load_dataset("json", data_files={"train": dataset_path})

# ---------------- Step 3: Load Model and Tokenizer ----------------
# model_path = "/model-weights/DeepSeek-R1-Distill-Qwen-7B"
model_path = "/model-weights/DeepSeek-R1-Distill-Qwen-14B"
# model_path = "/model-weights/internlm2-math-plus-7b"
# ---------------- Step 3: Load Model and Tokenizer ----------------
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          trust_remote_code=True)


model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, 
    device_map="auto",
    trust_remote_code=True
    
)
# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Explicitly add PAD token
    model.resize_token_embeddings(len(tokenizer))  # Adjust model's token embedding size


# ---------------- Step 4: Preprocess Dataset ----------------
def preprocess_function(examples):
    texts = ["\n".join(msg["content"] for msg in msg_list) for msg_list in examples["messages"]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=1024)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# ---------------- Step 5: Define Training Functions ----------------
def train_sft():
    """Supervised Fine-Tuning (SFT)"""
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,  # Save checkpoints in MooseFS storage
        per_device_train_batch_size=2,  # Optimized batch size for A40
        gradient_accumulation_steps=8,  # Adjusted for memory balance
        num_train_epochs=3,  # Consider 5 epochs for small dataset
        save_strategy="epoch",  # Save checkpoint after every epoch
        save_total_limit=3,  # Keep only the 3 latest checkpoints
        logging_dir=f"{checkpoint_dir}/logs",  # Logs stored in MooseFS
        learning_rate=2e-5,  # More stable learning rate
        warmup_steps=100,
        weight_decay=0.01,
        report_to="none",  # Disable wandb logging
        bf16=True if torch.cuda.is_bf16_supported() else False,  # Use bf16 if supported
        fp16=False,  # Avoid FP16 instability
        optim="adamw_bnb_8bit",  # Memory-efficient optimizer
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_datasets["train"],
        args=training_args,
        tokenizer=tokenizer,
    )
    latest_checkpoint = find_latest_checkpoint()

    while True:
        try:
            if latest_checkpoint:
                print(f"Resuming training from checkpoint: {latest_checkpoint}")
                trainer.train(resume_from_checkpoint=latest_checkpoint)
            else:
                print("No checkpoint found. Starting training from scratch.")
                trainer.train()
            break  # Exit loop if training completes successfully
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):

                print("CUDA Out of Memory detected. Clearing cache and retrying from last checkpoint...")
                torch.cuda.empty_cache()
                latest_checkpoint = find_latest_checkpoint()  # Update checkpoint
            else:
                raise e  # If it's another error, don't retry; raise it


def train_dpo():
    """Direct Preference Optimization (DPO)"""
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=3,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False,
        logging_dir=f"{checkpoint_dir}/logs",
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
    )

    latest_checkpoint = find_latest_checkpoint()
    
    if latest_checkpoint:
        dpo_trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        dpo_trainer.train()

    save_best_checkpoint()


def train_ppo():
    """Proximal Policy Optimization (PPO)"""
    ppo_config = PPOConfig(
        model_name=model_path,
        learning_rate=1.41e-5,
        log_with="wandb",
    )

    ppo_trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
    )

    for batch in ppo_trainer.dataloader:
        query_tensors = batch["input_ids"]
        response_tensors = model.generate(query_tensors, max_length=1024)
        rewards = torch.tensor([1.0] * len(response_tensors))

        ppo_trainer.step(query_tensors, response_tensors, rewards)

    save_best_checkpoint()


# ---------------- Step 6: Utility Functions ----------------
def find_latest_checkpoint():
    """Finds the latest available checkpoint in the checkpoint directory"""
    if os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            return os.path.join(checkpoint_dir, latest_checkpoint)
    return None


def save_best_checkpoint():
    """Saves the best checkpoint and final model in the checkpoint directory"""
    best_model_path = f"{checkpoint_dir}/checkpoint-best"
    
    # Ensure the final model directory exists
    os.makedirs(final_model_dir, exist_ok=True)

    if os.path.exists(best_model_path):
        shutil.copytree(best_model_path, final_model_dir, dirs_exist_ok=True)
        print(f"Best model saved to {final_model_dir}")
    else:
        print(f"No best model found at {best_model_path}")

    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"Final model saved to {final_model_dir}")

    # Print the final model directory for reference
    print(f"\n✅ Final fine-tuned model is stored at: {final_model_dir}")


# ---------------- Step 7: Main Execution ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Tuning Script for DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--method", type=str, choices=["sft", "dpo", "ppo"], default="sft", help="Choose fine-tuning method")
    args = parser.parse_args()

    if args.method == "sft":
        train_sft()
    elif args.method == "dpo":
        train_dpo()
    elif args.method == "ppo":
        train_ppo()
        
        
# job id 15138416

# 15542359 for a100