from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
import os

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
PHASE = 3
BLOCK_SIZE = 512

# ============================================================
# SAFE MODEL LOADER
# ============================================================
def load_latest_checkpoint(folder):
    if not os.path.exists(folder):
        return None

    checkpoints = [d for d in os.listdir(folder) if d.startswith("checkpoint")]
    if not checkpoints:
        return None

    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return os.path.join(folder, checkpoints[-1])

def load_model_safely(folder):
    checkpoint = load_latest_checkpoint(folder)

    if checkpoint:
        print(f"🔄 Loading checkpoint: {checkpoint}")
        return GPT2LMHeadModel.from_pretrained(
            checkpoint,
            ignore_mismatched_sizes=True
        ), True
    else:
        print("🆕 Training from scratch")
        return get_model(), False
    

# ============================================================
# MODEL CREATION
# ============================================================
def get_model():
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=BLOCK_SIZE,
        n_ctx=BLOCK_SIZE,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    model = GPT2LMHeadModel(config)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model

# ============================================================
# MODEL CREATION
# ============================================================
def get_model():
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=BLOCK_SIZE,
        n_ctx=BLOCK_SIZE,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    model = GPT2LMHeadModel(config)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model
