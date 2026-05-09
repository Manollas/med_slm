import torch
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
import os
from datasets import load_dataset
from common.utils import *
from common.data_process import tokenize_phase1,group_texts_phase1

# ============================================================
# PATHS
# ============================================================
# BASE_PATH = "C:/RP/hospital_management_api/slm"
BASE_PATH=os.getenv("BASE_PATH")
LOG_PATH = f"{BASE_PATH}/logs"
LOG_PATH_PHASE1=LOG_PATH+"/phase1"

os.makedirs(f"{LOG_PATH}/phase1", exist_ok=True)
os.environ["TENSORBOARD_LOGGING_DIR"] = LOG_PATH_PHASE1

# ============================================================
# CONFIG
# ============================================================
BLOCK_SIZE = os.getenv("BLOCK_SIZE")
PHASE1_RAW_PATH = os.getenv("PHASE1_RAW_PATH")
PHASE1_PROCESSED_PATH = os.getenv("PHASE1_PROCESSED_PATH")

PHASE1_MODEL_CHECKPOINT_PATH = os.getenv("PHASE1_MODEL_CHECKPOINT_PATH")
PHASE1_MODEL_SAVE_PATH=os.getenv("PHASE1_MODEL_SAVE_PATH")

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 🟢 PHASE 1
# ============================================================
def train_phase1_model():

    dataset = load_dataset("roneneldan/TinyStories",cache_dir=PHASE1_RAW_PATH)

    tokenized = dataset.map(tokenize_phase1, batched=True, remove_columns=["text"])
    lm_dataset = tokenized.map(group_texts_phase1, batched=True)

    model, resumed = load_model_safely(PHASE1_MODEL_CHECKPOINT_PATH)

    # training_args = TrainingArguments(
    #     output_dir="./phase1",

    #     per_device_train_batch_size=2,
    #     gradient_accumulation_steps=8,

    #     max_steps=5000,

    #     learning_rate=2e-4 if resumed else 3e-4,

    #     # 🔥 IMPORTANT: disable fp16 if resuming
    #     fp16=False if resumed else True,

    #     logging_steps=100,

    #     save_steps=100,
    #     save_total_limit=3,

    #     report_to="tensorboard",
    # )


    training_args = TrainingArguments(
        output_dir=PHASE1_MODEL_CHECKPOINT_PATH,

        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,

        max_steps=10000,

        learning_rate=2e-4 if resumed else 3e-4,

        fp16=True,  # ✅ always ON now

        # 🔥 CRITICAL FIX
        lr_scheduler_type="cosine",
        warmup_steps=300,

        logging_steps=100,

        save_steps=500,   # ✅ reduce overhead
        save_total_limit=3,

        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
    )

    trainer.train()

    trainer.save_model(PHASE1_MODEL_SAVE_PATH)
    tokenizer.save_pretrained(PHASE1_MODEL_SAVE_PATH)

train_phase1_model()