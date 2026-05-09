from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
import os
from datasets import load_dataset, load_from_disk
from common.utils import *
from common.data_process import tokenize_opentxt,group_texts_opentxt

# ============================================================
# PATHS
# ============================================================
# BASE_PATH = "C:/RP/hospital_management_api/slm"
BASE_PATH=os.getenv("BASE_PATH")
LOG_PATH = f"{BASE_PATH}/logs"
LOG_PATH_PHASE2=LOG_PATH+"/phase2"
os.makedirs(f"{LOG_PATH}/phase2", exist_ok=True)
os.environ["TENSORBOARD_LOGGING_DIR"] = LOG_PATH_PHASE2

# ============================================================
# CONFIG
# ============================================================
BLOCK_SIZE = os.getenv("BLOCK_SIZE")
PHASE2_RAW_PATH = os.getenv("PHASE2_RAW_PATH")
PHASE2_PROCESSED_PATH = os.getenv("PHASE2_PROCESSED_PATH")

PHASE2_MODEL_CHECKPOINT_PATH = os.getenv("PHASE2_MODEL_CHECKPOINT_PATH")
PHASE2_MODEL_SAVE_PATH=os.getenv("PHASE2_MODEL_SAVE_PATH")
PHASE1_MODEL_SAVE_PATH=os.getenv("PHASE1_MODEL_SAVE_PATH")

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def load_opentxt_data():
    dataset = load_dataset(
        "openwebtext",
        split="train[:20%]",
        cache_dir=PHASE2_RAW_PATH
    )

    tokenized = dataset.map(
        tokenize_opentxt,
        batched=True,
        remove_columns=["text"],
        num_proc=4
    )

    lm_dataset = tokenized.map(
        group_texts_opentxt,
        batched=True,
        num_proc=4
    )

    lm_dataset.save_to_disk(PHASE2_PROCESSED_PATH)

# ============================================================
# 🔵 PHASE 2
# ============================================================
def train_model_phase2():
    print("🚀 Starting Phase 2 model")

    if os.path.exists(PHASE2_PROCESSED_PATH):
        print("⚡ Loading processed dataset...")
        lm_dataset = load_from_disk(PHASE2_PROCESSED_PATH)
    else:
        print("🔄 Processing dataset...")
        load_opentxt_data()

    # ---------------- MODEL ----------------
    model, resumed = load_model_safely(PHASE2_MODEL_CHECKPOINT_PATH)

    if not resumed:
        print("🚀 Loading Phase 1 model")
        model = GPT2LMHeadModel.from_pretrained(PHASE1_MODEL_SAVE_PATH)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # ---------------- TRAINING ----------------
    training_args = TrainingArguments(
        output_dir=PHASE2_MODEL_CHECKPOINT_PATH,

        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,

        max_steps=5000,

        learning_rate=5e-5 if resumed else 1e-4,
        fp16=not resumed,

        lr_scheduler_type="cosine",
        warmup_steps=300,

        weight_decay=0.01,

        logging_steps=100,

        save_steps=1000,
        save_total_limit=3,

        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,  # ✅ FIXED
    )

    trainer.train()

    trainer.save_model(PHASE2_MODEL_SAVE_PATH)
    tokenizer.save_pretrained(PHASE2_MODEL_SAVE_PATH)

train_model_phase2()