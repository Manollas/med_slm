import os
from datasets import load_dataset, load_from_disk
from transformers import (
    GPT2LMHeadModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from common.utils import *
from common.data_process import format_instruction,tokenize_phase3

# ============================================================
# PATHS
# ============================================================

BASE_PATH=os.getenv("BASE_PATH")
LOG_PATH = f"{BASE_PATH}/logs"
LOG_PATH_PHASE3=LOG_PATH+"/phase3"

os.makedirs(f"{LOG_PATH}/phase3", exist_ok=True)
os.environ["TENSORBOARD_LOGGING_DIR"] = LOG_PATH_PHASE3

# ============================================================
# CONFIG
# ============================================================
BLOCK_SIZE = os.getenv("BLOCK_SIZE")
PHASE3_RAW_PATH = os.getenv("PHASE3_RAW_PATH")
PHASE3_PROCESSED_PATH = os.getenv("PHASE3_PROCESSED_PATH")
PHASE3_MODEL_CHECKPOINT_PATH = os.getenv("PHASE3_MODEL_CHECKPOINT_PATH")
PHASE3_MODEL_SAVE_PATH=os.getenv("PHASE3_MODEL_SAVE_PATH")
PHASE2_MODEL_SAVE_PATH=os.getenv("PHASE2_MODEL_SAVE_PATH")

# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# LOAD + PROCESS DATASET
# ============================================================
def load_alpaca():

    # already processed
    if os.path.exists(PHASE3_PROCESSED_PATH):
        print("✅ Loading processed Alpaca dataset...")
        return load_from_disk(PHASE3_PROCESSED_PATH)

    print("📥 Downloading + processing Alpaca dataset...")

    dataset = load_dataset(
        "tatsu-lab/alpaca",
        cache_dir=PHASE3_RAW_PATH
    )

    dataset = dataset.map(format_instruction)

    tokenized = dataset.map(
        tokenize_phase3,
        batched=False,
        remove_columns=dataset["train"].column_names,
    )

    tokenized.save_to_disk(PHASE3_PROCESSED_PATH)

    return tokenized

# ============================================================
# LOAD DATASET
# ============================================================
def train_alpaca():
    
    dataset = load_alpaca()

    print("🧠 Loading phase2 model...")
    model, resumed = load_model_safely(PHASE3_MODEL_CHECKPOINT_PATH)

    if not resumed:
        model = GPT2LMHeadModel.from_pretrained(PHASE2_MODEL_SAVE_PATH)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=PHASE3_MODEL_CHECKPOINT_PATH,

        # batching
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,

        # training length
        max_steps=8000,

        # learning
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=200,

        # regularization
        weight_decay=0.01,

        # mixed precision
        fp16=True,

        # logging
        logging_steps=50,
        save_steps=1000,
        save_total_limit=3,

        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
    )

    trainer.train()

    trainer.save_model(PHASE3_MODEL_SAVE_PATH)
    tokenizer.save_pretrained(PHASE3_MODEL_SAVE_PATH)

    print("✅ Phase 3 training complete")


train_alpaca()


# ============================================================
# TEST MODEL
# ============================================================

# if __name__ == "__main__":

#     from transformers import pipeline

#     generator = pipeline(
#         "text-generation",
#         model="./final-model",
#         tokenizer=tokenizer,
#         device=0 if torch.cuda.is_available() else -1,
#     )

#     prompt = """### Instruction:
# Explain why the sky appears blue during the day.

# ### Response:
# """

#     output = generator(
#         prompt,
#         max_new_tokens=200,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.9,
#         repetition_penalty=1.2,
#     )

#     print("\n================ MODEL OUTPUT ================\n")
#     print(output[0]["generated_text"])