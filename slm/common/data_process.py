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
from itertools import chain

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
BLOCK_SIZE = os.getenv("BLOCK_SIZE")

# ============================================================
# DATA FUNCTIONS PHASE1
# ============================================================
def tokenize_phase1(example):
    return tokenizer(example["text"])

def group_texts_phase1(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // BLOCK_SIZE) * BLOCK_SIZE

    result = {
        k: [t[i:i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# ============================================================
# DATA FUNCTIONS PHASE2
# ============================================================
def tokenize_opentxt(example):
    return tokenizer(example["text"], truncation=True)

def group_texts_opentxt(examples):
    concatenated = {
        k: list(chain(*examples[k])) for k in examples.keys()
    }

    total_length = (len(concatenated["input_ids"]) // BLOCK_SIZE) * BLOCK_SIZE

    result = {
        k: [t[i:i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }

    result["labels"] = result["input_ids"].copy()
    return result


# ============================================================
# DATA FUNCTIONS PHASE3
# ============================================================

# FORMAT DATASET
def format_instruction(example):
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    }


# TOKENIZATION WITH RESPONSE-ONLY LOSS
def tokenize_phase3(example):

    full_text = example["text"]

    # split instruction vs response
    instruction_part = full_text.split("### Response:")[0]

    # tokenize full sample
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=BLOCK_SIZE,
    )

    # tokenize only instruction
    instruction_tokens = tokenizer(
        instruction_part,
        truncation=True,
        max_length=BLOCK_SIZE,
    )

    labels = full_tokens["input_ids"].copy()

    # mask instruction tokens
    instruction_len = len(instruction_tokens["input_ids"])

    labels[:instruction_len] = [-100] * instruction_len

    full_tokens["labels"] = labels

    return full_tokens