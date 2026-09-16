from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer

# Setup directory: data/multi30k/tok_word_level_en_de_hf
SCRIPT_DIR = Path(__file__).resolve().parent

# Raw data directory: data/multi30k/raw
RAW_DIR = SCRIPT_DIR.parent / "raw"


def setup_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def add_post_processor(tok: Tokenizer) -> None:
    tok.post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        special_tokens=[
            ("[SOS]", tok.token_to_id("[SOS]")),
            ("[EOS]", tok.token_to_id("[EOS]")),
        ],
    )


def get_lines(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.strip()


def make_tokenizer(raw_file_name: str, tokenizer_name: str) -> Tokenizer:
    tokenizer = setup_tokenizer()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=1
    )

    raw_file_path = RAW_DIR / raw_file_name
    print(f"Training vocabulary on {raw_file_name}...")
    tokenizer.train_from_iterator(get_lines(raw_file_path), trainer)

    add_post_processor(tokenizer)

    save_path = SCRIPT_DIR / f"{tokenizer_name}.json"
    tokenizer.save(str(save_path))
    print(f"Saved tokenizer -> {save_path.name}")

    return tokenizer


def make_dataset(raw_file_name: str, dataset_name: str, tokenizer: Tokenizer):
    print(f"Encoding {raw_file_name}...")
    dataset = []
    with open(RAW_DIR / raw_file_name, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(tokenizer.encode(line.strip()).ids)

    save_path = SCRIPT_DIR / f"{dataset_name}.pt"
    torch.save(dataset, save_path)
    print(f"Saved split -> {save_path.name} ({len(dataset)} items)")


if __name__ == "__main__":
    print("making tokenizers and datasets for Multi30k (word-level)...")
    # 1. Train tokenizers ONLY on training files
    tokenizer_en = make_tokenizer("train.en", "tokenizer_en")
    tokenizer_de = make_tokenizer("train.de", "tokenizer_de")

    # 2. Encode all splits per language using their respective trained tokenizers
    splits = {
        "train": ("train.en", "train.de"),
        "val": ("val.en", "val.de"),
        "test": ("test_2016_flickr.en", "test_2016_flickr.de"),
    }

    for split_name, (en_file, de_file) in splits.items():
        make_dataset(en_file, f"{split_name}_en_ids", tokenizer_en)
        make_dataset(de_file, f"{split_name}_de_ids", tokenizer_de)
    print("data preparation complete!")