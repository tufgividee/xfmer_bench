import json
from pathlib import Path

import torch

from transformer import Transformer
from transformer.data import load_tokenizer
from transformer.inference import Translator

# experiment directory
script_dir = Path(__file__).resolve().parent

# Select which experiment to run
is_compile = False

checkpoint_dir = script_dir / (
    "checkpoints_compile"
    if is_compile
    else "checkpoints_eager"
)

project_dir = script_dir.parents[1]

# Load experiment metadata
with open(checkpoint_dir / "config.json") as f:
    config = json.load(f)

data_dir = project_dir / config["dataset"]

# Tokenizers
tokenizer_en = load_tokenizer(
    data_dir / config["tokenizer_src"]
)
tokenizer_de = load_tokenizer(
    data_dir / config["tokenizer_tgt"]
)

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Model
model = Transformer(
    config["src_vocab_size"],
    config["tgt_vocab_size"],
    config["d_model"],
    config["num_heads"],
    config["num_layers"],
    config["d_ff"],
    config["dropout"],
    src_pad_idx=config["src_pad_idx"],
    tgt_pad_idx=config["tgt_pad_idx"],
).to(device)

# Load weights
model.load_state_dict(
    torch.load(
        checkpoint_dir / "weights_final.pt",
        map_location=device,
        weights_only=True,
    )
)

# Compile if this experiment was trained with compile
if config["torch_compile"]:
    model = torch.compile(model)

translator = Translator(
    model=model,
    src_tokenizer=tokenizer_en,
    tgt_tokenizer=tokenizer_de,
    device=device,
)

english_prompt = "Two young, White males are outside near many bushes."

# with open 

translation = translator.translate(english_prompt)

print(f"\nSource: {english_prompt}")
print(f"Translation: {translation}")