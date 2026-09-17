from pathlib import Path

import torch
from torch import nn, optim

from xfmr2017.data import create_dataloader, load_tokenizer
from xfmr2017.training import Trainer
from xfmr2017.transformer import Transformer

is_compile = False

# directory
script_dir = Path(__file__).resolve().parent

if is_compile:
    checkpoint_dir = script_dir / "checkpoints_compile"
else:
    checkpoint_dir = script_dir / "checkpoints_eager"

checkpoint_dir.mkdir(parents=True, exist_ok=True)

project_dir = script_dir.parent.parent
data_dir = project_dir / "data/multi30k/tok_wordlevel_data_build_hf"

# tokenizers
tokenizer_en = load_tokenizer(data_dir / "tokenizer_en.json")
tokenizer_de = load_tokenizer(data_dir / "tokenizer_de.json")

src_vocab_size = tokenizer_en.get_vocab_size()
tgt_vocab_size = tokenizer_de.get_vocab_size()

src_pad_idx = tokenizer_en.token_to_id("[PAD]")
tgt_pad_idx = tokenizer_de.token_to_id("[PAD]")

print(f"English vocabulary size: {src_vocab_size}")
print(f"German vocabulary size:  {tgt_vocab_size}")
print(f"English PAD ID: {src_pad_idx}")
print(f"German PAD ID: {tgt_pad_idx}")

# token ids data
batch_size = 32

train_loader = create_dataloader(
    files=[
        data_dir / "train_en_ids.pt",
        data_dir / "train_de_ids.pt",
    ],
    pad_ids=[
        src_pad_idx,
        tgt_pad_idx,
    ],
    batch_size=batch_size,
    shuffle=True,
)

val_loader = create_dataloader(
    files=[
        data_dir / "val_en_ids.pt",
        data_dir / "val_de_ids.pt",
    ],
    pad_ids=[
        src_pad_idx,
        tgt_pad_idx,
    ],
    batch_size=batch_size,
    shuffle=False,
)

# device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# hyperparameters
hyperparams = {
    "d_model": 256,
    "num_heads": 4,
    "num_layers": 4,
    "d_ff": 512,
    "dropout": 0.1,
}

n_epochs = 2
# experiment configuration
config = {
    "dataset": str(data_dir.relative_to(project_dir)),
    "data_src": "train_en_ids.pt",
    "data_tgt": "train_de_ids.pt",
    "tokenizer_src": "tokenizer_en.json",
    "tokenizer_tgt": "tokenizer_de.json",
    "src_vocab_size": src_vocab_size,
    "tgt_vocab_size": tgt_vocab_size,
    "src_pad_idx": src_pad_idx,
    "tgt_pad_idx": tgt_pad_idx,
    "torch_compile": is_compile,
    "batch_size": batch_size,
    "learning_rate": 1e-4,
    "optimizer": "Adam",
    "epochs": n_epochs,
    **hyperparams,
}

# model
model = Transformer(
    src_vocab_size,
    tgt_vocab_size,
    hyperparams["d_model"],
    hyperparams["num_heads"],
    hyperparams["num_layers"],
    hyperparams["d_ff"],
    hyperparams["dropout"],
    src_pad_idx=src_pad_idx,
    tgt_pad_idx=tgt_pad_idx,
)

# loss criterion
# To match the paper (in train.py):
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx, label_smoothing=0.1)

# optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=config["learning_rate"],
    betas=(0.9, 0.98),
    eps=1e-9,
)

# trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    checkpoint_dir=checkpoint_dir,
    config=config,
    device=device,
)

trainer.fit(epochs=n_epochs)