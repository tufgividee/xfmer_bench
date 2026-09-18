import sys
from pathlib import Path

import train_config as cfg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_xfmr2017 import build_exp_for_train

experiment_dir = Path(__file__).resolve().parent

trainer, config = build_exp_for_train(
    experiment_dir,
    vars(cfg),
)

trainer.fit(epochs=cfg.epochs)