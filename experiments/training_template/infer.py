import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_xfmr2017 import load_exp_for_infer

experiment_dir = Path(__file__).resolve().parent

translator, config = load_exp_for_infer(
    experiment_dir,
    checkpoint_name="checkpoints_eager",
    weights_name="weights_final.pt",
    is_compile=False,
)

prompt = "Two young, White males are outside near many bushes."

response = translator.translate(prompt)

print(f"\nSource: {prompt}")
print(f"Translation: {response}")