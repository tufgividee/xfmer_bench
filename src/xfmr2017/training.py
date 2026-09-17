import datetime
import json
import subprocess
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        checkpoint_dir: str | Path,
        config: dict,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.config = config

        self.model = model.to(self.device)
        
        if self.config["torch_compile"]:
            self.model = torch.compile(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = self.checkpoint_dir / "config.json"
        self.log_path = self.checkpoint_dir / "training.log"
        self.log_path.unlink(missing_ok=True)

    def save_config(self, timestamp_start: str) -> None:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()

        git_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
            ).strip()
        )

        config = {
            **self.config,
            "timestamp_start": timestamp_start,
            "device": str(self.device),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        }

        if self.device.type == "cuda":
            config["gpu"] = torch.cuda.get_device_name(self.device)

        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=4)

        def log(self, message: str) -> None:
            print(message)

            with open(self.log_path, "a") as f:
                f.write(message + "\n")

    def train_epoch(self) -> float:
        self.model.train()

        total_loss = 0.0

        for src, tgt in self.train_loader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            self.optimizer.zero_grad()

            output = self.model(src, tgt_input)

            loss = self.criterion(
                output.reshape(-1, output.size(-1)),
                tgt_expected.reshape(-1),
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def reset_peak_gpu_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak_gpu_memory_mb(self) -> float | None:
        if self.device.type != "cuda":
            return None

        peak_memory = torch.cuda.max_memory_allocated(self.device)
        return peak_memory / (1024 ** 2)

    def fit(self, epochs: int) -> None:
        self.reset_peak_gpu_memory()

        timestamp_start = datetime.datetime.now(datetime.UTC).isoformat()
        start_time = time.perf_counter()

        self.save_config(timestamp_start)

        self.log(f"Launching training on: {self.device}")
        self.log(f"Training started: {timestamp_start}")

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()

            train_loss = self.train_epoch()

            epoch_time = time.perf_counter() - epoch_start

            val_loss = self.validate()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(self.checkpoint_dir / "weights_min_val_loss.pt")
                self.log(
                    f"New best validation loss: {best_val_loss:.4f} "
                    f"at epoch {epoch}"
                )

            self.log(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Time: {epoch_time:.2f} s"
            )

        total_time = time.perf_counter() - start_time
        timestamp_end = datetime.datetime.now(datetime.UTC).isoformat()
        self.save(self.checkpoint_dir / "weights_final.pt")

        self.log(f"Training completed: {timestamp_end}")
        self.log(f"Total training time: {total_time:.2f} s")
        self.log(
            f"Average epoch time: "
            f"{total_time / epochs:.2f} s"
        )
        peak_memory = self.peak_gpu_memory_mb()

        if peak_memory is not None:
            self.log(f"Peak GPU memory usage: {peak_memory:.2f} MB")
        else:
            self.log("Peak GPU memory usage: N/A")

    def validate(self) -> float:
        self.model.eval()

        total_loss = 0.0

        with torch.no_grad():
            for src, tgt in self.val_loader:
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_expected = tgt[:, 1:]

                output = self.model(src, tgt_input)

                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_expected.reshape(-1),
                )

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save(self, path: str | Path) -> None:
        model = getattr(self.model, "_orig_mod", self.model)

        state_dict = {
            key: value.cpu()
            for key, value in model.state_dict().items()
        }

        torch.save(state_dict, path)