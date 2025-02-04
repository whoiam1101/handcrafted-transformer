import torch
import torch.nn as nn

from torch import DeviceObjType
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        num_epochs: int,
        batch_size: int,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        loss_fn: callable,
        metrics_fn: callable,
        optimizer: Optimizer,
        optimizer_kwargs: dict[str, any],
        device: DeviceObjType | None = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_path: Path | str | None = None,
        log_interval: int = 10,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_path = checkpoint_path
        self.log_interval = log_interval

        self.model = model.to(self.device)

        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn

        self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.eval_dataloader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False
        )

        self.train_loss_history = []
        self.eval_loss_history = []
        self.train_metrics_history = []
        self.eval_metrics_history = []

        if self.checkpoint_path:
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        self.model.train()
        global_step = 0

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            self.model.train()

            progress_bar = tqdm(
                self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"
            )
            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = self.loss_fn(outputs, batch["labels"])
                loss = loss / self.gradient_accumulation_steps

                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    global_step += 1

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                if global_step % self.log_interval == 0:
                    progress_bar.set_postfix({"loss": loss.item()})

            epoch_loss /= len(self.train_dataloader)
            self.train_loss_history.append(epoch_loss)

            eval_loss, eval_metrics = self.evaluate()
            self.eval_loss_history.append(eval_loss)
            self.eval_metrics_history.append(eval_metrics)

            if self.checkpoint_path:
                self.save_checkpoint(epoch, global_step)

            print(
                f"Epoch {epoch + 1}/{self.num_epochs} |
                Train Loss: {epoch_loss:.4f} |
                Eval Loss: {eval_loss:.4f} |
                Eval Metrics: {eval_metrics}"
            )

    @torch.no_grad()
    def evaluate(self) -> tuple[float, dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            loss = self.loss_fn(outputs, batch["labels"])
            total_loss += loss.item()

            all_preds.append(outputs.logits.argmax(dim=-1))
            all_labels.append(batch["labels"])

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self.metrics_fn(all_preds, all_labels)

        avg_loss = total_loss / len(self.eval_dataloader)
        return avg_loss, metrics

    def save_checkpoint(self, epoch: int, global_step: int) -> None:
        checkpoint_path = (
            self.checkpoint_path / f"checkpoint_epoch_{epoch}_step_{global_step}.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss_history": self.train_loss_history,
                "eval_loss_history": self.eval_loss_history,
                "eval_metrics_history": self.eval_metrics_history,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path | str) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_loss_history = checkpoint["train_loss_history"]
        self.eval_loss_history = checkpoint["eval_loss_history"]
        self.eval_metrics_history = checkpoint["eval_metrics_history"]
        print(f"Checkpoint loaded from {checkpoint_path}")


Trainer.__doc__ = """
A training wrapper class for PyTorch models.

This class provides a standard training loop for PyTorch models, including support for
evaluation, checkpoint saving/loading, gradient accumulation, and progress tracking.

Args:
    model (nn.Module): The PyTorch model to train.
    num_epochs (int): Number of training epochs.
    batch_size (int): Size of batches for training and evaluation.
    train_dataset (Dataset): PyTorch Dataset for training.
    eval_dataset (Dataset): PyTorch Dataset for evaluation.
    loss_fn (callable): Loss function to use for training and evaluation.
    metrics_fn (callable): Function to compute evaluation metrics.
    optimizer (Optimizer): PyTorch optimizer class (not instance).
    optimizer_kwargs (dict): Keyword arguments for the optimizer.
    device (DeviceObjType, optional): Device to run the model on. Defaults to CUDA if available, else CPU.
    gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients. Defaults to 1.
    max_grad_norm (float, optional): Maximum gradient norm for gradient clipping. Defaults to 1.0.
    checkpoint_path (Path | str, optional): Directory to save model checkpoints. Defaults to None.
    log_interval (int, optional): Number of steps between logging updates. Defaults to 10.

Attributes:
    train_loss_history (list): History of training losses.
    eval_loss_history (list): History of evaluation losses.
    eval_metrics_history (list): History of evaluation metrics.

Examples:
    >>> trainer = Trainer(
    ...     model=my_model,
    ...     num_epochs=10,
    ...     batch_size=32,
    ...     train_dataset=train_data,
    ...     eval_dataset=eval_data,
    ...     loss_fn=torch.nn.CrossEntropyLoss(),
    ...     metrics_fn=compute_metrics,
    ...     optimizer=torch.optim.Adam,
    ...     optimizer_kwargs={'lr': 0.001}
    ... )
    >>> trainer.train()
"""
