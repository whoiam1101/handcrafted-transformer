import torch

from pathlib import Path


__all__ = ["EarlyStopping"]


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(
        self,
        patience: int = 3,
        verbose: bool = False,
        path: Path | str = "checkpoints/checkpoint.pt",
        delta: float = 0
    ):
        """
        Args:
            patience (int): How long to wait after last improvement.
            verbose (bool): Print messages when stopping.
            path (Path | str): Path to save the best model.
            delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.best_model_state = None

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        """Saves the best model."""
        if self.verbose:
            print(f"Validation loss improved to {val_loss}. Saving model...")
        torch.save(model.state_dict(), self.path)
