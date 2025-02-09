import torch
import torch.nn as nn

from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Callable, Any


def train_transformer(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    optimizer_class: Optimizer = Adam,
    optimizer_kwargs: dict[str, Any] | None = None,
    pad_idx: int = 0,
    device: torch.device | None = None,
    save_dir: Path | None = None,
    max_grad_norm: float = 1.0,
    metrics_fn: Callable | None = None,
) -> dict:
    """
    Train a Transformer model with a simplified setup.

    Args:
        model (nn.Module): Transformer model to train.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs to train.
        optimizer_class (Type[optim.Optimizer], optional): Optimizer class. Defaults to Adam.
        optimizer_kwargs (dict, optional): Optimizer parameters. Defaults to None.
        pad_idx (int, optional): Padding index for masking. Defaults to 0.
        device (torch.device, optional): Device for training (CPU or GPU). Defaults to None.
        save_dir (Optional[Path], optional): Directory to save model checkpoints. Defaults to None.
        max_grad_norm (float, optional): Maximum gradient norm for clipping. Defaults to 1.0.
        metrics_fn (Optional[Callable], optional): Function to compute evaluation metrics. Defaults to None.

    Returns:
        dict: Training history containing losses and metrics.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer_kwargs = optimizer_kwargs or {}
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    
    def create_masks(src, tgt):
        """Generate source and target masks for Transformer."""
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        subsequent_mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()
        return src_mask, tgt_pad_mask & ~subsequent_mask.unsqueeze(0)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in train_bar:
            src, tgt = batch['src'].to(device), batch['tgt'].to(device)
            src_mask, tgt_mask = create_masks(src, tgt[:, :-1])
            
            optimizer.zero_grad()
            outputs = model(src=src, tgt=tgt[:, :-1], src_mask=src_mask, tgt_mask=tgt_mask)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc='Validation'):
                src, tgt = batch['src'].to(device), batch['tgt'].to(device)
                src_mask, tgt_mask = create_masks(src, tgt[:, :-1])
                outputs = model(src=src, tgt=tgt[:, :-1], src_mask=src_mask, tgt_mask=tgt_mask)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
                total_val_loss += loss.item()
                if metrics_fn:
                    preds = torch.argmax(outputs, dim=-1)
                    all_preds.append(preds)
                    all_targets.append(tgt[:, 1:])
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        history['val_loss'].append(avg_val_loss)
        
        if metrics_fn:
            all_preds, all_targets = torch.cat(all_preds), torch.cat(all_targets)
            history['val_metrics'].append(metrics_fn(all_preds, all_targets))
        
        print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'Val metrics: {history['val_metrics'][-1]}')

    return history
