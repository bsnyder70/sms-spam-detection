import os
import torch
from typing import Optional, Callable, Any
from torch.utils.data import DataLoader
from training.trainer import train_epoch


def train(
    num_epochs: int,
    train: DataLoader,
    valid: DataLoader,
    model,
    optimizer,
    device,
    loss_fn: Callable[[Any, Any], float],
    validate: Optional[Callable[[Any, Any], float]] = None,
    save_path: str = "outputs/model.pth",
):
    print(f"Training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs}")

        train_acc, train_loss = train_epoch(
            model,
            train,
            optimizer,
            device,
            loss_fn=loss_fn,
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    if validate:
        val_loss = validate(model, valid, device, loss_fn)
        print(f"Validation Loss: {val_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
