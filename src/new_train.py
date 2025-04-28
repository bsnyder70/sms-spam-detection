import os
import torch
from typing import Optional, Callable, Any
from torch.utils.data import DataLoader
from training.trainer import train_epoch


def train(
    num_epochs: int,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    model,
    optimizer,
    device,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    validate: Optional[Callable] = None,
    save_path: str = "outputs/model.pth",
):
    """
    Standard training loop. Moves everything to `device`.
    """
    model.to(device)
    print(f"Training for {num_epochs} epochs on {device}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        running_correct = 0
        running_total = 0

        for examples, labels in train_loader:
            # move to device
            examples = examples.to(device)
            labels   = labels.to(device)

            optimizer.zero_grad()
            outputs = model(examples)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # accumulate metrics
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_acc = running_correct / running_total
        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    # optional validation
    if validate:
        val_loss, val_acc, *_ = validate(model, valid_loader, loss_fn, device)
        print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")

    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

