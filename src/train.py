import os
from sklearn.metrics import f1_score, precision_score, recall_score
from interfaces import SupportsFromConfig
import torch
from typing import Callable, Any
from torch.utils.data import DataLoader
from training.trainer import train_epoch, validate_epoch

from utils.plots import plot_learning_curves


def train(
    config,
    model_cls,
    train_data: DataLoader,
    valid_data: DataLoader,
    loss_fn: Callable[[Any, Any], float],
    save_path = "outputs/model.pth",
):
    """
    Train the model using the given train/validation data sets and save the model.
    """

    train_losses = []
    val_losses = []
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    model = model_cls.from_config(**config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["num_epochs"]):
        print(f"\n Epoch {epoch+1}/{config['num_epochs']}")

        train_loss, train_acc = train_epoch(
            model,
            train_data,
            optimizer,
            device,
            loss_fn=loss_fn,
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        train_losses.append(train_loss)

        val_loss, val_acc, all_preds, all_labels = validate_epoch(
            model, valid_data, device, loss_fn
        )

        probs = torch.sigmoid(torch.tensor(all_preds))
        binary_preds = (probs > 0.5).int().tolist()

        print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")
        val_losses.append(val_loss)

        # Calculate precision, recall, and F1 score
        precision = precision_score(all_labels, binary_preds, zero_division=0)
        recall = recall_score(all_labels, binary_preds, zero_division=0)
        f1 = f1_score(all_labels, binary_preds, zero_division=0)

        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    plot_learning_curves(
        train_losses, val_losses, save_path.replace(".pth", "_curve.png")
    )

    return model
