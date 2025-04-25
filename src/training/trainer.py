import torch
from typing import Callable
import tqdm

def train_epoch(
    model, 
    dataloader, 
    optimizer, 
    device,             
    loss_fn,
) -> tuple[float, float]:
    """
    XXX
    """

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for examples, labels in tqdm.tqdm(dataloader, desc="Training", unit="batch"):
        examples, labels = examples.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(examples)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss = total_loss / len(dataloader)

    return train_acc, train_loss

def validate_epoch(
    model,
    dataloader,
    device,
    loss_fn,
) -> tuple[float, float, list, list]:
    """
    Evaluates the model on the given dataloader.

    Parameters:
    XXX
    
    Returns:
        val_loss: Average loss over the dataset
        val_acc: Accuracy over the dataset
        all_preds: List of predicted labels
        all_labels: List of ground truth labels
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for examples, labels in dataloader:
            examples, labels = examples.to(device), labels.to(device)

            outputs = model(examples)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_loss = total_loss / len(dataloader)
    val_acc = correct / total

    return val_loss, val_acc, all_preds, all_labels