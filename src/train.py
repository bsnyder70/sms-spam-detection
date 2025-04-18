import torch
from torch import nn
import numpy as np
from torch.utils.data import random_split, DataLoader
import data_process
from TransformerClassifier import TransformerClassifier

def generate_train_test(dataset, batch_size):
    """
    Generates train / validation / test DataLoader sets from our data set.

    We use a train / val / test split of 0.7 / 0.15 / 0.15. 

    Parameters:
        dataset: Dataset to use
        batch_size: Batch size for each DataLoader

    Returns:
        Train / validation / test DataLoaders
    """

    # Separate into train/val/test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - (train_size + val_size)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=False)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=False)
    test_loader =  DataLoader(test_dataset, batch_size=batch_size,
                       shuffle=False)
    
    return train_loader, valid_loader, test_loader

def train(model, train_loader, valid_loader, optimizer, criterion, num_epochs):
    """
    Trains the model using given train/validation sets with
    the provided optimizer and criterion.

    Parameters:
        model: Model to train
        train_loader: Training data set
        valid_loader: Validation data set
        optimizer: Optimizer to use for updating weights
        criterion: Criterion to use for calculating loss
        num_epochs: Number of epochs to train for

    Returns:
        train_acc: Final training accuracy
        total_loss: Final training loss
        val_acc: Final validation accuracy 
        val_loss: Final validation loss

    """

    for epoch_idx in range(num_epochs):
        
        total_loss = 0
        correct = 0
        total = 0
        model.train()

        for batch_idx, (examples, labels) in enumerate(train_loader):

            optimizer.zero_grad()
            outputs = model.forward(examples)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        val_loss, val_acc, _, _ = evaluate(model, valid_loader, criterion)

        print(f"Epoch {epoch_idx+1}/{num_epochs} | Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    return train_acc, total_loss, val_acc, val_loss
def evaluate(model, data_loader, criterion):
    """
    Evaluate the model on given data using a provided criterion.

    Parameters:
        model: Model to evaluate
        data_loader: Dataset to use for evaluating the model
        criterion: Criterion to use for calculating loss

    Returns:
        avg_loss: Average loss over all the batches
        acc: Accuracy over all the batches
        tot_preds: All the predicted labels over all the batches.
        tot_labels: All the real labels over all the batches
    """

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    tot_preds = []
    tot_labels = []

    for batch_idx, (examples, labels) in enumerate(data_loader):

        outputs = model(examples)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        tot_preds.extend(preds)
        tot_labels.extend(labels)

    avg_loss = total_loss / len(data_loader)
    acc = correct / total

    return avg_loss, acc, tot_preds, tot_labels
    
#generate_train_test()