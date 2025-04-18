import torch
from torch import nn
import numpy as np
from torch.utils.data import random_split, DataLoader
import data_process
from TransformerClassifier import TransformerClassifier

def generate_train_test():
    """
    Generates train / validation / test DataLoader sets from our data set.
    """
    BATCH_SIZE = 32

    # Note size of dataset is 5572
    spam_dataset = data_process.build_data()

    train_size = int(0.7 * len(spam_dataset))
    val_size = int(0.15 * len(spam_dataset))
    test_size = len(spam_dataset) - (train_size + val_size)

    train_dataset, val_dataset, test_dataset = random_split(spam_dataset, [train_size, val_size, test_size])
    
    # probably have to do collate fn here to pad all the elements 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                        shuffle=False)
    valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False)
    test_loader =  DataLoader(test_dataset, batch_size=BATCH_SIZE,
                       shuffle=False)
    
    return train_loader, valid_loader, test_loader

def train(model, train_loader, valid_loader, optimizer, criterion):
    """
    xxx
    """
    
    NUM_EPOCHS = 10

    for epoch_idx in range(NUM_EPOCHS):
        
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

        print(f"Epoch {epoch_idx+1}/{NUM_EPOCHS} | Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

def evaluate(model, data_loader, criterion):
    """
    xxx
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