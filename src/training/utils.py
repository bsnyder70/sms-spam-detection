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
    