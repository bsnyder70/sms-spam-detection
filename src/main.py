import torch
from torch import nn
from TransformerClassifier import TransformerClassifier
from train import generate_train_test, train, evaluate
from sklearn.metrics import confusion_matrix, classification_report

def main():

    model = TransformerClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss() 

    train_loader, valid_loader, test_loader = generate_train_test()

    train(model, train_loader, valid_loader, optimizer, criterion)

    test_loss, test_acc, preds, labels = evaluate(model, test_loader, nn.BCELoss())

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    cm = confusion_matrix(labels, preds)
    print(cm)

    
    print(classification_report(labels, preds, target_names=["Ham", "Spam"]))
main()