import torch
from torch import nn
from TransformerClassifier import TransformerClassifier
from train import generate_train_test, train, evaluate
from sklearn.metrics import confusion_matrix, classification_report
from data_process import get_input_from_text, Vocabulary, build_data

def main():

    # move these to a config 
    batch_size = 32
    num_epochs = 10
    embed_dim = 128
    num_heads = 4
    ff_dim = 128
    dropout = 0.3
    max_length = 180
    num_encoder_layers = 2
    class_hidden_dim = 64
    learning_rate = 1e-4

    # Download the data and generate train/test splits.
    dataset, vocab_size = build_data()
    train_loader, valid_loader, test_loader = generate_train_test(dataset=dataset, batch_size=batch_size)
    
    # Initialize the model, optimizer, and crtierion.
    model = TransformerClassifier(vocab_size, embed_dim, num_heads, ff_dim, dropout, max_length, num_encoder_layers, class_hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss() 

    train(model=model, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, criterion=criterion, num_epochs=num_epochs)

    test_loss, test_acc, preds, labels = evaluate(model, test_loader, nn.BCELoss())

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    cm = confusion_matrix(labels, preds)
    print(cm)

    print(classification_report(labels, preds, target_names=["Ham", "Spam"]))

def run_model(text=None):
    text = "Testing spam text"
    vocabulary = Vocabulary()
    input = get_input_from_text(text, vocabulary)

main()