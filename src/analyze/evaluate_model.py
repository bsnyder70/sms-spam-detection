from src.data_process import build_data
from model.TransformerClassifier import TransformerClassifier
from src.train import generate_train_test
from sklearn.metrics import confusion_matrix, classification_report
import torch

# Default config for initializing model
default_config = {
    # Training parameters
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    # Model architecture parameters
    "vocab_size": 3963,
    "embed_dim": 128,
    "num_heads": 4,
    "ff_dim": 128,
    "dropout": 0.0,
    "max_length": 180,
    "num_encoder_layers": 2,
    "class_hidden_dim": 64,
}

# Models to evaluate
models = [
    {
        "name": "BCE_Classifier",
        "path": "outputs/bce_model.pth",
        "model": TransformerClassifier,
        "config": default_config,
    },
]


def evaluate_from_path(
    model_cls,
    model_config: dict,
    checkpoint_path: str,
    dataloader,
    device,
) -> None:
    # Build model from config using class method
    model = model_cls.from_config(**model_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.tolist())

    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print(
        "\nClassification Report:\n",
        classification_report(all_labels, all_preds, target_names=["Ham", "Spam"]),
    )


if __name__ == "__main__":
    dataset, _ = build_data()
    _, _, test_loader = generate_train_test(
        dataset=dataset, batch_size=default_config["batch_size"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_info in models:
        print(f"\n Evaluating {model_info['name']}")
        evaluate_from_path(
            model_cls=model_info["model"],
            model_config=model_info["config"],
            checkpoint_path=model_info["path"],
            dataloader=test_loader,
            device=device,
        )
