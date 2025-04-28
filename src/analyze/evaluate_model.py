from matplotlib import pyplot as plt
import numpy as np
from src.data_process import build_data
from model.TransformerClassifier import TransformerClassifier
from sklearn.metrics import confusion_matrix, classification_report
import torch

from utils.splits import generate_stratified_splits

# Default config for initializing model
default_config = {
    # Training parameters
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    # Model architecture parameters
    "vocab_size": 3961,
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


@torch.no_grad()
def plot_prediction_confidence(model, dataloader, device="cuda"):

    model.eval()
    model.to(device)

    all_probs = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)  # assumed logits
        probs = torch.sigmoid(outputs).squeeze()  # (batch,) if binary classification

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    plt.figure(figsize=(10, 6))
    plt.hist(
        all_probs[all_labels == 0], bins=20, alpha=0.6, label="Ham (0)", color="skyblue"
    )
    plt.hist(
        all_probs[all_labels == 1], bins=20, alpha=0.6, label="Spam (1)", color="salmon"
    )
    plt.axvline(0.5, color="gray", linestyle="--", label="Decision Threshold")
    plt.title("Prediction Confidence Histogram (Test Set)")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/prediction_confidence_histogram.png")


if __name__ == "__main__":
    dataset, _ = build_data()
    _, _, test_loader = generate_stratified_splits(
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
