from src import data_process
from src.train import TransformerClassifier
import torch


# run the code

models = [
    {
        "name": "BCE_Classifier",
        "path:": "outputs/bce_model.pth",
        "model": TransformerClassifier,
    }
]


def evaluate_from_path(
    model_cls,  # e.x TransformerClassifier
    model_config: dict,  # config to initialize the model
    checkpoint_path: str,
    dataloader,
    device,
):
    model = model_cls(**model_config).to(device)
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


if __name__ == "__main__":

    # Download the data and generate train/test splits.
    dataset, vocab_size = data_process.build_data()

    for model in models:
        model_name = model["name"]
        model_path = model["path:"]
        model_cls = model["model"]
