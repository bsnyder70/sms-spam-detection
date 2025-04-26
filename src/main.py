import os
import random
import pickle
import torch
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

import numpy as np

from train import train
from training.trainer import validate_epoch
from tuning import tuning

from model.config import default_config, focal_best_config
from model.TransformerClassifier import TransformerClassifier

from utils.loss import (
    FocalLoss,
    binary_cross_entropy_loss,
    get_loss_wrapper,
    get_pos_weight,
    weighted_bce_with_logits_loss,
)

from utils.splits import generate_kfold_splits, generate_stratified_splits
from analyze.evaluate_model import plot_prediction_confidence

from data_process import (
    build_data,
    preprocess_text,
    preprocess_text_minimal,
    preprocess_no_special_tokens,
    preprocess_raw,
    preprocess_stemmed,
    preprocess_wordpiece,
    get_input_from_text,
    get_top_k_words,
)



# ale experiement section
def run_training():
    #  best model for task was focal loss with alpha=0.25 and gamma=2.0
    dataset, vocab_size = build_data()

    loss_fn = FocalLoss(alpha=0.25, gamma=3.0)

    config = focal_best_config.copy()
    train_loader, valid_loader, test_loader = generate_stratified_splits(
        dataset, batch_size=config["batch_size"]
    )

    model = train(
        config=config,
        model_cls=TransformerClassifier,
        train=train_loader,
        valid=valid_loader,
        loss_fn=loss_fn,
    )

    # load model from
    checkpoint_path = "outputs/focal_model.pth"
    model = TransformerClassifier.from_config(**config).to("cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    # Evaluate the model
    plot_prediction_confidence(
        model=model,
        dataloader=test_loader,
        device="cpu",
    )


def run_tuning():

    # load the dataset
    dataset, vocab_size = build_data()
    folds = generate_kfold_splits(dataset, k=5)

    # loss
    wrapped_loss = get_loss_wrapper(binary_cross_entropy_loss, apply_sigmoid=True)

    # BCE Loss
    config = default_config.copy()
    best = tuning(
        config=config,
        dataset=dataset,
        folds=folds,
        model_cls=TransformerClassifier,
        loss_fn=wrapped_loss,
    )

    # save the pd dataframe
    df = pd.DataFrame([best])
    df.to_csv("outputs/bce_model_summary.csv", index=False)

    # Weighted BCE Loss
    model_config = default_config.copy()

    base_pos_weight = get_pos_weight(dataset)
    results = []

    # Try different scaled versions
    for multiplier in [0.5, 1.0, 1.5, 2.0]:
        scaled_weight = base_pos_weight * multiplier
        print(
            f"\nRunning tuning with pos_weight={scaled_weight:.2f} (multiplier={multiplier})"
        )

        # Define loss function
        loss_fn = weighted_bce_with_logits_loss(scaled_weight)

        # Optional: label for plots/logs
        model_config["loss_label"] = f"pos_weight={scaled_weight:.2f}"

        best = tuning(
            config=model_config,
            dataset=dataset,
            folds=folds,
            model_cls=TransformerClassifier,
            loss_fn=loss_fn,
            sweep_label=f"pos_weight_{scaled_weight:.2f}",
        )
        best["best_pos_weight"] = base_pos_weight
        best["pos_multiplier"] = multiplier
        best["pos_weight"] = scaled_weight

        results.append(best)
    # Save all best configs
    pd.DataFrame(results).sort_values("f1", ascending=False).to_csv(
        "outputs/pos_weight_sweep_summary.csv", index=False
    )

    # Focal loss tuning
    alphas = [0.25, 0.5, 1.0]
    gammas = [1.0, 2.0, 3.0]

    results = []

    for alpha in alphas:
        for gamma in gammas:
            label = f"focal_a{alpha}_g{gamma}"
            print(f"\n Tuning for alpha={alpha}, gamma={gamma}")
            loss_fn = FocalLoss(alpha=alpha, gamma=gamma)

            best = tuning(
                config=model_config,
                dataset=dataset,
                folds=folds,
                model_cls=TransformerClassifier,
                loss_fn=loss_fn,
                sweep_label=label,
            )

            best["alpha"] = alpha
            best["gamma"] = gamma
            results.append(best)

    # Save all best configs
    pd.DataFrame(results).sort_values("f1", ascending=False).to_csv(
        "outputs/focal_sweep_summary.csv", index=False
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    seed_everything(42)

    # Map a name to each preprocessing strategy
    strategies = {
        'default': preprocess_text,
        'minimal': preprocess_text_minimal,
        'no_special': preprocess_no_special_tokens,
        'raw': preprocess_raw,
        'stemmed': preprocess_stemmed,
        'wordpiece': preprocess_wordpiece
    }

    for name, proc_fn in strategies.items():
        print(f"\n\n=== Training with '{name}' tokenization ===")

        vocab_file_path = f"cache/{name}.vocab.pkl"
        # Rebuild dataset & vocabulary with this strategy
        dataset, vocab_size = build_data(preprocess_fn=proc_fn, save_vocab_to_file=True, vocab_path=vocab_file_path)

        config = default_config.copy()
        config["vocab_size"] = vocab_size

        train_loader, valid_loader, test_loader = generate_stratified_splits(
            dataset=dataset,
            batch_size=config["batch_size"]
        )

        criterion = FocalLoss(alpha=0.5, gamma=1.0)
        loss_used = 'focal'

        # criterion = get_loss_wrapper(binary_cross_entropy_loss, apply_sigmoid=True)
        # loss_used = 'BCE'

        # Save each model under a distinct name
        save_path = f"outputs/{name}_{loss_used}_model.pth"
        model = train(
            config=config,
            model_cls=TransformerClassifier,
            train=train_loader,
            valid=valid_loader,
            loss_fn=criterion,
            save_path=save_path,
        )

        # Evaluate on test set
        test_loss, test_acc, preds, labels = validate_epoch(
            model,
            test_loader,
            device=device,
            loss_fn=criterion,
        )
        print(f"[{name}] Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        cm = confusion_matrix(labels, preds)
        print(cm)
        print(classification_report(labels, preds, target_names=["Ham", "Spam"]))




def run_model(text=None):
    """Run inference on a saved model and output the prediction + Top K words."""

    config = default_config.copy()

    text = "Congrats! Year, free money scams spam"

    # Load the Vocabulary object from the file
    filename = "cache/vocab.pkl"
    if os.path.exists(filename):
        with open(filename, "rb") as file:
            vocabulary = pickle.load(file)
    else:
        print("Missing Vocabulary file. Try rebuilding the dataset.")
        return

    # Tokenize the input text
    input = get_input_from_text(text, vocabulary)

    # Load the existing model
    model = TransformerClassifier.from_config(**config)
    model_state_path = "outputs/bce_model_tst.pth"
    model_state = torch.load(model_state_path)
    model.load_state_dict(model_state)
    model.eval()

    logits, attention_weights = model.forward(input, get_attn_weights=True)
    vals = torch.sigmoid(logits)

    # Calculate probabilities and Top K
    if vals[0] < 0.5:
        likelihood = (0.5 - vals[0]) * 200
        typ = "Not Spam"
    else:
        likelihood = (vals[0] - 0.5) * 200
        typ = "Spam"

    print(f"The message is classified as {typ} with a probability of {likelihood}%")

    top_k_words = get_top_k_words(
        input.squeeze(0).tolist(), attention_weights.squeeze(0).tolist()[1:], vocabulary
    )

    print(
        f"The top words that were used to determines this are: {",".join(top_k_words)}"
    )


main()
run_model()
