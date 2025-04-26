import os
import pickle
import pandas as pd
from analyze.evaluate_model import (
    plot_prediction_confidence,
)
from train import train
from tuning import tuning
from utils.loss import (
    FocalLoss,
    binary_cross_entropy_loss,
    get_loss_wrapper,
    get_pos_weight,
    weighted_bce_with_logits_loss,
)
from model.config import default_config, focal_best_config

import torch
from model.TransformerClassifier import TransformerClassifier
from data_process import get_input_from_text, Vocabulary, build_data, get_top_k_words
from utils.splits import generate_kfold_splits, generate_stratified_splits


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


def main():

    # ale experiment section
    run_training()
    # run_tuning()


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


# main()
run_model()
