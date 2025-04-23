import pandas as pd
from new_train import train
from tuning import tuning
from utils.loss import (
    FocalLoss,
    binary_cross_entropy_loss,
    get_loss_wrapper,
    get_pos_weight,
    weighted_bce_with_logits_loss,
)
from model.config import default_config

import torch
from torch import nn
from TransformerClassifier import TransformerClassifier
from train import generate_train_test, evaluate
from sklearn.metrics import confusion_matrix, classification_report
import data_process


def main():
    # Download the data and generate train/test splits.
    dataset, vocab_size = data_process.build_data()

    # Train the model.
    config = default_config.copy()

    train_loader, valid_loader, test_loader = generate_train_test(
        dataset=dataset, batch_size=config["batch_size"]
    )

    # Initialize the model, optimizer, and crtierion.
    model = TransformerClassifier(
        vocab_size,
        config["embed_dim"],
        config["num_heads"],
        config["ff_dim"],
        config["dropout"],
        config["max_length"],
        config["num_encoder_layers"],
        config["class_hidden_dim"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    wrapped_loss = get_loss_wrapper(binary_cross_entropy_loss, apply_sigmoid=True)

    train(
        num_epochs=config["num_epochs"],
        train=train_loader,
        valid=valid_loader,
        model=model,
        optimizer=optimizer,
        device="cpu",
        loss_fn=wrapped_loss,
        save_path="outputs/bce_model.pth",
    )

    # # Tune the model

    # 1. grab config from config.py
    model_config = default_config.copy()

    # # binary cross entropy loss
    # tuning(
    #     model_config,
    #     dataset,
    #     TransformerClassifier,
    #     binary_cross_entropy_loss,
    # )

    # weighted binary cross entropy loss
    # Compute pos_weight for spam vs ham
    # pos_weight = get_pos_weight(dataset)

    # # Create loss function that handles imbalance
    # loss_fn = weighted_bce_with_logits_loss(pos_weight)

    # # Tuning
    # tuning(
    #     model_config,
    #     dataset,
    #     TransformerClassifier,
    #     loss_fn,
    # )

    # # Get base pos_weight
    # base_pos_weight = get_pos_weight(dataset)
    # results = []

    # # Try different scaled versions
    # for multiplier in [0.5, 1.0, 1.5, 2.0]:
    #     scaled_weight = base_pos_weight * multiplier
    #     print(
    #         f"\nRunning tuning with pos_weight={scaled_weight:.2f} (multiplier={multiplier})"
    #     )

    #     # Define loss function
    #     loss_fn = weighted_bce_with_logits_loss(scaled_weight)

    #     # Optional: label for plots/logs
    #     model_config["loss_label"] = f"pos_weight={scaled_weight:.2f}"

    #     best = tuning(
    #         config=model_config,
    #         dataset=dataset,
    #         model_cls=TransformerClassifier,
    #         loss_fn=loss_fn,
    #         sweep_label=f"pos_weight_{scaled_weight:.2f}",
    #     )

    #     best["pos_weight"] = scaled_weight

    #     results.append(best)
    # # Save all best configs
    # pd.DataFrame(results).sort_values("f1", ascending=False).to_csv(
    #     "outputs/pos_weight_sweep_summary.csv", index=False
    # )

    # # Focal loss tuning
    # alphas = [0.25, 0.5, 1.0]
    # gammas = [1.0, 2.0, 3.0]

    # results = []

    # for alpha in alphas:
    #     for gamma in gammas:
    #         label = f"focal_a{alpha}_g{gamma}"
    #         print(f"\n Tuning for alpha={alpha}, gamma={gamma}")
    #         loss_fn = FocalLoss(alpha=alpha, gamma=gamma)

    #         best = tuning(
    #             config=model_config,
    #             dataset=dataset,
    #             model_cls=TransformerClassifier,
    #             loss_fn=loss_fn,
    #             sweep_label=label,
    #         )

    #         best["alpha"] = alpha
    #         best["gamma"] = gamma
    #         results.append(best)

    # # Save all best configs
    # pd.DataFrame(results).sort_values("f1", ascending=False).to_csv(
    #     "outputs/focal_sweep_summary.csv", index=False
    # )


#     cm = confusion_matrix(labels, preds)
#     print(cm)


#     print(classification_report(labels, preds, target_names=["Ham", "Spam"]))
main()
