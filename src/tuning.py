import os
from typing import Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from interfaces import SupportsFromConfig
from results import BestResult
from utils.plots import plot_multiple_losses
from training.trainer import train_epoch, validate_epoch
from sklearn.metrics import precision_score, recall_score, f1_score

sweep_param = {
    "learning_rate": [5e-5, 1e-4, 2e-4, 3e-4],
    "embed_dim": [64, 128, 256],
    "dropout": [0.1, 0.3],
    "ff_dim": [128, 256],
    "num_heads": [2, 4],
}


def tuning(
    config: dict[str, Any],
    dataset,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    model_cls: SupportsFromConfig,
    loss_fn,
    optimizer_cls: type = torch.optim.AdamW,
    optimizer_kwargs: dict = None,
    sweep_label: str = "",
) -> BestResult:
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    base_output = "outputs"
    output_dir = os.path.join(base_output, sweep_label) if sweep_label else base_output
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    best_f1 = 0
    best_result = BestResult.init()

    for param_name, values in sweep_param.items():
        all_results = {}

        for val in values:
            label = f"{param_name}={val}"
            f1_scores = []
            val_losses_across_folds = []
            train_losses_across_folds = []

            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                current_config = config.copy()
                current_config[param_name] = val

                train_subset = Subset(dataset, train_idx)
                val_subset = Subset(dataset, val_idx)

                train_loader = DataLoader(
                    train_subset, batch_size=config["batch_size"], shuffle=True
                )
                val_loader = DataLoader(
                    val_subset, batch_size=config["batch_size"], shuffle=False
                )

                model = model_cls.from_config(**current_config).to(device)
                optimizer = optimizer_cls(
                    model.parameters(),
                    lr=current_config["learning_rate"],
                    **optimizer_kwargs,
                )
                model_num_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )

                train_losses, val_losses = [], []

                for epoch in range(config["num_epochs"]):
                    print(
                        f"\n[{label} | Fold {fold_idx+1}] Epoch {epoch + 1}/{config['num_epochs']}"
                    )

                    train_acc, train_loss = train_epoch(
                        model=model,
                        dataloader=train_loader,
                        optimizer=optimizer,
                        device=device,
                        loss_fn=loss_fn,
                    )

                    val_loss, val_acc, all_preds, all_labels = validate_epoch(
                        model=model,
                        dataloader=val_loader,
                        device=device,
                        loss_fn=loss_fn,
                    )

                    probs = torch.sigmoid(torch.tensor(all_preds))
                    binary_preds = (probs > 0.5).int().tolist()

                    precision = precision_score(
                        all_labels, binary_preds, zero_division=0
                    )
                    recall = recall_score(all_labels, binary_preds, zero_division=0)
                    f1 = f1_score(all_labels, binary_preds, zero_division=0)

                    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                    print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
                    print(
                        f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}"
                    )

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)

                    # Save best result (per fold, per value) if it improves
                    if f1 > best_f1:
                        best_f1 = f1
                        best_result = BestResult(
                            param=param_name,
                            f1=f1,
                            precision=precision,
                            recall=recall,
                            val_loss=val_loss,
                            train_loss=train_loss,
                            epoch=epoch + 1,
                            config=current_config,
                            sweep_param=label,
                            sweep_value=val,
                            num_params=model_num_params,
                            loss_label=current_config.get("loss_label", ""),
                        )

                f1_scores.append(f1)
                val_losses_across_folds.append(val_losses)
                train_losses_across_folds.append(train_losses)

            # Average + std across folds
            all_results[label] = (
                list(np.mean(train_losses_across_folds, axis=0)),
                list(np.mean(val_losses_across_folds, axis=0)),
                list(np.std(train_losses_across_folds, axis=0)),
                list(np.std(val_losses_across_folds, axis=0)),
            )

        plot_multiple_losses(
            all_results,
            title=f"{param_name.capitalize()} Sweep",
            save_path=os.path.join(output_dir, f"{param_name}_sweep.png"),
        )

    return best_result
