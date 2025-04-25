import os
from interfaces import SupportsFromConfig
import torch
from train import generate_train_test
from utils.plots import plot_multiple_losses
from training.trainer import train_epoch, validate_epoch
from sklearn.metrics import precision_score, recall_score, f1_score

sweep_param = {
    "learning_rate": [5e-5, 1e-4, 2e-4, 3e-4]
    # "dropout": [0.1, 0.2, 0.3],
    # "embed_dim": [128, 256],
}


def tuning(
    config: dict[str, any],
    dataset,
    model_cls: SupportsFromConfig,
    loss_fn,
    optimizer_cls: type = torch.optim.AdamW,
    optimizer_kwargs: dict = None,
    sweep_label: str = "",
):
    """
    
    """
    
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    base_output = "outputs"
    output_dir = os.path.join(base_output, sweep_label) if sweep_label else base_output
    os.makedirs(output_dir, exist_ok=True)

    print("Tuning hyperparameters...")
    train, valid, _ = generate_train_test(
        dataset=dataset, batch_size=config["batch_size"]
    )

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    all_results = {}
    best_f1 = 0
    best_result = {
        "param": None,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "val_loss": float("inf"),
        "train_loss": float("inf"),
    }

    for param_name, values in sweep_param.items():
        for val in values:
            label = f"{param_name}={val}"
            current_config = config.copy()
            current_config[param_name] = val

            model = model_cls.from_config(**current_config).to(device)
            optimizer = optimizer_cls(
                model.parameters(),
                lr=current_config["learning_rate"],
                **optimizer_kwargs,
            )

            train_losses, val_losses = [], []

            for epoch in range(current_config["num_epochs"]):
                print(f"\n[{label}] Epoch {epoch + 1}/{current_config['num_epochs']}")
                train_acc, train_loss = train_epoch(
                    model=model,
                    dataloader=train,
                    optimizer=optimizer,
                    device=device,
                    loss_fn=loss_fn,
                )

                val_loss, val_acc, all_preds, all_labels = validate_epoch(
                    model=model,
                    dataloader=valid,
                    device=device,
                    loss_fn=loss_fn,
                )

                probs = torch.sigmoid(torch.tensor(all_preds))
                binary_preds = (probs > 0.5).int().tolist()

                precision = precision_score(all_labels, binary_preds, zero_division=0)
                recall = recall_score(all_labels, binary_preds, zero_division=0)
                f1 = f1_score(all_labels, binary_preds, zero_division=0)

                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
                print(
                    f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}"
                )

                # Save best metrics by F1 score
                if f1 > best_f1:
                    best_f1 = f1
                    best_result.update(
                        {
                            "param": label,
                            "f1": f1,
                            "precision": precision,
                            "recall": recall,
                            "val_loss": val_loss,
                            "train_loss": train_loss,
                        }
                    )

                train_losses.append(train_loss)
                val_losses.append(val_loss)

            all_results[label] = (train_losses, val_losses)

        # Save the loss curve
        plot_multiple_losses(
            all_results,
            title=f"{param_name.capitalize()} Sweep",
            save_path=os.path.join(output_dir, f"{param_name}_sweep.png"),
        )

    return best_result
