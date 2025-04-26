import matplotlib.pyplot as plt
import numpy as np
import os


def plot_multiple_losses(
    results_dict: dict[str, tuple[list[float], list[float], list[float], list[float]]],
    title: str,
    save_path: str,
):
    plt.figure(figsize=(10, 6))
    handles = []
    labels = []

    plt.axhline(
        y=2.07, color="gray", linestyle=":", linewidth=1, label="Random baseline"
    )

    for label, (train_mean, val_mean, train_std, val_std) in results_dict.items():
        epochs = list(range(len(train_mean)))

        (train_line,) = plt.plot(
            epochs,
            train_mean,
            label=f"{label} - train",
            linestyle="-",
            linewidth=2.5,
            alpha=0.8,
            marker="o",
            markersize=3,
        )
        (val_line,) = plt.plot(
            epochs,
            val_mean,
            label=f"{label} - val",
            linestyle="--",
            linewidth=2.5,
            alpha=0.8,
            marker="s",
            markersize=3,
        )

        plt.fill_between(
            epochs,
            np.array(train_mean) - np.array(train_std),
            np.array(train_mean) + np.array(train_std),
            alpha=0.2,
            color=train_line.get_color(),
        )

        plt.fill_between(
            epochs,
            np.array(val_mean) - np.array(val_std),
            np.array(val_mean) + np.array(val_std),
            alpha=0.2,
            color=val_line.get_color(),
        )

        min_val_epoch = np.argmin(val_mean)
        min_val = val_mean[min_val_epoch]
        plt.scatter(
            min_val_epoch, min_val, color=val_line.get_color(), marker="x", s=50
        )

        handles.extend([train_line, val_line])
        labels.extend([f"{label} - train", f"{label} - val"])

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title or "Loss Curves", fontsize=14)
    plt.legend(
        handles,
        labels,
        fontsize=9,
        ncol=2,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )
    plt.grid(True)

    all_losses = [v for pair in results_dict.values() for v in pair[0] + pair[1]]
    min_loss = min(all_losses)
    max_loss = max(all_losses)
    margin = (max_loss - min_loss) * 0.1 if max_loss > min_loss else 0.01

    plt.ylim(bottom=max(0, min_loss - margin), top=max_loss + margin)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_learning_curves(
    train_loss: list[float],
    val_loss: list[float],
    save_path: str,
    title: str = "Learning Curves",
):
    plt.figure(figsize=(8, 5))
    epochs = list(range(1, len(train_loss) + 1))

    plt.plot(epochs, train_loss, label="Train Loss", marker="o", linewidth=2)
    plt.plot(
        epochs, val_loss, label="Val Loss", marker="s", linestyle="--", linewidth=2
    )

    min_val_epoch = np.argmin(val_loss)
    min_val = val_loss[min_val_epoch]
    plt.scatter(
        min_val_epoch + 1, min_val, color="red", marker="x", s=80, label="Min Val Loss"
    )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
