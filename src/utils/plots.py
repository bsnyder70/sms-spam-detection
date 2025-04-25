import matplotlib.pyplot as plt
import numpy as np
import os


def plot_multiple_losses(
    results_dict: dict[str, tuple[list[float], list[float]]],
    title: str,
    save_path: str,
):
    plt.figure(figsize=(10, 6))

    handles = []
    labels = []

    plt.axhline(
        y=2.07, color="gray", linestyle=":", linewidth=1, label="Random baseline"
    )

    for label, (train_loss, val_loss) in results_dict.items():
        epochs = list(range(len(train_loss)))

        (train_line,) = plt.plot(
            epochs,
            train_loss,
            label=f"{label} - train",
            linestyle="-",
            linewidth=2.5,
            alpha=0.8,
            marker="o",
            markersize=3,
        )
        (val_line,) = plt.plot(
            epochs,
            val_loss,
            label=f"{label} - val",
            linestyle="--",
            linewidth=2.5,
            alpha=0.8,
            marker="s",
            markersize=3,
        )

        min_val_epoch = np.argmin(val_loss)
        min_val = val_loss[min_val_epoch]
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

    # === Dynamic Y-axis adjustment ===
    all_losses = [loss for pair in results_dict.values() for loss in pair[0] + pair[1]]
    min_loss = min(all_losses)
    max_loss = max(all_losses)
    margin = (max_loss - min_loss) * 0.1 if max_loss > min_loss else 0.01

    plt.ylim(
        bottom=max(0, min_loss - margin),
        top=max_loss + margin,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
