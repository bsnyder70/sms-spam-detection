from torch import nn
import torch
from typing import Union
from torch.utils.data import Dataset


#  Binary Cross Entropy Loss
def binary_cross_entropy_loss(
    predictions: torch.Tensor, targets: torch.Tensor
) -> float:
    """
    Computes the binary cross-entropy loss between predictions and targets.
    """
    bce_loss = nn.BCELoss()
    return bce_loss(predictions, targets)


def get_pos_weight(dataset: Dataset) -> float:
    """
    Computes the positive class weight to handle imbalance in binary classification.
    """
    labels: Union[torch.Tensor, list[int]] = dataset.labels

    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()

    pos = sum(labels)
    neg = len(labels) - pos
    return neg / pos


# Weighted BCE with Logits Loss
def weighted_bce_with_logits_loss(pos_weight: float):
    """
    Returns a callable BCEWithLogitsLoss function with class imbalance handling.
    """

    def loss_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight_tensor = torch.tensor(
            [pos_weight], dtype=torch.float, device=predictions.device
        )
        loss = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        return loss(predictions, targets)

    return loss_fn


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="none"
        )
        probas = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probas, 1 - probas)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# Focal Loss Factory Function
def focal_loss(alpha=1.0, gamma=2.0, reduction="mean"):
    """
    Returns an instance of FocalLoss.

    Args:
        alpha (float): Scaling factor for positive examples.
        gamma (float): Focusing parameter.
        reduction (str): 'mean', 'sum', or 'none'.

    Returns:
        Callable: FocalLoss instance.
    """
    return FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)


def get_loss_wrapper(loss_fn, apply_sigmoid=False):
    """
    Wraps a loss function to optionally apply sigmoid before calling.
    This makes it easier to handle BCELoss separately.
    """

    def wrapped(preds, targets):
        if apply_sigmoid:
            preds = torch.sigmoid(preds)
        return loss_fn(preds, targets)

    return wrapped
