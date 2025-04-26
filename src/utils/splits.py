from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List
import numpy as np


def generate_stratified_splits(
    dataset: Dataset, batch_size: int, seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    indices = np.arange(len(dataset))

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, targets, test_size=0.3, stratify=targets, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=y_temp, random_state=seed
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=batch_size, shuffle=False
    )

    return train_loader, valid_loader, test_loader


def generate_kfold_splits(
    dataset: Dataset, k: int = 5, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    return list(skf.split(np.zeros(len(targets)), targets))
