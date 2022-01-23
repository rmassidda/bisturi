from typing import Tuple
import numpy as np
import torch


def balanced_splits(weights: torch.Tensor, n_samples: int,
                    dataset: torch.utils.data.Dataset = None,
                    cut_point: float = 0.8) \
                    -> Tuple[torch.utils.data.Dataset,
                             torch.utils.data.Dataset,
                             torch.utils.data.Dataset]:
    # Weighted Sample
    sample_indeces = list(torch.utils.data.WeightedRandomSampler(
        weights, n_samples, replacement=False))

    # Shuffle list
    sample_indeces = np.random.permutation(sample_indeces)

    # Training set
    train_indeces = sample_indeces[:int(n_samples * cut_point)]

    # Validation set
    val_indeces = sample_indeces[int(n_samples * cut_point):]

    if dataset:
        train_set = torch.utils.data.Subset(dataset, train_indeces)
        val_set = torch.utils.data.Subset(dataset, val_indeces)
        subset = torch.utils.data.Subset(dataset, sample_indeces)
        return train_set, val_set, subset
    else:
        return list(train_indeces), list(val_indeces), list(sample_indeces)
