from bisturi.dataset import Dataset
from bisturi.model import LayerID
from bisturi.util.vecquantile import QuantileVector
from tqdm.auto import tqdm
from typing import List, Tuple
import numpy as np
import pickle
import torch


# Typings
Direction = Tuple[LayerID, np.ndarray, np.ndarray]


def random_directions(model: torch.nn.Module, layer_id: LayerID,
                      n_units: int, n_directions: int,
                      sigma: float = 1., mu: float = 0.) -> List[Direction]:
    """
    Generate random directions at a given layer.

    Parameters
    ----------
    model : torch.nn.Module
        The model to generate directions for.
    layer_id : LayerID
        The layer to generate directions for.
    n_directions : int
        The number of directions to generate.
    sigma : float, optional
        The standard deviation of the normal distribution
        to sample from.
    mu : float, optional
        The mean of the normal distribution to sample from.

    Returns
    -------
    List[Direction]
        The generated directions.
    """
    return [(layer_id,
             sigma * np.random.randn(n_units) + mu,
             sigma * np.random.randn(1).item() + mu)
            for _ in range(n_directions)]


def neuron_directions(model: torch.nn.Module, layer_id: LayerID,
                      n_units: int = None,
                      activations: np.ndarray = None,
                      quantile: float = 5e-3, batch_size: int = 128,
                      seed: int = 1, verbose: bool = True) -> List[Direction]:
    """
    Generate directions for each neuron in a given layer.
    The bias is computed by computing the quantile of
    the activations for each neuron as in NetworkDissection.

    Parameters
    ----------
    model : torch.nn.Module
        The model to generate directions for.
    layer_id : LayerID
        The layer to generate directions for.
    activations : np.ndarray
        The activations of the model on the given layer.
    quantile : float, optional
        The quantile to set the direction bias.
    batch_size : int, optional
        Batch size for the forward pass.
    seed : int, optional
        The seed to use for the random number generator.
    verbose : bool, optional
        Whether to print the progress bar.

    Returns
    -------
    List[Direction]
        The generated directions.
    """
    # Number of neurons
    if n_units is None:
        n_units = activations.shape[1]

    # Canonical basis
    basis = np.eye(n_units)

    # TODO: There are other interesting ways to compute the bias
    #       for instance it could be fit as a concept classifier

    # TODO: It would be nice to propose an external method to
    #       compute the bias for an arbitrary set of directions
    #       and not only during construction.

    # Compute the bias as the quantile of the activations
    quant = QuantileVector(depth=n_units, seed=seed)

    # Iterate over the activations
    for i in tqdm(range(0, activations.shape[0], batch_size),
                  disable=not verbose):
        # Select batch
        batch = activations[i:i + batch_size]

        # Convolutional batch must be reshaped
        if len(batch.shape) == 4:
            batch = np.transpose(batch, axes=(0, 2, 3, 1)) \
                      .reshape(-1, activations.shape[1])

        # Batch has now shape (n_examples, n_units)
        # where n_examples is the batch size
        # multiplied by the size of the feature map

        # TODO: it could be useful to sample only
        #       a portion of the n_examples

        # Update quantile vector
        quant.add(batch)

    # Recover quantile vector
    q = quant.readout(1000)[:, int(1000 * (1-quantile)-1)]

    # Bias is negative of the quantile
    bias = -q

    # Construct directions
    dirs = [(layer_id, basis[i, :], bias[i]) for i in range(n_units)]

    return dirs


def store_directions(directions: List[Direction],
                     path: str) -> None:
    """
    Store the directions in a given path.

    Parameters
    ----------
    directions : List[Direction]
        Directions to store.
    path : str
        Path of the folder in which to
        store the directions.
    """
    # Pickle directions
    with open(path, 'wb') as f:
        pickle.dump(directions, f)


def load_directions(path: str) -> List[Direction]:
    """
    Load the directions from a given path.

    Parameters
    ----------
    path : str
        Path of the folder in which to
        retrieve the directions.

    Returns
    -------
    List[Direction]
        The loaded directions.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def learn_directions(model: torch.nn.Module,
                     layer_id: LayerID,
                     dataset: Dataset) -> List[Direction]:
    """
    Learn directions for a given model
    according to the concepts represented
    in a given dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Model to containing the target layer.
    layer_id : LayerId
        Layer to learn directions for.
    dataset : Dataset
        Annotated dataset portraying visual concepts.

    Returns
    -------
    List[Direction]
        The learned directions.
    """
    raise NotImplementedError
