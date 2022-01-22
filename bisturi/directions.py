from bisturi.model import LayerID
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
                      n_units: int) -> List[Direction]:
    """
    Generate directions for each neuron in a given layer.

    Parameters
    ----------
    model : torch.nn.Module
        The model to generate directions for.
    layer_id : LayerID
        The layer to generate directions for.

    Returns
    -------
    List[Direction]
        The generated directions.
    """
    basis = np.eye(n_units)
    return [(layer_id, basis[i, :], 0.) for i in range(n_units)]


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
