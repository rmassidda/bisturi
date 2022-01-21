from bisturi.dataset import Dataset
from bisturi.ontology import Concept
from bisturi.model import LayerID, Direction
from typing import List, Tuple
import torch

# Types
Alignment = Tuple[Direction, Concept]


def random_directions(model: torch.nn.Module,
                      layer_id: LayerID,
                      n_directions: int) -> List[Direction]:
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

    Returns
    -------
    List[Direction]
        The generated directions.
    """
    raise NotImplementedError


def neuron_directions(model: torch.nn.Module,
                      layer_id: LayerID) -> List[Direction]:
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
    raise NotImplementedError


def fit_bias(dirs: List[Direction],
             model: torch.nn.Module,
             dataset: Dataset,
             style: str = 'Quantile') -> List[Direction]:
    """
    Set the bias of the given directions according
    to the activations of the module on a given dataset.

    Parameters
    ----------
    dirs : List[Direction]
        Directions to fit the bias for.
    model : torch.nn.Module
        Model containing the directions.
    dataset : Dataset
        Dataset to exploit for activation analysis.

    Returns
    -------
    List[Direction]
        The directions with fitted bias.
    """
    raise NotImplementedError


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


def semalign(model: torch.nn.Module,
             dirs: List[Direction],
             dataset: Dataset,
             measure: str = 'IoU') -> List[Tuple[Alignment, float]]:
    """
    Semantically align the given directions
    according to the concepts represented
    in a given dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Model containing the directions.
    dirs : List[Direction]
        Directions to align.
    dataset : Dataset
        Annotated dataset portraying visual concepts.
    measure : str
        The measure to use for alignment.

    Returns
    -------
    psi : List[Tuple[Alignment, float]]
        Estimate of the alignment.
    """
    raise NotImplementedError
