from bisturi.dataset import Dataset
from bisturi.model import LayerID, Direction
from bisturi.ontology import Concept
from bisturi.util.vecquantile import QuantileVector
from multiprocessing import Queue, Manager, Pool
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Union
import numpy as np
import torch

# Types
Alignment = Tuple[Direction, Concept]


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
             sigma * np.random.randn(1) + mu)
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
    return [(layer_id, basis[i, :], np.array([0])) for i in range(n_units)]


def _quantile_thresholds(activations: Union[Tuple[str, Tuple], np.ndarray],
                         directions: np.ndarray, quantile: float,
                         batch_size: int, seed: int,
                         queue: Queue) -> np.ndarray:

    # Eventually reload the memmap
    if isinstance(activations, tuple):
        activations = np.memmap(activations[0],
                                dtype=float,
                                mode='r',
                                shape=activations[1])

    # Init vector quantile
    quant = QuantileVector(depth=directions.shape[1], seed=seed)

    for i in range(0, activations.shape[0], batch_size):
        # Select batch
        batch = activations[i:i + batch_size]

        # Fully connected
        if len(batch.shape) == 2:
            batch = np.einsum('bn, nm -> bm', batch, directions)
        # Convolutional
        elif len(batch.shape) == 4:
            batch = np.einsum('bnhw, nm -> bmhw', batch, directions)
            batch = np.transpose(batch, axes=(0, 2, 3, 1)) \
                      .reshape(-1, activations.shape[1])

        # Update quantile vector
        quant.add(batch)

        # Notify batch end
        if queue:
            queue.put(1)

    # Notify procedure end
    if queue:
        queue.put(None)

    return quant.readout(1000)[:, int(1000 * (1-quantile)-1)]


def quantile_thresholds(activations: Dict[LayerID, np.ndarray],
                        directions: List[Direction],
                        quantile: float = 5e-3,
                        n_workers: int = 0,
                        batch_size: int = 128,
                        cache: str = None,
                        seed: int = 1) -> Dict[LayerID, np.ndarray]:
    """
    Parameters
    ----------
    activations: Dict[ModuleID, np.ndarray]
        Dictionary pointing to the
        array_like activations for
        each module.
    directions: List[Direction]
        List of directions to compute
        the thresholds for.
    quantile: float
        Quantile to consider the
        activations over 1-quantile
    n_workers: int, optional
        Number of workers to parallelize
        the computation of the thresholds
    batch_size: int, optional
        Batch size for the forward pass
    cache: str, optional
        Path of the folder in which to
        eventually store the activations
    seed: int, optional
        Seed for reproducibility

    Returns
    -------
    thresholds: Dict[ModuleID, np.ndarray]
        Dictionary mapping the module
        name to either a NumPy array
        or a memmap containing the
        threshold per unit
    """

    # Unique modules within all directions
    modules_ids = {d[0] for d in directions}

    # Module to direction dictionary
    module_to_dirs = {m: np.stack([d[1] for d in directions if d[0] == m]).T
                      for m in modules_ids}

    # Parallelization over the modules
    n_workers = min(n_workers, len(modules_ids))

    # Eventually init pool
    if n_workers > 0:
        pool = Pool(n_workers)
        manager = Manager()
        queue = manager.Queue()
    else:
        queue = None

    # Parameter per worker
    params = []
    for m_id in modules_ids:
        if cache:
            params.append(((activations[m_id].filename,
                            activations[m_id].shape),
                          module_to_dirs[m_id], quantile, batch_size,
                          seed, queue))
        else:
            params.append((activations[m_id], module_to_dirs[m_id], quantile,
                           batch_size, seed, queue))

    if n_workers > 0:
        # Map
        map_result = pool.starmap_async(_quantile_thresholds, params)

        # Total batches
        total = 0
        for m_id in modules_ids:
            total += np.ceil((activations[m_id].shape[0])/batch_size)
        pbar = tqdm(total=total)

        # Read updates from the workers
        ended = 0
        while ended != len(modules_ids):
            e = queue.get()
            if e is None:
                ended += 1
            else:
                pbar.update(e)
        pbar.close()

        partial = map_result.get()

        # Correctly close the pool
        pool.close()
        pool.join()
    else:
        partial = [_quantile_thresholds(*p) for p in params]

    # Reduce results
    thresholds = {m_id: partial[n] for n, m_id in enumerate(modules_ids)}

    # Assign bias to each direction
    biased_directions = []
    module_counter = {m: 0 for m in modules_ids}
    for d in directions:
        # Reconstruct direction
        new_dir = (d[0], d[1], thresholds[d[0]][module_counter[d[0]]])

        # Append new direction
        biased_directions.append(new_dir)

        # Increment counter
        module_counter[d[0]] += 1

    return biased_directions


def fit_directions_bias(directions: List[Direction],
                        activations: Dict[LayerID, np.ndarray],
                        style: str = 'quantile',
                        quantile: float = 5e-3,
                        cache: str = None,
                        nw: int = 0,
                        batch_size: int = 128,
                        verbose: bool = True) -> List[Direction]:
    """
    Set the bias of the given directions according
    to the activations of the module on a given dataset.

    Parameters
    ----------
    directions : List[Direction]
        Directions to fit the bias for.
    model : torch.nn.Module
        Model containing the directions.
    dataset : Dataset
        Dataset to exploit for activation analysis.
    style : str, optional
        How to compute the bias. Either 'quantile'
        or 'learned'.
    cache : str, optional
        Path of the folder in which to
        eventually retrieve the activations.
    nw : int, optional
        Number of workers to parallelize
        the computation.
    batch_size : int, optional
        Batch size for the forward pass.
    verbose : bool, optional
        Whether to print the progress bar.

    Returns
    -------
    List[Direction]
        The directions with fitted bias.
    """
    if style == 'learned':
        raise NotImplementedError
    elif style == 'quantile':
        directions = quantile_thresholds(activations, directions,
                                         quantile, n_workers=nw,
                                         batch_size=batch_size, cache=cache)
        return directions
    else:
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
