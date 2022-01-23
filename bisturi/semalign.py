from PIL import Image
from enum import Enum, auto
from functools import partial
from bisturi.model import get_module, ModuleID
from bisturi.util.vecquantile import QuantileVector
from bisturi.ontology import Ontology, Concept
from multiprocessing import Pool, Manager, Queue
from typing import Union, List, Tuple, Dict
from tqdm.auto import tqdm
import numpy as np
import os
import torch


def collate_masks(batch):
    """
    Custom function to collate both images and
    concept masks, to be used with the DataLoader.
    """
    images = [x for x, y in batch]

    if None in images:
        images = None
    else:
        images = torch.utils.data._utils.collate.default_collate(images)

    masks = [y for x, y in batch]
    if None in masks:
        masks = None
    return images, masks


def moduleid_to_string(module_id):
    if isinstance(module_id, tuple):
        return '%s#%d' % module_id
    else:
        return module_id


def is_convolutional(batch):
    '''
    Checks if the batch has been produced
    by a convolutional layer
    '''
    return len(batch.shape) == 4


def record_activations(model: torch.nn.Module,
                       modules_ids: List[ModuleID],
                       dataset: torch.utils.data.Dataset,
                       batch_size: int = 128,
                       cache: str = None,
                       gpu: bool = False,
                       silent: bool = False) -> Dict[ModuleID, np.ndarray]:
    """
    Parameters
    ----------
    model: torch.nn.Module
        PyTorch model to analyze
    modules_id: List[ModuleID]
        Coordinates of the modules to analyze
    dataset: torch.utils.data.Dataset
        Dataset containing the inputs
    batch_size: int, optional
        Batch size for the forward pass
    cache: str, optional
        Path of the folder in which to
        eventually store the activations
    gpu: bool, optional
        Flag to handle GPU usage
    silent: bool, optional
        Disables the progress bar

    Returns
    -------
    activations: Dict[ModuleID, np.ndarray]
        Dictionary mapping the module
        id to either a NumPy array
        or a memmap containing the
        activations per input and
        per unit
    """

    activations = {}
    act_size = {}

    # normalize module ids
    modules_ids = [(m, 0) if isinstance(m, str) else m for m in modules_ids]

    # module ids to string
    modules_str = [moduleid_to_string(m) for m in modules_ids]

    # eventually load from file
    if cache:
        # skip network forward pass
        skip = True

        # shape filenames
        shape_filenames = {m_id: os.path.join(cache, "size_%s.npy" % m_str)
                           for m_id, m_str in zip(modules_ids, modules_str)}

        # activations filenames
        act_filenames = {m_id: os.path.join(cache, "act_%s.mmap" % m_str)
                         for m_id, m_str in zip(modules_ids, modules_str)}

        # load from file
        for m_id in modules_ids:
            s_fn = shape_filenames[m_id]
            a_fn = act_filenames[m_id]
            if os.path.exists(s_fn) and os.path.exists(a_fn):
                act_size[m_id] = np.load(s_fn)
                activations[m_id] = np.memmap(a_fn,
                                              dtype=float,
                                              mode='r',
                                              shape=tuple(act_size[m_id]))
            else:
                skip = False

        # All the activations are on disk
        if skip:
            return activations

    # disable concept masks retrieval
    was_skipping_masks = dataset.skip_masks
    dataset.skip_masks = True

    # fix batch size for the image loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         collate_fn=collate_masks)

    # retrieve modules from the model
    modules_names = list({m_name for m_name, m_idx in modules_ids})
    modules = [get_module(model, m_name) for m_name in modules_names]

    # init recording variables
    recording = {m_name: [] for m_name, m_idx in modules_ids}

    # Define generic hook
    def hook_feature(module, input, output, module_name):
        recording[module_name].append(output.detach().cpu().numpy())

    # Register hooks
    hooks = [partial(hook_feature, module_name=m_name)
             for m_name in modules_names]
    hooks = [module.register_forward_hook(h)
             for module, h in zip(modules, hooks)]

    # keep track of the model status
    was_training = model.training
    model.eval()

    # batch iteration over the inputs
    first_batch = True
    for batch_idx, batch in enumerate(tqdm(loader, total=len(loader),
                                           disable=silent)):

        # Eventually ignore masks
        if isinstance(batch, tuple):
            batch = batch[0]

        # Delete previous recording
        keys = list(recording.keys())
        for key in keys:
            del recording[key][:]

        # Prepare input batch
        if gpu:
            batch = batch.cuda()

        # Forward pass of the input
        with torch.no_grad():
            _ = model.forward(batch)

        # initialize tensors
        if first_batch:
            for m_id in modules_ids:
                m_name, m_idx = m_id
                act_size[m_id] = (len(dataset),
                                  *recording[m_name][m_idx].shape[1:])
                if cache:
                    s_fn = shape_filenames[m_id]
                    a_fn = act_filenames[m_id]
                    np.save(s_fn, act_size[m_id])
                    activations[m_id] = np.memmap(a_fn,
                                                  dtype=float,
                                                  mode='w+',
                                                  shape=act_size[m_id])
                else:
                    activations[m_id] = np.zeros(act_size[m_id])

            # Do not repeat the initialization
            first_batch = False

        # copy activations
        start_idx = batch_idx*loader.batch_size
        end_idx = min((batch_idx+1)*loader.batch_size, len(dataset))
        for m_id in modules_ids:
            m_name, m_idx = m_id
            activations[m_id][start_idx:end_idx] = recording[m_name][m_idx]

    # revert model status
    if was_training:
        model.train()

    # revert dataset preference
    dataset.skip_masks = was_skipping_masks

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations


def _compute_thresholds(activations: Union[Tuple[str, Tuple], np.ndarray],
                        quantile: float, batch_size: int, seed: int,
                        queue: Queue) -> np.ndarray:

    # Eventually reload the memmap
    if isinstance(activations, tuple):
        activations = np.memmap(activations[0],
                                dtype=float,
                                mode='r',
                                shape=activations[1])

    quant = QuantileVector(depth=activations.shape[1], seed=seed)

    for i in range(0, activations.shape[0], batch_size):
        batch = activations[i:i + batch_size]
        # Convolutional batch must be reshaped
        if is_convolutional(batch):
            batch = np.transpose(batch, axes=(0, 2, 3, 1)) \
                      .reshape(-1, activations.shape[1])
        quant.add(batch)

        if queue:
            queue.put(1)

    if queue:
        queue.put(None)

    return quant.readout(1000)[:, int(1000 * (1-quantile)-1)]


def compute_thresholds(activations: Dict[ModuleID, np.ndarray],
                       quantile: float = 5e-3,
                       n_workers: int = 0,
                       batch_size: int = 128,
                       cache: str = None,
                       seed: int = 1) -> Dict[ModuleID, np.ndarray]:
    """
    Parameters
    ----------
    activations: Dict[ModuleID, np.ndarray]
        Dictionary pointing to the
        array_like activations for
        each module.
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

    # retrieve ids from keys
    modules_ids = list(activations.keys())

    # normalize module ids
    modules_ids = [(m, 0) if isinstance(m, str) else m for m in modules_ids]

    # module ids to string
    modules_str = [moduleid_to_string(m) for m in modules_ids]

    # Load from file
    thresholds = {}

    if cache:
        for m_str, m_id in zip(modules_str, modules_ids):
            m_path = os.path.join(cache, 'thresholds_%s.npy' % m_str)
            try:
                thresholds[m_id] = np.load(m_path)
            except FileNotFoundError:
                pass

    # Filter out modules already computed
    modules_ids = [m_id for m_id in modules_ids if m_id not in thresholds]
    modules_str = [moduleid_to_string(m_id) for m_id in modules_ids]

    # Check if there is something to compute
    if not modules_ids:
        return thresholds

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
                          quantile, batch_size, seed, queue))
        else:
            params.append((activations[m_id], quantile, batch_size, seed,
                          queue))

    if n_workers > 0:
        # Map
        map_result = pool.starmap_async(_compute_thresholds, params)

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
        partial = [_compute_thresholds(*p) for p in params]

    # Reduce results
    partial = {m_id: partial[n] for n, m_id in enumerate(modules_ids)}
    thresholds = {**thresholds, **partial}

    if cache:
        for m_id, m_str in zip(modules_ids, modules_str):
            qtpath = os.path.join(cache, 'thresholds_%s.npy' % m_str)
            np.save(qtpath, thresholds[m_id])

    return thresholds


def _compute_sigma(activations: Union[Tuple[str, Tuple], np.ndarray],
                   dataset: torch.utils.data.Dataset,
                   directions: np.ndarray,
                   concepts: List[Concept],
                   thresholds: np.ndarray, queue: Queue,
                   batch_size: int, start: int, end: int) -> np.ndarray:

    # Eventually reload the memmap
    if isinstance(activations, tuple):
        activations = np.memmap(activations[0],
                                dtype=float,
                                mode='r',
                                shape=activations[1])

    # Without custom directions adopt canonical basis
    # TODO: keep directions = None and exploit this later
    if not directions:
        directions = np.eye(activations.shape[1])

    # Number of directions and concepts to align
    n_directions = directions.shape[0]
    n_concepts = len(concepts)

    # Check number of directions and thresholds
    if n_directions != thresholds.shape[0]:
        raise ValueError('Number of directions and thresholds do not match')

    # Eventually allocate arrays
    intersection = np.zeros((n_directions, n_concepts))
    act_sum = np.zeros((n_directions))
    cmask_sum = np.zeros((n_concepts))

    # Ignore images
    dataset.skip_image = True

    # Create subset
    dataset = torch.utils.data.Subset(dataset, range(start, end))

    # Init loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         collate_fn=collate_masks)

    # Show progress bar when not multiprocessing
    if queue is None:
        loader = tqdm(loader)

    first = True
    for batch in loader:

        # Ignore images
        if isinstance(batch, tuple):
            batch = batch[1]

        if first:
            a_mask = np.full((n_directions, *batch[0].shape), False)
            c_mask = np.full(batch[0].shape, False)
            first = False

        for image in batch:

            # Precompute directional activations
            dir_act = [(activations[image.index].T @ directions[i]).T
                       for i in range(n_directions)]

            # Only keep valid directions
            # NOTE: is any faster than max?
            valid_dirs = [d_idx for d_idx in range(n_directions)
                          if np.any(dir_act[d_idx] > thresholds[d_idx])]

            # Generate activation masks
            for d_idx in valid_dirs:
                # Retrieve directional activations
                tmp_a_mask = dir_act[d_idx]

                # Resize if convolutional
                if len(tmp_a_mask.shape):
                    tmp_a_mask = Image.fromarray(tmp_a_mask) \
                        .resize(image.shape,
                                resample=Image.BILINEAR)
                # Create mask
                a_mask[d_idx] = tmp_a_mask > thresholds[d_idx]

                # Update \sum_x |M_u(x)|
                act_sum[d_idx] += np.count_nonzero(a_mask[d_idx])

            # Retrieve concepts in the image
            selected_concepts = image.select_concepts(concepts)

            for c_idx, concept in enumerate(concepts):

                if concept in selected_concepts:
                    # retrieve L_c(x)
                    c_mask = image.get_concept_mask(concept, c_mask)

                    # update \sum_x |L_c(x)|
                    cmask_sum[c_idx] += np.count_nonzero(c_mask)

                    # Update counters
                    for d_idx in valid_dirs:

                        # |M_u(x) && L_c(x)|
                        intersection[d_idx, c_idx] += np.count_nonzero(
                            np.logical_and(a_mask[d_idx], c_mask))

        # Notify end of batch
        if queue:
            queue.put(1)

    if queue:
        queue.put(None)

    # |M_u(x) || L_c(x)|
    union = act_sum[:, None] + cmask_sum[None, :] - intersection

    return intersection, union, act_sum, cmask_sum


class SigmaMeasure(Enum):
    """
    Possible semantic alignment measures.
    """
    IOU = auto()
    LIKELIHOOD = auto()
    NIOU = auto()
    FULL_NIOU = auto()
    RAW = auto()


def compute_sigma(activations: Dict[ModuleID, np.ndarray],
                  dataset: torch.utils.data.Dataset,
                  thresholds: Dict[ModuleID, np.ndarray],
                  ontology: Ontology,
                  directions: Dict[ModuleID, np.ndarray] = None,
                  n_workers: int = 1,
                  cache: str = None,
                  batch_size: int = 32,
                  measure: SigmaMeasure = SigmaMeasure.LIKELIHOOD,
                  n_concepts: int = None,
                  module: ModuleID = None) -> Dict[ModuleID, np.ndarray]:
    """
    Given the activations patterns and the
    thresholds, it computes the IOU metric
    to estimate the semantic alignment
    between units and concepts.

    Parameters
    ----------
    activations: dict or array_like
        Dictionary pointing to the
        array_like activations for
        each module.
    dataset: torch.utils.data.Dataset
        Callable iterator to retrieve
        the dataset for each image
        in the dataset.
    thresholds: dict or array_like
        Dictionary pointing to the
        array_like thresholds for
        each module.
    ontology: Ontology
        Ontological structure of the
        concepts in the dataset
    n_workers: int, optional
        Number of workers for parallel
        computation of the IoU
    cache: str, optional
        Path of the folder in which to
        eventually store the activations
    batch_size: int, optional
        Batch size for the forward pass
    style: str, optional
        Determines what is returned by
        the method, admitted values are:
        iou and niou, for the normalized
        metric
    module_name: str, optional
        Name of the module to analyze if
        activations and thresholds are
        dictionaries. If none, all of the
        modules are analyzed.

    Returns
    -------
    alignment: Dict[Tuple[ModuleID, int], Concept], float]
        Dictionary of the semantic alignment
        between directions and concepts.
    """

    # Separately handle different modules
    if isinstance(activations, dict) and not module:
        assert isinstance(thresholds, dict)
        assert activations.keys() == thresholds.keys()

        # Default directions
        if directions is None:
            directions = {module: None for module in activations.keys()}

        def build_args(module):
            return (activations[module], dataset, thresholds[module],
                    ontology, directions[module], n_workers, cache, batch_size,
                    measure, n_concepts, module)

        return {mod: compute_sigma(*build_args(mod)) for mod in activations}

    # Build path
    fname = 'semalign_' + measure.name + '_' \
        + moduleid_to_string(module) + '.npy'

    # Persistency on disk
    if cache:
        try:
            alignment = np.load(os.path.join(cache, fname))
            return alignment
        except FileNotFoundError:
            pass

    # Directions to analyze
    if not directions:
        n_directions = activations.shape[1]
    else:
        n_directions = directions.shape[0]

    # Images in the dataset
    n_images = activations.shape[0]

    # Alignable concepts
    concepts = ontology.to_list(keep_placeholders=False)
    n_concepts = len(concepts)

    # Keep track of the best concepts per unit
    alignment = np.zeros((n_directions, n_concepts))

    # Allocate partial arrays for parallelism
    # NOTE: this could be shared since it is
    #       used sequentially for different modules
    if n_workers > 0:
        pool = Pool(n_workers)

    # Cache leaves
    for node in concepts:
        node.cache_leaves()

    # Compute intersection and union arrays
    if n_workers > 0:
        psize = int(np.ceil(float(n_images) / n_workers))
        ranges = [(s, min(n_images, s + psize)) for s
                  in range(0, n_images, psize) if s < n_images]

        # Queue to handle progress
        manager = Manager()
        queue = manager.Queue()

        # Parameter per worker
        params = []
        for i, r in enumerate(ranges):
            # FIX: might be late binding.. again!
            params.append(((activations.filename, activations.shape),
                          dataset, directions, concepts, thresholds, queue,
                          batch_size, *r))

        # Map
        map_result = pool.starmap_async(_compute_sigma, params)

        # Total batches
        total = 0
        for r in ranges:
            total += np.ceil((r[1]-r[0])/batch_size)
        pbar = tqdm(total=total)

        # Read updates from the workers
        ended = 0
        while ended != n_workers:
            e = queue.get()
            if e is None:
                ended += 1
            else:
                pbar.update(e)

        pbar.close()

        partial = map_result.get()

        # Reduce
        intersection = np.sum([e[0] for e in partial], axis=0)
        union = np.sum([e[1] for e in partial], axis=0)
        act_sum = np.sum([e[2] for e in partial], axis=0)
        cmask_sum = np.sum([e[3] for e in partial], axis=0)
    else:
        results = _compute_sigma(activations, dataset, directions, concepts,
                                 thresholds, None, batch_size, 0, len(dataset))
        intersection, union, act_sum, cmask_sum = results

    # Compute IoU and find the best keep_n results
    if measure == SigmaMeasure.LIKELIHOOD:
        alignment = intersection / (cmask_sum[None, :] + 1e-12)
    else:
        alignment = intersection / (union + 1e-12)
        if measure == SigmaMeasure.NIOU:
            alignment = alignment / (cmask_sum[None, :] + 1e-12)
        elif measure == SigmaMeasure.FULL_NIOU:
            alignment = alignment * (act_sum[:, None] /
                                     (cmask_sum[None, :] + 1e-12))

    # Correctly close the pool
    if n_workers > 0:
        pool.close()
        pool.join()

    # Persistency on disk
    if cache:
        np.save(os.path.join(cache, fname), alignment)

    return alignment


def compute_psi(alignment: Dict[ModuleID, np.ndarray],
                ontology: Ontology,
                tau: float = 0.2) \
        -> Dict[ModuleID, Dict[int, List[Tuple[Concept, float]]]]:
    """
    Given the semantic alignment, this function
    returns, for each direction, the concepts
    overcoming a given $\tau$ threshold.
    """

    concepts = ontology.to_list(keep_placeholders=False)
    psi = {}

    # Iterate over modules
    for module in alignment:
        psi[module] = {}
        # Iterate over directions
        n_direction = alignment[module].shape[0]
        for d_idx in range(n_direction):
            psi[module][d_idx] = []
            # Iterate over concepts
            for c_idx, c in enumerate(concepts):
                # Check if σ(u,c) > τ and assign to ψ(u)
                if alignment[module][d_idx, c_idx] > tau:
                    psi[module][d_idx].append((c, alignment[module][d_idx,
                                                                    c_idx]))
            psi[module][d_idx] = sorted(psi[module][d_idx], key=lambda e: e[1])

    return psi


def retrieve_concepts(alignment: Dict[ModuleID, np.ndarray],
                      ontology: Ontology) \
        -> Dict[ModuleID, Dict[int, List[Tuple[Concept, float]]]]:
    """
    Produces a dictionary accessible
    per concepts from an array
    containing the alignment per
    direction and concept numerical ID.
    """
    return compute_psi(alignment, ontology, tau=0.0)


def report_psi(psi: Dict[ModuleID, Dict[int, List[Tuple[Concept, float]]]],
               module: ModuleID = None) -> Dict[ModuleID, Dict]:
    """
    Return per-module statistics on
    the semantic alignment with the
    ontology.
    """
    if module is None:
        # Iterate over all modules
        partial = [report_psi(psi, mod) for mod in psi]

        # Join the alignment of all modules
        aggregate = {}
        for module in psi:
            shift = max(aggregate.keys(), default=0) + 1
            extension = {u + shift: psi[module][u] for u in psi[module]}
            aggregate.update(extension)

        # Return statistics for module and aggregated
        return partial + [report_psi({'Aggregated': aggregate}, 'Aggregated')]

    all_concepts = sum([[c for c, v in psi[module][u]]
                        for u in psi[module]], [])
    all_values = sum([[v for c, v in psi[module][u]] for u in psi[module]], [])
    unique_concepts = set(all_concepts)
    return {
        'Module': module,
        'Units': len(psi[module]),
        'Unique concepts': len(unique_concepts),
        'Leaves': len([c for c in unique_concepts if c.is_leaf()]),
        'Non-leaves': len([c for c in unique_concepts if not c.is_leaf()]),
        'Total concepts': len(all_concepts),
        'Size': np.average([len(psi[module][u]) for u in psi[module]]),
        'Size if any': np.average([len(psi[module][u]) for u
                                   in psi[module] if len(psi[module][u])]),
        'Depth': np.average([c.depth for c in unique_concepts]),
        'Sigma': np.average(all_values)
    }
