from PIL import Image
from bisturi.dataset import Dataset, ConceptMask
from bisturi.directions import Direction
from bisturi.model import LayerID
from bisturi.ontology import Concept
from multiprocessing import Queue, Manager, Pool
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Union
import numpy as np

# Types
Alignment = Tuple[Direction, Concept]


def _semalign_batch(act_batch: np.ndarray,
                    images: List[ConceptMask],
                    concepts: List[Concept],
                    directions: np.ndarray,
                    bias: np.ndarray,
                    cmask_sum: np.ndarray,
                    act_sum: np.ndarray,
                    intersection: np.ndarray,
                    a_mask: np.ndarray = None,
                    c_mask: np.ndarray = None) \
                            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Estimates semantic alignment for
    a given batch of annotated images.
    """
    for idx, image in enumerate(images):

        # Directional activations
        if len(act_batch.shape) == 2:
            # Fully connected
            dir_act = np.einsum('mn, n -> m', directions,
                                act_batch[idx])
        elif len(act_batch.shape) == 4:
            # Convolutional
            dir_act = np.einsum('mn, nhw -> mhw', directions,
                                act_batch[idx])

        # Directional mask
        dir_mask = dir_act + bias > 0

        # Only keep valid directions
        valid_dirs = [d_idx for d_idx in range(directions.shape[0])
                      if np.any(dir_mask[d_idx] > 0)]

        # Generate activation masks
        for d_idx in valid_dirs:
            # Retrieve directional activations
            tmp_a_mask = dir_act[d_idx]

            # Resize if convolutional
            # TODO: handle other ways to match concept/activation masks
            if len(tmp_a_mask.shape):
                tmp_a_mask = Image.fromarray(tmp_a_mask) \
                    .resize(image.shape,
                            resample=Image.BILINEAR)
            # Create mask
            a_mask[d_idx] = tmp_a_mask > -bias[d_idx]

            # Update \sum_x |M_u(x)|
            act_sum[d_idx] += np.count_nonzero(a_mask[d_idx])

        # Retrieve annotated concepts
        selected_concepts = image.select_concepts(concepts)

        # Update mask intersection
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

    return intersection, act_sum, cmask_sum


def _semalign(activations: Union[Tuple[str, Tuple], np.ndarray],
              dataset: Dataset, directions: np.ndarray,
              bias: np.ndarray, queue: Queue,
              batch_size: int, start: int, end: int) -> np.ndarray:

    # Eventually reload the memmap
    if isinstance(activations, tuple):
        activations = np.memmap(activations[0],
                                dtype=float,
                                mode='r',
                                shape=activations[1])

    # List of concepts from the ontology
    concepts = dataset.ontology.to_list()

    # Number of directions and concepts to align
    n_directions = directions.shape[0]
    n_concepts = len(concepts)

    # Ignore images
    dataset.skip_image = True

    # Init loader
    loader = range(start, end, batch_size)

    # Eventually allocate arrays
    intersection = np.zeros((n_directions, n_concepts))
    act_sum = np.zeros((n_directions))
    cmask_sum = np.zeros((n_concepts))

    # Pre-allocate activation mask
    if len(activations.shape) == 2:
        a_mask = np.full((n_directions), False)
    elif len(activations.shape) == 4:
        a_mask = np.full((n_directions, *dataset[0][2].shape), False)
        bias = bias.reshape(bias.shape[0], 1, 1)

    # Pre-allocate concept mask
    c_mask = np.full(dataset[0][2].shape, False)

    # Show progress bar when not multiprocessing
    if queue is None:
        loader = tqdm(loader)

    for idx in loader:
        end_r = min(idx + batch_size, end)

        # Select activations
        act_batch = activations[idx:idx + end_r]

        # Select images
        images = [dataset[i][2] for i in range(idx, end_r)]

        # Batch semantic alignment
        intersection, act_sum, cmask_sum = _semalign_batch(act_batch, images,
                                                           concepts,
                                                           directions, bias,
                                                           cmask_sum, act_sum,
                                                           intersection,
                                                           a_mask, c_mask)

        # Notify end of batch
        if queue:
            queue.put(1)

    # Notify end of worker
    if queue:
        queue.put(None)

    # |M_u(x) || L_c(x)|
    union = act_sum[:, None] + cmask_sum[None, :] - intersection

    return intersection, union, act_sum, cmask_sum


def semalign(activations: Dict[LayerID, np.ndarray],
             directions: List[Direction],
             dataset: Dataset,
             n_workers: int = 0,
             batch_size: int = 32,
             measure: str = 'IoU',
             module: LayerID = None,
             verbose: bool = True) -> List[Tuple[Alignment, float]]:
    """
    Semantically align the given directions
    according to the concepts represented
    in a given dataset.

    Parameters
    ----------
    activations : Dict[LayerID, np.ndarray]
        Dictionary of activations for each layer.
    directions : List[Direction]
        Directions to align.
    dataset : Dataset
        Annotated dataset portraying visual concepts.
    n_workers : int, optional
        Number of workers to use for multiprocessing.
    batch_size : int, optional
        Batch size for the analysis.
    measure : str, optional
        The measure to use for alignment.
    module : LayerID, optional
        Restrict the analysis to a specific module.

    Returns
    -------
    psi : List[Tuple[Alignment, float]]
        Estimate of the alignment.
    """

    # Compute for all modules
    if module is None:
        unique_modules = {d[0] for d in directions}
        psi = []
        for module in unique_modules:
            psi += semalign(activations, directions, dataset,
                            n_workers=n_workers, batch_size=batch_size,
                            measure=measure, module=module)
        return psi

    # Verbose
    if verbose:
        print('Semantic alignment for module {}'.format(module))

    # Activations of the module
    activations = activations[module]

    # Images in the dataset
    n_images = activations.shape[0]

    # Stack bias
    bias = np.array([d[2] for d in directions if d[0] == module])

    # Stack directions
    directions = np.stack([d[1] for d in directions if d[0] == module])

    # Directions in the current module
    n_directions = directions.shape[0]

    # Concepts in the ontology
    n_concepts = len(dataset.ontology.to_list())

    # Keep track of the best concepts per unit
    alignment = np.zeros((n_directions, n_concepts))

    # Parallel computation
    if n_workers > 0:
        pool = Pool(n_workers)

        # Chunk the dataset
        psize = int(np.ceil(float(n_images) / n_workers))
        ranges = [(s, min(n_images, s + psize)) for s
                  in range(0, n_images, psize) if s < n_images]

        # Queue to handle progress
        manager = Manager()
        queue = manager.Queue()

        # Parameter per worker
        params = []
        for i, r in enumerate(ranges):
            params.append(((activations.filename, activations.shape),
                          dataset, directions, bias, queue, batch_size, *r))

        # Map
        map_result = pool.starmap_async(_semalign, params)

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
        results = _semalign(activations, dataset, directions, bias,
                            None, batch_size, 0, len(dataset))
        intersection, union, act_sum, cmask_sum = results

    # Compute the alignment
    if measure == 'recall':
        alignment = intersection / (cmask_sum[None, :] + 1e-12)
    elif measure == 'iou':
        alignment = intersection / (union + 1e-12)
    elif measure == 'f1':
        alignment = 2 * intersection / (act_sum[:, None] +
                                        cmask_sum[None, :] + 1e-12)
    else:
        raise NotImplementedError

    # Correctly close the pool
    if n_workers > 0:
        pool.close()
        pool.join()

    # Compute set of alignment
    psi = []
    for d_idx, direction in enumerate(directions):
        for c_idx, concept in enumerate(dataset.ontology.to_list()):
            # Ignore zero alignment
            if alignment[d_idx, c_idx] > 0:
                psi.append(((direction, concept), alignment[d_idx, c_idx]))

    # Return list of alignments
    return psi
