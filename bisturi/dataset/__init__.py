from bisturi.util import reshape_concept_mask, sigmoid
from bisturi.ontology import Ontology, Concept
from torchvision import transforms, io
from tqdm import tqdm
from typing import Tuple, List, Set
import json
import numpy as np
import os
import torch


class ConceptMask:
    """
    Class representing the annotation of
    a segmented dataset.

    Methods
    ----------
    select_concepts(concepts)
        Returns the intersection provided concepts
        and the annotated ones.
    get_concept_mask(concept)
        Returns the concept mask of the given concept.
    """

    def __init__(self):
        pass

    def select_concepts(self, concepts: List[Concept]) -> Set[Concept]:
        """
        Returns the intersection between a given
        list of concepts and the concepts of the
        example.

        Parameters
        ----------
        concepts : List[Concept]
            List of concepts to intersect.

        Returns
        -------
        selection : Set[Concept]
            Intersection between the given list
            of concepts and the concepts of the
            example.
        """
        raise NotImplementedError

    def get_concept_mask(self, concept: Concept,
                         c_mask: np.ndarray = None) -> np.ndarray:
        """
        Returns the concept mask of the given concept.

        Parameters
        ----------
        concept : Concept
            Concept to get the concept mask of.
        c_mask : np.ndarray, optional
            Allocated mask to use, if None a new one
            will be allocated.

        Returns
        -------
        c_mask : np.ndarray
            Concept mask of the given concept.
        """
        raise NotImplementedError


def collate_masks(batch):
    """
    Custom function to collate indexes, images and
    concept masks within pyTorch DataLoader.
    """

    # Indexes
    indexes = [idx for idx, _, _ in batch]
    indexes = torch.utils.data._utils.collate.default_collate(indexes)

    # Images
    images = [x for _, x, _ in batch]
    images = torch.utils.data._utils.collate.default_collate(images)

    # Concept masks
    masks = [y for _, _, y in batch]

    return indexes, images, masks


class Dataset(torch.utils.data.Dataset):
    """
    Returns the images contained in
    an annotated ImageNet dataset.
    """
    def __init__(self, directory: str,
                 reverse_index: str = 'reverse_index.json',
                 mean: List[float] = [0., 0., 0.],
                 std: List[float] = [0., 0., 0.],
                 skip_image: bool = False,
                 skip_masks: bool = False,
                 ontology: Ontology = None):

        # Dataset directory
        self.directory = directory

        # Store preferences
        self.skip_image = skip_image
        self.skip_masks = skip_masks

        # Input normalization
        self.normalizer = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=mean, std=std)
        ])

        # Reverse index
        reverse_index_path = os.path.join(directory, reverse_index)
        try:
            with open(reverse_index_path, 'r') as fp:
                self.reverse_index = json.load(fp)
        except FileNotFoundError:
            self.reverse_index = self._build_reverse_index(reverse_index_path)

        # Ontology
        self.ontology = ontology

        # Eventually identify propagated concepts
        if ontology:
            for c_id, concept in ontology.nodes.items():
                concept.propagated = c_id not in self.reverse_index

    def _build_reverse_index(self, reverse_index_path: str = None):
        """
        Builds the reverse index of the dataset
        and eventually saves it to disk.

        Parameters
        ----------
        reverse_index_path: str
            Path to the reverse index file.
        """
        raise NotImplementedError

    def instances(self, concept: Concept) -> Set[int]:
        """
        Returns the set of instances
        of a given concept.

        Parameters
        ----------
        concept : Concept
            Concept to retrieve instances.

        Returns
        -------
        instances : Set[int]
            Set of instances of the given concept.
        """
        instances = set()

        # A concept is portrayed by its descendants
        for c in concept.descendants:
            instances.update(self.reverse_index[c.id])

        return instances

    def _get_image(self, idx: int) -> torch.Tensor:
        """
        Returns the image at the given index.

        Parameters
        ----------
        idx : int
            Index of the image to retrieve.

        Returns
        -------
        img_arr : torch.Tensor
            Image at the given index.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.index[idx]
        img_path = image['path']
        img_arr = io.read_image(os.path.join(self.directory, img_path))

        # Repeat channels for gray images
        if (img_arr.shape[0] == 1):
            img_arr = torch.repeat_interleave(img_arr, 3, axis=0)

        # Remove channels for RGBA
        if (img_arr.shape[0] == 4):
            img_arr = img_arr[:3, :, :]

        img_arr = self.normalizer(img_arr)

        return img_arr

    def _get_mask(self, idx: int) -> ConceptMask:
        """
        Returns the mask at the given index.

        Parameters
        ----------
        idx : int
            Index of the mask to retrieve.

        Returns
        -------
        mask : ConceptMask
            Mask at the given index.
        """
        return ConceptMask()

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, ConceptMask]:
        # Eventually retrieve the image
        img = self._get_image(idx) if not self.skip_image else 0

        # Eventually retrieve annotations
        mask = self._get_mask(idx) if not self.skip_masks else None

        # Return the image and annotations
        return idx, img, mask

    def __len__(self):
        return len(self.index)


def compute_balance_weights(dataset: Dataset,
                            targets: List[Concept],
                            batch_size: int = 32,
                            out_shape: Tuple[int, int] = None,
                            verbose: bool = False,
                            steepness: float = 1.0,
                            nw: int = 0) -> np.ndarray:
    """
    Independently weights the examples of a dataset
    according to the coverage of the given concepts.

    Parameters
    ----------
    dataset : Dataset
        Dataset to weight.
    targets : List[Concept]
        Concepts to weight the dataset.
    batch_size : int, optional
        Batch size to visit the dataset.
    out_shape : Tuple[int, int], optional
        Shape to resize concept masks to.
    verbose : bool, optional
        Whether to print information about the
        computation.
    steepness : float, optional
        Steepness of the sigmoid function used
        to polarize the weights.
    nw : int, optional
        Number of workers to use in the data loader.

    Returns
    -------
    weights : np.ndarray
        Per image and concept weights of the dataset.
    """

    # Ratio of positives to total per image
    ratio = np.zeros((len(dataset), len(targets)))

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=nw,
                                         collate_fn=collate_masks)
    # Verbose loading
    if verbose:
        loader = tqdm(loader)

    # Count of positive samples
    for batch in loader:

        # Iterate over batch
        for idx, x, y in zip(*batch):

            # Annotated concepts
            selected_concepts = y.select_concepts(targets)
            selected_concepts = [(c_idx, c) for c_idx, c in enumerate(targets)
                                 if c in selected_concepts]

            # Iterate over selected concepts
            for c_idx, concept in selected_concepts:

                # Retrieve concept mask
                c_mask = y.get_concept_mask(concept)

                # Reshape the concept mask
                if out_shape is not None:
                    c_mask = c_mask.float()
                    c_mask = reshape_concept_mask(c_mask, out_shape)

                # Positive locations
                pos_locations = np.sum(c_mask)

                # Mask size
                target_size = np.prod(c_mask.shape)

                # Compute the ratio of the images in the batch
                ratio[idx][c_idx] = pos_locations / target_size

    # Probability of positive samples
    # NOTE: +1e-8 to handle concepts with no positive samples
    pos_prob = ratio.mean(axis=0) + 1e-8

    # Eventually remark the probability
    if steepness != 1.0:
        pos_prob = sigmoid(pos_prob, steepness)

    # Weights according to the probability
    weights = ratio * (1 - pos_prob) + (1 - ratio) * pos_prob

    return weights.T


def balanced_splits(weights: np.ndarray, n_samples: int,
                    dataset: Dataset = None,
                    cut_point: float = 0.8) \
                    -> Tuple[torch.utils.data.Dataset,
                             torch.utils.data.Dataset]:
    """
    Produce balances splits of a dataset.

    Parameters
    ----------
    weights : np.ndarray
        Per image weights of the dataset.
    n_samples : int
        Number of samples to extract from the dataset.
    dataset : Dataset, optional
        Dataset to split.
    cut_point : float, optional
        Cut point to split the dataset.

    Returns
    -------
    train_set : torch.utils.data.Dataset
        Training set.
    val_set : torch.utils.data.Dataset
        Validation set.
    """

    # Weighted Sample
    sample_indeces = list(torch.utils.data.WeightedRandomSampler(
        weights, n_samples, replacement=False))

    # Shuffle list
    sample_indeces = np.random.permutation(sample_indeces)

    # Cut point example
    cut_point = int(n_samples * cut_point)

    # Training set
    train_indeces = sample_indeces[:cut_point]

    # Validation set
    val_indeces = sample_indeces[cut_point:]

    if dataset:
        # Eventually split the dataset into subsets
        train_set = torch.utils.data.Subset(dataset, train_indeces)
        val_set = torch.utils.data.Subset(dataset, val_indeces)
        return train_set, val_set
    else:
        # Return the indeces
        return list(train_indeces), list(val_indeces)
