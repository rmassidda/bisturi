from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
from torchvision import transforms, io
from tqdm import tqdm
from typing import Tuple, Sequence, List, Union, Dict, Set
import numpy as np
import torch

from bisturi.ontology import Concept
from bisturi.util import reshape_concept_mask, sigmoid


def balanced_indices(weights: Sequence[float], n_samples: int,
                     cut_point: float = 0.8) \
                     -> Tuple[List[int], List[int], List[int]]:

    # Weighted Sample
    sample_indeces = list(WeightedRandomSampler(
        weights, n_samples, replacement=False))

    # Shuffle list
    sample_indeces = list(np.random.permutation(sample_indeces))

    # Training set
    train_indeces = sample_indeces[:int(n_samples * cut_point)]

    # Validation set
    val_indeces = sample_indeces[int(n_samples * cut_point):]

    return train_indeces, val_indeces, sample_indeces


def balanced_splits(weights: Sequence[float], n_samples: int,
                    dataset: Dataset, cut_point: float = 0.8) \
                    -> Tuple[Dataset, Dataset, Dataset]:
    train_indices, val_indices, sample_indices = balanced_indices(
        weights, n_samples, cut_point)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    subset = Subset(dataset, sample_indices)
    return train_set, val_set, subset


class SegmentedDataset(Dataset):
    def __init__(self,
                 directory: str,
                 reverse_index: Union[None, str] = None,
                 mean: List[float] = [0.48898, 0.46544, 0.42956],
                 std: List[float] = [1, 1, 1],
                 skip_image: bool = False,
                 skip_masks: bool = False,
                 target_concept: Union[Concept,
                                       List[Concept],
                                       None] = None,
                 return_index: bool = False):

        # Dataset Directory
        self.directory = directory

        # Reverse index
        if reverse_index:
            raise NotImplementedError

        # Store preferences
        self.skip_image = skip_image
        self.skip_masks = skip_masks
        self.target_concept = target_concept
        self.return_index = return_index

        # Parse path
        self.path: List[str] = []

        # Parse Shape
        self.shape: List[Dict[str, int]] = []

        # Normalize
        self.mean = mean
        self.std = std
        self.normalizer = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def _build_reverse_index(self):
        raise NotImplementedError
        # # Notify the user
        # print('Missing reverse index, bulding it...',
        #       file=sys.stderr)

        # # Compute reverse index
        # rev_index = {}
        # # Images within the dataset
        # for idx, (_, label) in tqdm(enumerate(self), total=len(self)):
        #     # Synsets within the image
        #     for synset in label.synsets:

        #         # Retrieve concept ID from WordNet name
        #         concept_id = int('1'+synset[1:])

        #         # Create entry if new concept
        #         if concept_id not in rev_index:
        #             rev_index[concept_id] = []

        #         # Insert image
        #         rev_index[concept_id].append(idx)

        # # Store to file
        # with open(reverse_index_path, 'w') as fp:
        #     json.dump(rev_index, fp, indent=2)

        # return rev_index

    def instances(self, concept: Concept):
        raise NotImplementedError

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        if not self.skip_image:
            img_arr = self._get_image(idx)
        else:
            raise NotImplementedError

        if not self.skip_masks:
            mask_arr = self._get_masks(idx)
        else:
            raise NotImplementedError

        return img_arr, mask_arr

    def _get_image(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.path[idx]
        img_arr = io.read_image(img_path)

        # Repeat channels for gray images
        if (img_arr.shape[0] == 1):
            img_arr = torch.repeat_interleave(img_arr, 3, dim=0)

        # Remove channels for RGBA
        if (img_arr.shape[0] == 4):
            img_arr = img_arr[:3, :, :]

        # FIXME: when using multiprocessing on jupyter notebook,
        #        the computation stucks with the following call.
        img_arr = self.normalizer(img_arr)

        return img_arr

    def _get_masks(self, idx: int) -> np.ndarray:
        raise NotImplementedError

    def compute_balance_weights(self, targets: List[Concept],
                                batch_size: int = 32,
                                act: Union[torch.Tensor, None] = None,
                                verbose: bool = False,
                                steepness: float = 1.0,
                                nw: int = 0) -> torch.Tensor:
        self.target_concept = targets
        self.return_index = True

        # Ratio of positives to total per image
        ratio = torch.zeros(len(self), len(targets))

        loader = DataLoader(self, batch_size=batch_size,
                            shuffle=False, num_workers=nw)  # type: ignore
        # Verbose loading
        if verbose:
            loader = tqdm(loader)

        # Count of positive samples
        for batch in loader:
            idx, _, y = batch

            if act is not None:
                y = y.float()
                y = reshape_concept_mask(y, act)

            # Count positive locations and number of
            # total locations in the target mask
            if len(y.shape) == 4:
                pos_locations = torch.sum(y, dim=(2, 3))
                target_size = y.shape[2] * y.shape[3]
            elif len(y.shape) == 2:
                pos_locations = y
                target_size = 1
            else:
                raise ValueError('The concept mask must be a 2D or 4D tensor')

            # Compute the ratio of the images in the batch
            ratio[idx] = pos_locations / target_size

        # Probability of positive samples
        # for each concept
        pos_prob = ratio.mean(dim=0)

        # Eventually remark the probability
        if steepness != 1.0:
            pos_prob = sigmoid(pos_prob, steepness)

        # Weights according to the probability
        weights = ratio * (1 - pos_prob) + (1 - ratio) * pos_prob

        return weights


def concept_mask(segmentation: np.ndarray, concept: Concept) -> np.ndarray:
    """
    Given a segmentation, it computes the
    concept mask of a given concept. Each
    concept contains a list of its labels.
    """
    depth, height, width = segmentation.shape

    c_mask = np.zeros((height, width), dtype=np.bool8)

    for channel in range(depth):
        for label in concept.labels:
            c_mask |= (segmentation[channel] == label)

    return c_mask


def label_mask(segmentation: np.ndarray, label: int) -> np.ndarray:
    """
    Given a segmentation, it computes the
    label mask of a given label.
    """
    depth, height, width = segmentation.shape

    l_mask = np.zeros((height, width), dtype=np.bool8)

    for channel in range(depth):
        l_mask |= (segmentation[channel] == label)

    return l_mask


def unique_labels(segmentation: np.ndarray) -> Set[int]:
    """
    Given a segmentation, it returns
    a set containing the unique labels.
    """
    depth, height, width = segmentation.shape

    labels = set()

    for channel in range(depth):
        labels |= set(segmentation[channel].flatten())

    return labels


def unique_concepts(segmentation: np.ndarray,
                    concepts: List[Concept]) \
        -> Set[Concept]:
    """
    Given a segmentation, it returns
    a set containing the concepts
    annotated in the image.
    """

    # Obtain the labels
    labels = unique_labels(segmentation)

    return {c for c in concepts if c.labels & labels}
