from csv import DictReader
from imageio import imread
from typing import List, Union, Dict
import numpy as np
import os

from bisturi.ontology import Concept
from bisturi.ontology.wordnet import WordnetOntology, WordnetConcept
from bisturi.dataset import SegmentedDataset


class BrodenDataset(SegmentedDataset):
    """
    Returns the images contained in
    an annotated Broden dataset.

    The BGR mean was:
    109.5388, 118.6897, 124.6901
    Here it is normalized and in RGB.
    """
    def __init__(self, directory: str,
                 fname: str = 'index.csv',
                 labels: str = 'label.csv',
                 reverse_index: Union[str, None] = None,
                 mean: List[float] = [0.48898, 0.46544, 0.42956],
                 std: List[float] = [1, 1, 1],
                 skip_image: bool = False,
                 skip_masks: bool = False,
                 target_concept: Union[Concept,
                                       List[Concept],
                                       None] = None,
                 categories: List[str] = ['object', 'part', 'material'],
                 return_index: bool = False):

        # Initialize generic Segmented Dataset
        super().__init__(directory, reverse_index, mean, std, skip_image,
                         skip_masks, target_concept, return_index)

        # Read index from file
        index = os.path.join(directory, fname)
        with open(index, 'r') as fp:
            csv_reader = DictReader(fp)
            rows = [row for row in csv_reader]

        # Read labels from file
        labels_path = os.path.join(directory, labels)
        with open(labels_path, 'r') as fp:
            csv_reader = DictReader(fp)
            self.labels = {int(row['number']): {**row} for row in csv_reader}

        # Parse path
        self.path: List[str] = (
            [
                os.path.join(self.directory, 'images', row['image'])
                for row in rows
            ]
        )

        # Parse categories
        self.categories: List[Dict[str, List[str]]] = (
            [
                {
                    # Split and filter out empty categories
                    k: [e for e in row[k].split(';') if e != '']
                    for k in categories
                } for row in rows
            ]
        )

        # Parse Shape
        self.shape: List[Dict[str, int]] = (
            [
                {
                    k: int(row[k])
                    for k in ['ih', 'iw', 'sh', 'sw']
                } for row in rows
            ]
        )

    def _get_masks(self, idx: int) -> np.ndarray:
        # Dictionary containing masks
        categories: Dict[str, List[str]] = self.categories[idx]

        # How many masks are there?
        depth: int = sum(len(category) for category in categories.values())

        # Mask size
        sh: int = self.shape[idx]['sh']
        sw: int = self.shape[idx]['sw']

        # Create a segmentation mask
        out: np.ndarray = np.zeros((depth, sh, sw))

        # Init channel counter
        c_idx: int = 0

        for category in categories:
            # Assign channel
            for channel in categories[category]:
                try:
                    # Unique label over the whole image
                    out[c_idx] = int(channel)
                except ValueError:
                    # FIXME: this does not work correctly when
                    #        loading using PyTorch io.read_image
                    mask_fname: str = os.path.join(
                        self.directory, 'images', channel)
                    rgb_imread = imread(mask_fname)
                    out[c_idx] = rgb_imread[:, :, 0] + \
                        rgb_imread[:, :, 1] * 256

                # Increment channel counter
                c_idx += 1

        if self.target_concept is not None:
            raise NotImplementedError

        return out


class BrodenOntology(WordnetOntology):
    def __init__(self,
                 directory: str,
                 fname: str = 'broden_wordnet_alignment.csv',
                 propagate: bool = True):

        # Read alignment file
        with open(os.path.join(directory, fname),
                  'r') as fp:
            csv_reader = DictReader(fp)
            alignment = [row for row in csv_reader]

        # Directly annotated concepts
        nodes = {r['WordNet ID']: WordnetConcept(r['WordNet ID'])
                 for r in alignment}

        # Assign Broden labels to WordNet concepts
        for concept in nodes.values():
            concept.labels = {int(r['Broden ID']) for r in alignment
                              if r['WordNet ID'] == concept.name}
            concept.original_labels = concept.labels.copy()

        # Init superclass
        super().__init__(nodes, propagate=propagate)
