from typing import List, Union, Set, Dict, Tuple
import json
import numpy as np
import os

from bisturi.dataset import SegmentedDataset
from bisturi.ontology import Concept
from bisturi.ontology.wordnet import WordnetConcept, WordnetOntology

Point = Tuple[int, int]


class ImageNetDataset(SegmentedDataset):
    """
    Returns the images contained in
    an annotated ImageNet dataset.
    """
    def __init__(self, directory: str,
                 fname: str = 'index_224_center.json',
                 reverse_index: Union[str, None] = None,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225],
                 skip_image: bool = False,
                 skip_masks: bool = False,
                 target_concept: Union[Concept,
                                       List[Concept],
                                       None] = None,
                 return_index: bool = False):

        # Initialize generic Segmented Dataset
        super().__init__(directory, reverse_index, mean, std, skip_image,
                         skip_masks, target_concept, return_index)

        # Open index
        index = os.path.join(directory, fname)
        with open(index, 'r') as fp:
            self.index = json.load(fp)

        # Set path for the images
        self.path: List[str] = [
            os.path.join(self.directory, entry['path'])
            for entry in self.index
        ]

    def _get_masks(self, idx: int) -> np.ndarray:
        # Select entry from the index
        entry: Dict = self.index[idx]

        # What synsets are annotated in the image?
        synsets: Set[str] = set([e[0] for e in entry['boxes']])

        # How many masks are there?
        depth: int = len(synsets)

        # Mask size
        height: int = entry['height']
        width: int = entry['width']

        # Create a segmentation mask
        out: np.ndarray = np.zeros((depth, height, width))

        # Iterate over synsets
        for c_idx, synset in enumerate(synsets):

            # Get the offset of the synset
            offset: int = int(synset[1:])

            # Get the boxes corresponding to the synset
            boxes: List[Tuple[Point, Point]] = (
                [e[1] for e in entry['boxes'] if e[0] == synset]
            )

            for box in boxes:
                pmin_x, pmin_y = box[0]
                pmax_x, pmax_y = box[1]

                # Consider only positive bounding boxes
                pmin_x = min(max(int(pmin_x), 0), width - 1)
                pmin_y = min(max(int(pmin_y), 0), height - 1)
                pmax_x = min(max(int(pmax_x), 0), width - 1)
                pmax_y = min(max(int(pmax_y), 0), height - 1)

                # Set the bounding box as true
                out[c_idx, pmin_y:pmax_y, pmin_x:pmax_x] = offset

        if self.target_concept is not None:
            raise NotImplementedError

        return out


class ImageNetOntology(WordnetOntology):
    def __init__(self,
                 directory: str,
                 fname: str = 'ontology.txt',
                 propagate: bool = True):

        # Ontology path
        ontology_path = os.path.join(directory, fname)

        # Retrieve ontology from file
        with open(ontology_path) as fp:
            lines = fp.read().split('\n')[:-1]

        # Parse is_a relationships
        raw_ontology = [e.split() for e in lines]

        # Partition the synsets
        hypernyms = set([e[0] for e in raw_ontology])
        hyponyms = set([e[1] for e in raw_ontology])
        synsets = hypernyms.union(hyponyms)

        # Build nodes
        nodes = {s: WordnetConcept(s) for s in synsets}

        # Init superclass
        super().__init__(nodes, propagate=propagate)
