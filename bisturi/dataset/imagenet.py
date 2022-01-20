from bisturi.ontology import Ontology, WordNetConcept
from bisturi.dataset import Dataset, ConceptMask
from tqdm import tqdm
from typing import List, Tuple
import json
import numpy as np
import os
import sys


class ImageNetConceptMask(ConceptMask):
    def __init__(self, dict_example):
        self.dict_example = dict_example
        self.wordnet_ids = set([e[0] for e in self.dict_example['boxes']])
        self.index = self.dict_example['idx']
        self.shape = self.dict_example['height'], self.dict_example['width']
        self.cache_descendants = None

    def _get_boxes(self, wordnet_id: str) \
            -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Get bounding boxes for a given synset.

        Parameters
        ----------
        wordnet_id : str
            Wordnet id of the synset.

        Returns
        -------
        boxes : List[Tuple[Tuple[float, float], Tuple[float, float]]]
            List of bounding boxes.
        """
        return [e[1] for e in self.dict_example['boxes'] if e[0] == wordnet_id]

    def select_concepts(self, concepts: List[WordNetConcept]) \
            -> List[WordNetConcept]:
        # Synset name descendants for each concept
        descendants = {c: {d.wordnet_id for d in c.descendants}
                       for c in concepts}

        # Eventually cache descendants WordNet IDs
        self.cache_descendants = descendants

        # Select only concepts whose descendants contain at least
        # one of the reported synsets
        selection = [c for c in concepts if descendants[c] & self.wordnet_ids]

        return selection

    def get_concept_mask(self, concept: WordNetConcept,
                         c_mask: np.ndarray = None):

        if c_mask is None:
            c_mask = np.empty(self.shape, dtype=bool)

        # Init mask
        c_mask &= False

        # Synset name of the descendants of the concept
        if self.cache_descendants and concept in self.cache_descendants:
            descendants = self.cache_descendants[concept]
        else:
            descendants = {d.wordnet_id for d in concept.descendants}

        # Intersection with synset names annotated in the image
        wordnet_ids = descendants & self.wordnet_ids

        # Synsets contained in the image
        for wordnet_id in wordnet_ids:
            # Construct bounding boxes
            for box in self._get_boxes(wordnet_id):
                pmin_x, pmin_y = box[0]
                pmax_x, pmax_y = box[1]

                # Consider only positive bounding boxes
                pmin_x = min(max(int(pmin_x), 0), self.shape[1]-1)
                pmin_y = min(max(int(pmin_y), 0), self.shape[0]-1)
                pmax_x = min(max(int(pmax_x), 0), self.shape[1]-1)
                pmax_y = min(max(int(pmax_y), 0), self.shape[0]-1)

                # Set the bounding box as true
                c_mask[pmin_y:pmax_y, pmin_x:pmax_x] = True

        return c_mask


class ImageNetOntology(Ontology):
    def __init__(self, directory,
                 fname: str = 'ontology.txt'):

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
        wordnet_ids = hypernyms.union(hyponyms)
        root_syn = list(hypernyms - hyponyms)[0]

        # Build nodes
        nodes = {s: WordNetConcept(s) for s in wordnet_ids}

        # Identify root
        root = nodes[root_syn]

        # Connect nodes
        for hypernym, hyponym in raw_ontology:
            nodes[hypernym].hyponyms += [nodes[hyponym]]
            nodes[hyponym].hypernyms += [nodes[hypernym]]

        # Init superclass
        super().__init__(root)


class ImageNetDataset(Dataset):
    def __init__(self, directory: str,
                 index: str = 'index_224_center.json',
                 reverse_index: str = 'reverse_index.json',
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225],
                 skip_image: bool = False,
                 skip_masks: bool = False,
                 ontology: ImageNetOntology = None):

        # Load index
        index_path = os.path.join(directory, index)
        with open(index_path, 'r') as fp:
            self.index = json.load(fp)

        # Call super constructor
        super().__init__(directory, reverse_index, mean, std,
                         skip_image, skip_masks, ontology)

    def _get_mask(self, idx: int) -> ImageNetConceptMask:
        return ImageNetConceptMask(self.index[idx])

    def _build_reverse_index(self, reverse_index_path: str):
        # Notify the user
        print('Missing reverse index, bulding it...',
              file=sys.stderr)

        # Compute reverse index
        rev_index = {}

        # Avoid loading images
        was_skip_image = self.skip_image
        self.skip_image = True

        # Images within the dataset
        for idx, _, y in tqdm(self, total=len(self)):
            # Synsets within the image
            for wordnet_id in y.wordnet_ids:

                # Retrieve concept ID from WordNet ID
                concept_id = int(wordnet_id[1:])

                # Create entry if new concept
                if concept_id not in rev_index:
                    rev_index[concept_id] = []

                # Insert image
                rev_index[concept_id].append(idx)

        # Store to file
        with open(reverse_index_path, 'w') as fp:
            json.dump(rev_index, fp, indent=2)

        # Restore state
        self.skip_image = was_skip_image

        return rev_index
