from __future__ import annotations
from bisturi.util import synset_to_wnid
from bisturi.ontology import Ontology, WordNetConcept
from bisturi.dataset import ConceptMask, Dataset
from csv import DictReader
from imageio import imread
from tqdm import tqdm
from typing import List
import numpy as np
import os
import sys
import json


class BrodenConcept(WordNetConcept):
    """
    Class representing a concept from
    the WordNet ontology that corresponds
    to multiple Broden labels.

    Attributes
    ----------
    b_ids : set
        Set of Broden IDs corresponding
        to this concept.
    original_b_ids : set
        Set of Broden IDs that were
        manually assigned to this concept.
    """
    def __init__(self, wordnet_id: str,
                 hypernyms: List[BrodenConcept] = None,
                 hyponyms: List[BrodenConcept] = None):
        super().__init__(wordnet_id, hypernyms, hyponyms)

        # Broden IDs corresponding to this concept
        self.b_ids = set()
        self.original_b_ids = set()

    def is_placeholder(self):
        """
        Returns
        -------
        bool
            True if this concept is a placeholder
            (i.e. it holds the same Broden IDs of
            its only child).
        """
        return (len(self.hyponyms) == 1
                and self.hyponyms[0].b_ids == self.b_ids)


class BrodenConceptMask(ConceptMask):
    def __init__(self, dict_example, categories):
        self.dict_example = dict_example
        self.categories = categories

        self.index = self.dict_example['idx']
        self.shape = self.dict_example['height'], self.dict_example['width']

        # NOTE: call to get_broden_ids() could also be done here
        self.b_ids = None

    def get_broden_ids(self):
        pixels = []
        scalars = []

        for category in self.categories:
            category_map = self.dict_example[category]
            shape = np.shape(category_map)

            # NOTE: why this?
            if len(shape) % 2 == 0:
                category_map = [category_map]

            if len(shape) < 2:
                # Scalar annotation
                scalars += category_map
            else:
                # Pixel-level annotation
                pixels.append(category_map)

        # Retrieve unique Broden IDs
        b_ids = [scalar for scalar in scalars]
        for p in pixels:
            b_ids += list(np.argwhere(np.bincount(p.ravel()) > 0)[:, 0])
        # '0' is not a Broden ID
        b_ids = {i for i in b_ids if i != 0}

        return b_ids

    def select_concepts(self, concepts: List[BrodenConcept]):
        # Cache Broden IDs contained in the image
        if not self.b_ids:
            self.b_ids = self.get_broden_ids()

        # Select concepts with relevant broden IDs
        selected_concepts = [c for c in concepts if c.b_ids & self.b_ids]

        return selected_concepts

    def get_concept_mask(self, concept: BrodenConcept,
                         c_mask: np.ndarray = None):
        # Init mask
        if c_mask is None:
            c_mask = np.empty(self.shape, dtype=bool)
        c_mask &= False

        # Broden IDs
        b_ids = concept.b_ids

        # Reconstruct mask
        for category in self.categories:
            category_map = self.dict_example[category]
            shape = np.shape(category_map)

            # Scalar annotation
            if len(shape) < 2:

                # The category has not any concept map
                if shape[0] == 0:
                    continue

                # All of the image contains
                # one of the leaves as a visual
                # concept, it is irrelevant to
                # continue since all of the mask
                # is therefore active.
                scalar = category_map[0]
                if scalar in b_ids:
                    c_mask |= True
                    return c_mask

            # Pixel-by-pixel annotation
            elif len(shape) == 3:
                for cid in b_ids:
                    for i in range(shape[0]):
                        c_mask |= category_map[i] == cid

        return c_mask


class BrodenOntology(Ontology):
    def __init__(self, directory,
                 fname='broden_wordnet_alignment.csv'):

        # Read alignment file
        with open(os.path.join(directory, fname),
                  'r') as fp:
            csv_reader = DictReader(fp)
            alignment = [row for row in csv_reader]

        # Directly annotated concepts
        nodes = {r['WordNet ID']: BrodenConcept(r['WordNet ID'])
                 for r in alignment}

        # Assign Broden labels to WordNet concepts
        for concept in nodes.values():
            concept.b_ids = {int(r['Broden ID']) for r in alignment
                             if r['WordNet ID'] == concept.wordnet_id}
            concept.original_b_ids = concept.b_ids.copy()

        # Construct hypernymy
        raw_ontology = []
        synsets = [c.synset for c in nodes.values()]
        while len(synsets) != 0:
            new_synsets = [ss.hypernyms() for ss in synsets]
            for hyponym, hypernyms in zip(synsets, new_synsets):
                for hypernym in hypernyms:
                    raw_ontology.append((synset_to_wnid(hypernym),
                                         synset_to_wnid(hyponym)))
            synsets = set(sum(new_synsets, []))
        raw_ontology = set(raw_ontology)

        # Construct the ontology
        hypernyms = set([e[0] for e in raw_ontology])
        hyponyms = set([e[1] for e in raw_ontology])
        synsets = hypernyms.union(hyponyms)
        root_syn = list(hypernyms - hyponyms)[0]

        # Higher-level concepts
        for synset in synsets:
            if synset not in nodes:
                nodes[synset] = BrodenConcept(synset)

        # Connect nodes
        for hypernym, hyponym in raw_ontology:
            nodes[hypernym].hyponyms += [nodes[hyponym]]
            nodes[hyponym].hypernyms += [nodes[hypernym]]

        # Identify root
        root = nodes[root_syn]

        # Cumulate Broden labels to eventually
        # retrieve higher level visual concepts
        for synset in nodes:
            concept = nodes[synset]
            for descendant in concept.descendants:
                concept.b_ids |= descendant.b_ids

        # Init superclass
        super().__init__(root)


class BrodenDataset(Dataset):
    def __init__(self, directory: str,
                 index: str = 'index.csv',
                 reverse_index: str = 'reverse_index.json',
                 mean: List[float] = [0.48898, 0.46544, 0.42956],
                 std: List[float] = [1, 1, 1],
                 skip_image: bool = False,
                 skip_masks: bool = False,
                 ontology: BrodenOntology = None,
                 categories: List[str] = ['object', 'part', 'material'],
                 labels: str = 'label.csv'):

        # Read index from file
        index_path = os.path.join(directory, index)
        with open(index_path, 'r') as fp:
            csv_reader = DictReader(fp)
            self.index = [row for row in csv_reader]

        # Load original Broden labels
        labels_path = os.path.join(directory, labels)
        with open(labels_path, 'r') as fp:
            csv_reader = DictReader(fp)
            self.labels = {int(row['number']): {**row} for row in csv_reader}

        # Store categories
        self.categories = categories

        # Adjust path
        for image in self.index:
            image['path'] = os.path.join('images', image['image'])

        # Parse integers
        for image in self.index:
            for key in image:
                if key in ['ih', 'iw', 'sh', 'sw']:
                    image[key] = int(image[key])

        # Split lists containing annotations
        for image in self.index:
            for key in image:
                if key in self.categories:
                    if image[key]:
                        image[key] = image[key].split(';')

        # Call super constructor
        super().__init__(directory, reverse_index, mean, std,
                         skip_image, skip_masks, ontology)

    def _build_reverse_index(self, reverse_index_path: str):
        # Notify the user
        print('Missing reverse index, bulding it...',
              file=sys.stderr)

        # Compute reverse index
        rev_index = {}

        # Avoid loading images
        was_skip_image = self.skip_image
        self.skip_image = True

        # Concepts from the ontology
        concepts = self.ontology.to_list()
        # Select only concepts manually associated to a Broden ID
        concepts = [c for c in concepts if c.original_b_ids]

        # Images within the dataset
        for idx, _, y in tqdm(self, total=len(self)):

            # Concepts within the image
            for c in y.select_concepts(concepts):
                # Create entry if new concept
                if c.id not in rev_index:
                    rev_index[c.id] = []

                # Insert image
                rev_index[c.id].append(idx)

        # Store to file
        with open(reverse_index_path, 'w') as fp:
            json.dump(rev_index, fp, indent=2)

        # Restore state
        self.skip_image = was_skip_image

        return rev_index

    def _get_mask(self, idx: int) -> BrodenConceptMask:
        # Dictionary containing masks
        row = self.index[idx]
        # Filter out categories not annotated for the image
        categories = [c for c in self.categories if row[c]]
        masks = {'idx': idx, 'height': row['sh'], 'width': row['sw']}

        # Iterate over categories
        for category in categories:

            # Empty map for each distinct annotation source
            out = np.zeros((len(row[category]), row['sh'], row['sw']),
                           dtype=np.int16)

            # Iterate over all annotations
            for i, channel in enumerate(row[category]):
                try:
                    # Unique concept over the image
                    out[i] = int(channel)
                except ValueError:
                    # Multiple concepts over the image
                    mask_fname = os.path.join(self.directory, 'images',
                                              channel)
                    # FIXME: this does not work correctly when
                    #        loading using PyTorch io.read_image
                    #        instead of imageio.imread
                    rgb_imread = imread(mask_fname)
                    out[i] = rgb_imread[:, :, 0] + \
                        rgb_imread[:, :, 1] * 256

            masks[category] = out

        return BrodenConceptMask(masks, categories)
