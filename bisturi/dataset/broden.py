from bisturi.util import sigmoid, reshape_concept_mask
from bisturi.ontology import Ontology, WordnetConcept, ConceptMask
from csv import DictReader
from imageio import imread
from torchvision import transforms, io
from tqdm import tqdm
from typing import List, Union
import json
import numpy as np
import os
import torch


def synset_to_id(synset):
    return synset.pos() + str(synset.offset()).zfill(8)


class BrodenConcept(WordnetConcept):
    """
    Represents a concept in the
    BrodenConcept ontology.
    """
    def __init__(self, name, hypernyms=None, hyponyms=None):
        super().__init__(name, hypernyms, hyponyms)

        # Broden IDs corresponding to this concept
        self.b_ids = set()
        self.original_b_ids = set()

    def is_placeholder(self):
        return (len(self.hyponyms) == 1
                and self.hyponyms[0].b_ids == self.b_ids)

    def is_propagated(self) -> bool:
        """
        Returns True if the concept
        is obtained via its children
        and does not have any original
        b_id associated
        """
        return len(self.original_b_ids) == 0

    def propagation_ratio(self) -> float:
        """
        Summarizes the gain in terms
        of associated b_ids given
        the children.

        Intuitively, the ratio is 0 if
        all the b_ids of the concept
        were manually assigned, while
        it is 1 if all of them are
        derived by the children of
        the ontology.
        """
        num = len(self.original_b_ids)
        dem = len(self.b_ids)
        return 1 - num / dem


class BrodenDataset(torch.utils.data.Dataset):
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
                 reverse_index: str = 'reverse_index.json',
                 mean: List[float] = [0.48898, 0.46544, 0.42956],
                 std: List[float] = [1, 1, 1],
                 skip_image: bool = False,
                 skip_masks: bool = False,
                 target_concept: Union[BrodenConcept,
                                       List[BrodenConcept]] = None,
                 categories: List[str] = ['object', 'part', 'material'],
                 return_index: bool = False):

        # Compose index file path
        self.directory = directory

        # Read index from file
        index = os.path.join(directory, fname)
        with open(index, 'r') as fp:
            csv_reader = DictReader(fp)
            self.index = [row for row in csv_reader]

        # Read labels from file
        labels_path = os.path.join(directory, labels)
        with open(labels_path, 'r') as fp:
            csv_reader = DictReader(fp)
            self.labels = {int(row['number']): {**row} for row in csv_reader}

        # Reverse index
        if reverse_index:
            reverse_index_path = os.path.join(directory, reverse_index)
            with open(reverse_index_path, 'r') as fp:
                self.reverse_index = json.load(fp)

            # Convert string IDs to int IDs
            self.reverse_index = {int(k): v for k, v in
                                  self.reverse_index.items()}

        # Store preferences
        self.skip_image = skip_image
        self.skip_masks = skip_masks
        self.categories = categories
        self.target_concept = target_concept
        self.return_index = return_index

        # Adjust path
        for row in self.index:
            row['path'] = os.path.join(self.directory,
                                       'images',
                                       row['image'])

        # Parse integers
        for row in self.index:
            for key in row:
                if key in ['ih', 'iw', 'sh', 'sw']:
                    row[key] = int(row[key])

        # Split lists
        for row in self.index:
            for key in row:
                if key in self.categories:
                    if row[key]:
                        row[key] = row[key].split(';')

        # Normalize
        self.mean = mean
        self.std = std
        self.normalizer = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def instances(self, concept: BrodenConcept):
        indices = []
        for b_id in concept.b_ids:
            if b_id in self.reverse_index:
                indices.extend(self.reverse_index[b_id])
        indices = set(indices)
        return indices

    def __len__(self):
        return len(self.index)

    def _get_image(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.index[idx]
        img_path = image['path']
        img_arr = io.read_image(img_path)

        # Repeat channels for gray images
        if (img_arr.shape[0] == 1):
            img_arr = torch.repeat_interleave(img_arr, 3, axis=0)

        # Remove channels for RGBA
        if (img_arr.shape[0] == 4):
            img_arr = img_arr[:3, :, :]

        # FIXME: when using multiprocessing on jupyter notebook,
        #        the computation stucks with the following call.
        img_arr = self.normalizer(img_arr)

        return img_arr

    def _get_masks(self, idx: int) -> Union[ConceptMask, torch.Tensor]:
        # Dictionary containing masks
        row = self.index[idx]
        categories = [c for c in self.categories if row[c]]
        masks = {'i': idx, 'sh': row['sh'], 'sw': row['sw']}
        for category in categories:
            if row[category]:
                depth = len(row[category])
                out = np.zeros((depth, row['sh'], row['sw']), dtype=np.int16)
                for i, channel in enumerate(row[category]):
                    try:
                        out[i] = int(channel)
                    except ValueError:
                        # FIXME: this does not work correctly when
                        #        loading using PyTorch io.read_image
                        mask_fname = os.path.join(self.directory, 'images',
                                                  channel)
                        rgb_imread = imread(mask_fname)
                        out[i] = rgb_imread[:, :, 0] + \
                            rgb_imread[:, :, 1] * 256

                masks[category] = out

        masks = BrodenConceptMask(masks, categories)

        # TODO: there must be a better way to do this
        if self.target_concept:
            if isinstance(self.target_concept, BrodenConcept):
                masks_list = [masks.get_concept_mask(self.target_concept)]
            else:
                masks_list = [masks.get_concept_mask(c) for c in
                              self.target_concept]
            masks_list = [torch.from_numpy(m) for m in masks_list]
            return torch.stack(masks_list, dim=0)
        else:
            return masks

    def __getitem__(self, idx):
        if not self.skip_image:
            img_arr = self._get_image(idx)
        else:
            img_arr = 0

        if not self.skip_masks:
            mask_arr = self._get_masks(idx)
        else:
            mask_arr = None

        if self.return_index:
            return idx, img_arr, mask_arr
        else:
            return img_arr, mask_arr

    def compute_balance_weights(self, targets: List[WordnetConcept],
                                batch_size: int = 32,
                                act: torch.Tensor = None,
                                verbose: bool = False,
                                steepness: float = 1.0,
                                nw: int = 0) -> torch.Tensor:
        self.target_concept = targets
        self.return_index = True

        # Ratio of positives to total per image
        ratio = torch.zeros(len(self), len(targets))

        loader = torch.utils.data.DataLoader(self, batch_size=batch_size,
                                             shuffle=False, num_workers=nw)
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


class BrodenOntology(Ontology):
    def __init__(self, directory,
                 alignment_fn='broden_wordnet_alignment.csv',
                 vanilla_nd=False):

        # Read alignment file
        with open(os.path.join(directory, alignment_fn),
                  'r') as fp:
            csv_reader = DictReader(fp)
            alignment = [row for row in csv_reader]

        # Directly annotated concepts
        nodes = {r['WordNet ID']: BrodenConcept(r['WordNet ID'])
                 for r in alignment}

        # Assign Broden labels to WordNet concepts
        for concept in nodes.values():
            concept.b_ids = {int(r['Broden ID']) for r in alignment
                             if r['WordNet ID'] == concept.name}
            concept.original_b_ids = concept.b_ids.copy()

        # Construct hypernymy
        raw_ontology = []
        synsets = [c.synset for c in nodes.values()]
        while len(synsets) != 0:
            new_synsets = [ss.hypernyms() for ss in synsets]
            for hyponym, hypernyms in zip(synsets, new_synsets):
                for hypernym in hypernyms:
                    raw_ontology.append((synset_to_id(hypernym),
                                         synset_to_id(hyponym)))
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
        if not vanilla_nd:
            for synset in nodes:
                concept = nodes[synset]
                for descendant in concept.get_descendants():
                    concept.b_ids |= descendant.b_ids

        # Init superclass
        super().__init__(root)


class BrodenConceptMask(ConceptMask):
    def __init__(self, dict_example, categories):
        # A 'dict_example' is a dictionary for
        # each image with the following keys:
        #   fn:str  filename
        #   i:int   unique index
        #   sh:int  map height resolution
        #   sw:int  map width resolution
        #   color:np.Array  array with shape (sh, sw) containing
        #                   in each position the index of the
        #                   given color in the original image.
        #                   The same holds for the keys object,
        #                   part, scene and texture.
        self.dict_example = dict_example
        self.categories = categories

    @property
    def index(self):
        return self.dict_example['i']

    @property
    def shape(self):
        return (self.dict_example['sh'], self.dict_example['sw'])

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

        # Retrieve unique broden ids
        b_ids = [scalar for scalar in scalars]
        for p in pixels:
            b_ids += list(np.argwhere(np.bincount(p.ravel()) > 0)[:, 0])
        # '0' is not a broden id
        b_ids = {i for i in b_ids if i != 0}

        return b_ids

    def select_concepts(self, concepts):
        # Broden IDs contained in the image
        b_ids = self.get_broden_ids()

        # Select concepts with relevant broden IDs
        selected_concepts = [c for c in concepts if c.b_ids & b_ids]

        return selected_concepts

    def get_concept_mask(self, concept=None, c_mask=None, broden_id=None):
        '''
        Given a concept and an image
        returns the concept map L_c(x)
        '''

        # Init mask
        if c_mask is None:
            c_mask = np.empty(self.shape, dtype=bool)
        c_mask &= False

        # Broden IDs
        if concept:
            b_ids = concept.b_ids
        elif isinstance(broden_id, int):
            b_ids = {broden_id}
        else:
            b_ids = set(broden_id)

        # TODO: this should be fixed for Broden-like
        #       datasets that use different categories
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
