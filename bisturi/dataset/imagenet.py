from bisturi.ontology import Ontology, WordnetConcept, ConceptMask
from bisturi.util import sigmoid, reshape_concept_mask
from torchvision import transforms, io
from tqdm import tqdm
from typing import List, Union
import json
import numpy as np
import os
import sys
import torch


class ImageNetConcept(WordnetConcept):
    """
    Represents a concept in the
    ImageNetConcept ontology.
    """
    def __init__(self, name, hypernyms=None, hyponyms=None):
        super().__init__(name, hypernyms, hyponyms)

        # By default concept as non propagated
        self.is_propagated_flag = False

    def is_propagated(self) -> bool:
        """
        Returns True if the concept
        is obtained via its children
        and does not have any original
        b_id associated
        """
        return self.is_propagated_flag


class ImageNetDataset(torch.utils.data.Dataset):
    """
    Returns the images contained in
    an annotated ImageNet dataset.
    """
    def __init__(self, directory: str,
                 fname: str = 'index_224_center.json',
                 reverse_index: str = 'reverse_index.json',
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225],
                 skip_image: bool = False,
                 skip_masks: bool = False,
                 target_concept: Union[WordnetConcept,
                                       List[WordnetConcept]] = None,
                 ontology: Ontology = None,  # TODO: should be ImageNetOntology
                 return_index: bool = False):

        index = os.path.join(directory, fname)
        with open(index, 'r') as fp:
            self.index = json.load(fp)

        # Store preferences
        self.skip_image = skip_image
        self.skip_masks = skip_masks
        self.target_concept = target_concept
        self.return_index = return_index

        self.data_directory = os.path.dirname(index)

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

        # Concepts from IDs in the reverse index
        self.reverse_index = {ontology.nodes[int(k)]: v for k, v in
                              self.reverse_index.items()}

        # If a concept is not in the reverse index
        # then it is propagated
        for concept_id in ontology.nodes:
            concept = ontology.nodes[concept_id]
            if concept in self.reverse_index:
                concept.is_propagated_flag = False
            else:
                concept.is_propagated_flag = True

    def _build_reverse_index(self, reverse_index_path: str):
        # Notify the user
        print('Missing reverse index, bulding it...',
              file=sys.stderr)

        # Compute reverse index
        rev_index = {}
        # Images within the dataset
        for idx, (_, label) in tqdm(enumerate(self), total=len(self)):
            # Synsets within the image
            for synset in label.synsets:

                # Retrieve concept ID from WordNet name
                concept_id = int('1'+synset[1:])

                # Create entry if new concept
                if concept_id not in rev_index:
                    rev_index[concept_id] = []

                # Insert image
                rev_index[concept_id].append(idx)

        # Store to file
        with open(reverse_index_path, 'w') as fp:
            json.dump(rev_index, fp, indent=2)

        return rev_index

    def instances(self, concept: ImageNetConcept):
        return self.reverse_index[concept.id]

    def __len__(self):
        return len(self.index)

    def _get_image(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.index[idx]
        img_path = image['path']
        img_arr = io.read_image(os.path.join(self.data_directory, img_path))

        # Repeat channels for gray images
        if (img_arr.shape[0] == 1):
            img_arr = torch.repeat_interleave(img_arr, 3, axis=0)

        # Remove channels for RGBA
        if (img_arr.shape[0] == 4):
            img_arr = img_arr[:3, :, :]

        img_arr = self.normalizer(img_arr)

        return img_arr

    def _get_masks(self, idx):
        masks = ImageNetExample(self.index[idx])

        if self.target_concept:
            if isinstance(self.target_concept, WordnetConcept):
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
        synsets = hypernyms.union(hyponyms)
        root_syn = list(hypernyms - hyponyms)[0]

        # Build nodes
        nodes = {s: ImageNetConcept(s) for s in synsets}

        # Connect nodes
        for hypernym, hyponym in raw_ontology:
            nodes[hypernym].hyponyms += [nodes[hyponym]]
            nodes[hyponym].hypernyms += [nodes[hypernym]]

        # Identify root
        root = nodes[root_syn]

        # Init superclass
        super().__init__(root)


class ImageNetExample(ConceptMask):
    def __init__(self, dict_example):
        self.dict_example = dict_example
        self.synsets = set([e[0] for e in self.dict_example['boxes']])

    @property
    def index(self):
        return self.dict_example['idx']

    @property
    def shape(self):
        return (self.dict_example['height'], self.dict_example['width'])

    def select_concepts(self, concepts):
        # Synset name descendants for each concept
        descendants = {c: {d.name for d in c.descendants} for c in concepts}

        # Select only concepts whose descendants contain at least
        # one of the reported synsets
        selection = [c for c in concepts if descendants[c] & self.synsets]

        return selection

    def _get_boxes(self, synset):
        return [e[1] for e in self.dict_example['boxes'] if e[0] == synset]

    def get_concept_mask(self, concept, c_mask=None):
        '''
        Given a concept and an image
        returns the concept map L_c(x)
        '''

        if c_mask is None:
            c_mask = np.empty(self.shape, dtype=bool)

        # Init mask
        c_mask &= False

        # Synset name of the descendants of the concept
        descendants = {c.name for c in concept.descendants}

        # Intersection with synset names annotated in the image
        synsets = descendants & self.synsets

        # Synsets contained in the image
        for synset in synsets:
            # Construct bounding boxes
            # for the synset
            for box in self._get_boxes(synset):
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
