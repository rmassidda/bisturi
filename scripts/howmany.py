from bisturi import semalign as sa
from bisturi.dataset.broden import BrodenDataset
from bisturi.dataset.broden import BrodenOntology
from bisturi.dataset.imagenet import ImageNetDataset
from bisturi.dataset.imagenet import ImageNetOntology
from bisturi.model import load_model
from bisturi.semalign import SigmaMeasure
from bisturi.semalign import moduleid_to_string
from matplotlib import pyplot as plt
from scipy.interpolate import pchip
from tqdm import tqdm
import argparse
import numpy as np
import os
import sys

base_dir = sys.argv[1]

dataset_path = os.path.join(base_dir, 'datasets/broden1_224/')
dataset = BrodenDataset(dataset_path)
ontology = BrodenOntology(dataset_path)
ontology_list = ontology.to_list()
propagated = [c for c in ontology_list if c.is_propagated()]
print('Broden', len(ontology_list), len(propagated), len(propagated) / len(ontology_list))

dataset_path = os.path.join(base_dir, 'datasets/ilsvrc2011/out/')
ontology = ImageNetOntology(dataset_path)
dataset = ImageNetDataset(dataset_path, ontology=ontology)
ontology_list = ontology.to_list()
propagated = [c for c in ontology_list if c.is_propagated()]
print('ImageNet', len(ontology_list), len(propagated), len(propagated) / len(ontology_list))
