from bisturi import semalign as sa
from bisturi.cav import CAV, eval_CAV
from bisturi.dataset import balanced_splits
from bisturi.dataset.broden import BrodenDataset, BrodenOntology
from bisturi.dataset.imagenet import ImageNetDataset, ImageNetOntology
from bisturi.model import load_model
from bisturi.ontology import Concept
from bisturi.loss import FocalLoss
from bisturi.semalign import moduleid_to_string
from multiprocessing import Pool, Manager, Queue
from tqdm import tqdm
from typing import Dict, List, Tuple, Union
import argparse
import os
import pickle
import torch
import numpy as np


if __name__ == '__main__':
    # Command line
    parser = argparse.ArgumentParser(description='Learn CAVs')
    parser.add_argument('base_dir', help='Location of the Anonlib data cache')
    parser.add_argument('model_name', help='Name of the model to analyze')
    parser.add_argument('--gpu', help='Wheter to try to use GPU',
                        action='store_true')
    parser.add_argument('--nw', help='Number of workers for CAV computation',
                        default=4, type=int)
    parser.add_argument('--loader_nw', help='Number of workers to load'
                                            ' dataset',
                        default=4, type=int)
    parser.add_argument('--bs', help='Batch size',
                        default=8, type=int)
    parser.add_argument('--samples', help='Number of samples',
                        default=64, type=int)
    parser.add_argument('--epochs', help='Number of epochs',
                        default=10, type=int)
    parser.add_argument('--steepness', help='Steepness of the sigmoid for the '
                                            'balancing weighting',
                        default=1.0, type=float)
    parser.add_argument('--dataset', help='Dataset to use',
                        choices=['broden', 'imagenet'], default='broden')
    parser.add_argument('--val', help='Size of the validation split',
                        default=0.2, type=float)

    args = parser.parse_args()

    # Arguments
    base_dir = args.base_dir
    model_name = args.model_name
    gpu = args.gpu
    nw = args.nw
    loader_nw = args.loader_nw
    batch_size = args.bs
    n_samples = args.samples
    epochs = args.epochs
    steepness = args.steepness
    dataset_name = args.dataset
    val_split = 1 - args.val

    # Eventually distribute over
    # different processes
    if nw > 0:
        pool = Pool(nw)
        manager = Manager()
        queue = manager.Queue()

    # Modules
    if model_name == 'alexnet':
        modules = [('features.8', 0),
                   ('features.10', 0),
                   ('classifier.1', 0),
                   ('classifier.4', 0),
                   ('classifier.6', 0)]
    elif model_name == 'resnet18':
        modules = [('layer4.0.conv1', 0),
                   ('layer4.0.conv2', 0),
                   ('layer4.0', 0),
                   ('layer4.1.conv1', 0),
                   ('layer4.1.conv2', 0),
                   ('layer4.1', 0),
                   ('fc', 0)]
    elif model_name == 'densenet161':
        modules = [('features.denseblock4.denselayer22', 0),
                   ('features.denseblock4.denselayer23', 0),
                   ('features.denseblock4.denselayer24', 0),
                   ('classifier', 0)]

    # Load model
    model_path = os.path.join(base_dir, f'models/{model_name}_'
                              'places365.pth.tar')
    model = load_model(f'{model_name}', path=model_path, GPU=gpu)
    out_folder = os.path.join(
        base_dir, f'results/{dataset_name}_places365/{model_name}')
    os.makedirs(out_folder, exist_ok=True)

    # Load Dataset
    if dataset_name == 'broden':
        dataset_path = os.path.join(base_dir, 'datasets/broden1_224/')
        dataset = BrodenDataset(dataset_path)
        ontology = BrodenOntology(dataset_path)
    elif dataset_name == 'imagenet':
        dataset_path = os.path.join(base_dir, 'datasets/ilsvrc2011/out/')
        ontology = ImageNetOntology(dataset_path)
        dataset = ImageNetDataset(dataset_path, ontology=ontology)

    # Record activations
    print('Recording activations')
    activations = sa.record_activations(model,
                                        modules,
                                        dataset,
                                        batch_size=batch_size,
                                        cache=out_folder,
                                        gpu=gpu)
    for module in activations:
        print('activations', module, activations[module].shape)

    # Retrieve list of concepts
    root_concept = ontology.root
    concepts = ontology.to_list(keep_placeholders=False)
    concepts = sorted(concepts, key=lambda c: c.id)

    for module in modules:
        print(f'Learning vectors for {module}')

        # Activations of the module
        mod_act = activations[module]

        # Path of weights and vectors
        cavs_path = os.path.join(
            out_folder, f'{moduleid_to_string(module)}_cavs.pt')
        weights_path = os.path.join(out_folder,
                                    f'{moduleid_to_string(module)}_balanced'
                                    '_sampling_weights.pt')

        # Try to load CAV from path
        try:
            with open(cavs_path, 'rb') as fp:
                mod2, cav_dict = pickle.load(fp)
                print(module, mod2, 'already learned')
                continue
        except FileNotFoundError:
            pass

        try:
            # Load tensor from file
            weights = torch.load(weights_path)
        except FileNotFoundError:
            # Precompute sampling weights
            weights = dataset.compute_balance_weights(concepts,
                                                      batch_size=batch_size,
                                                      act=mod_act,
                                                      verbose=True,
                                                      nw=loader_nw,
                                                      steepness=steepness)
            # Store tensor to file
            torch.save(weights, weights_path)

        print('Weights for balanced sampling:', weights.shape)

        # Compute indices
        subset_indices = [balanced_splits(weights[:, c_idx],
                                          n_samples=n_samples)
                          for c_idx, _ in enumerate(concepts)]
        # Train indices
        train_idx = [t for t, v, _ in subset_indices]
        # Validation indices
        val_idx = [v for t, v, _ in subset_indices]

        # Compute CAVs for each concept
        if nw == 0:
            list_cavs = [train_CAV([concept], mod_act, dataset, [train],
                                   [val], gpu, epochs, batch_size, loader_nw)
                         for concept, train, val
                         in tqdm(zip(concepts, train_idx, val_idx),
                                 total=len(concepts))]
            list_cavs = sum(list_cavs, [])
        else:

            chunk_size = len(concepts) // nw
            params = []
            for chunk_idx in range(nw):
                start = chunk_idx*chunk_size
                if chunk_idx == nw - 1:
                    end = len(concepts)
                else:
                    end = (chunk_idx+1)*chunk_size
                chunk_range = slice(start, end)

                act_tuple = (mod_act.filename, mod_act.shape)
                params.append((concepts[chunk_range], act_tuple, dataset,
                               train_idx[chunk_range], val_idx[chunk_range],
                               gpu, epochs, batch_size, 0, False, queue))

            list_cavs = pool.starmap_async(train_CAV, params)
            # Progress bar
            pbar = tqdm(total=len(concepts))
            count_ended = 0
            while count_ended != len(concepts):
                _ = queue.get()
                pbar.update(1)
                count_ended += 1
            pbar.close()

            list_cavs = list_cavs.get()
            list_cavs = sum(list_cavs, [])

        # Dump the results in a dictionary
        name_dict = {concept.id: cav_dict
                     for concept, cav_dict in zip(concepts, list_cavs)}
        with open(cavs_path, 'wb') as f:
            pickle.dump((module, name_dict), f)
