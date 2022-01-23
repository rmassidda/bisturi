from bisturi.cav import TCAV
from bisturi.circuits import overview_circuit
from bisturi.circuits import report_circuits
from bisturi.circuits import retrieve_circuits
from bisturi.circuits import sim_functions
from bisturi.circuits import smallest_dag
from bisturi.circuits import unique_concepts
from bisturi.semalign import moduleid_to_string
from bisturi.semalign import report_psi
from bisturi.dataset.broden import BrodenDataset, BrodenOntology
from bisturi.dataset.imagenet import ImageNetDataset, ImageNetOntology
from bisturi.model import load_model
from tabulate import tabulate
import argparse
import json
import numpy as np
import os
import pickle
import random
import torch


if __name__ == '__main__':
    # Command line
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help='Location of the Anonlib data cache')
    parser.add_argument('model_name', help='Name of the model to analyze')
    parser.add_argument('--tau', help='Threshold to filter out concepts',
                        default=0.7, type=float)
    parser.add_argument('--dataset', help='Dataset to use',
                        choices=['broden', 'imagenet'], default='broden')
    parser.add_argument('--gpu', help='Wheter to try to use GPU',
                        action='store_true')
    parser.add_argument('--sim_f', help='Function to build circuits',
                        choices=['jcn', 'lin'], default='jcn')
    parser.add_argument('--sim_t', help='Threshold to test similarity when'
                                        'building circuits',
                        default=0.4, type=float)
    parser.add_argument('--sim_e', help='Function to evaluate circuits',
                        choices=['jcn', 'lin'], default='lin')
    parser.add_argument('--nw', help='Number of workers in multiprocessing',
                        default=64, type=int)
    parser.add_argument('--bs', help='Batch size',
                        default=32, type=int)
    parser.add_argument('--c_idx', help='Index of the target circuit',
                        default=-1, type=int)

    args = parser.parse_args()

    # Arguments
    base_dir = args.base_dir
    model_name = args.model_name
    gpu = args.gpu
    tau = args.tau
    dataset_name = args.dataset
    sim_f = args.sim_f
    sim_e = args.sim_e
    sim_t = args.sim_t
    nw = args.nw
    batch_size = args.bs
    target_c_idx = args.c_idx

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

    def find_concept(concept, ontology):
        for c in ontology.nodes:
            if concept == ontology.nodes[c].synset.name():
                return ontology.nodes[c]
        return None

    def load_CAV_dict(path: str):
        with open(path, 'rb') as fp:
            # TODO: avoid storing module?
            module, cav_dict = pickle.load(fp)
            return cav_dict

    def filter_CAVs(module, cav_dict, tau: float = 0.5, verbose=False):
        # Valid concepts in the first module
        cav_dict = {c: cav_dict[c] for c in cav_dict
                    if cav_dict[c]['val_stat']['f1'] > tau}

        if verbose:
            print(json.dumps(cav_dict, indent=2, default=str))

        # $\Psi$ subset of the first module
        psi = {module: {idx: [(ontology.nodes[cid],
                               cav_dict[cid]['val_stat']['f1'])]
                        for idx, cid in enumerate(cav_dict)}}

        return psi

    # Load modules list and cavs
    cavs = [load_CAV_dict(os.path.join(out_folder,
                                       f'{moduleid_to_string(module)}_'
                                       f'cavs.pt'))
            for module in modules]

    # $Psi$ of the overall network
    psi = {}
    for module, cav in zip(modules, cavs):
        psi.update(filter_CAVs(module, cav, tau))
    print(tabulate(report_psi(psi), headers='keys'))

    def sim(a, b):
        try:
            return sim_functions[sim_e](a, b)
        except ZeroDivisionError:
            # NOTE: bug in the implementation of Lin similarity!
            #       This fix doesn't work for similarity measures
            #       that do not ensure sim(c,c) == 1.
            #       Similar to the issue for LCH described in:
            #           https://github.com/nltk/nltk/issues/301
            assert a == b
            return 1

    # Compute circuits
    out_circuits = os.path.join(out_folder,
                                f'{sim_f}_{sim_t}_tau_{tau}_CAV_circuits.json')
    circuits = retrieve_circuits(psi,
                                 lambda a, b: sim_functions[sim_f](
                                     a, b) > sim_t,
                                 verbose=True,
                                 cache=out_circuits,
                                 ontology=ontology)
    rep_circuits = [report_circuits(circuits, sim=sim)]
    circuits = [c for c in circuits if len(unique_concepts(c)) > 1]
    rep_circuits += [report_circuits(circuits, sim=sim)]

    print('# Circuits report')
    print(tabulate(rep_circuits, headers='keys'))

    # Report circuits
    print(tabulate([{'ID': c_idx, **overview_circuit(c, sim=sim)}
                    for c_idx, c in enumerate(circuits)], headers='keys'))

    # Select circuit
    if target_c_idx < 0:
        target_c_idx = int(input('Select Circuit > '))
    target_circuit = circuits[target_c_idx]

    # Visualize smallest DAG
    print(tabulate(smallest_dag(target_circuit), headers='keys'))

    # Randomize circuit
    randomize = input('Randomize? y/[n] >') == 'y'

    # Print circuit
    if randomize:
        print(f'### Circuit n.{target_c_idx} (randomized)')
        runs = 10
    else:
        print(f'### Circuit n.{target_c_idx}')
        runs = 1

    average_cav = []
    for _ in range(runs):

        # Eventually generate random circuit
        if randomize:
            concepts_per_module = {module: [psi[module][u][0][0] for u
                                            in psi[module]]
                                   for module in psi}
            circuit_random = []
            for entry in target_circuit:
                direction, concept = entry
                module, _ = direction
                concept = random.choice(concepts_per_module[module])
                circuit_random.append((direction, concept))
            target_circuit = circuit_random

        # Print circuit
        for direction, concept in target_circuit:
            print(direction[0][0], concept)

        # Compute TCAV
        for a_idx, node_a in enumerate(target_circuit):
            module_a = node_a[0][0]
            concept_a = node_a[1]
            next_a = None

            # Load CAVs dict
            cav_dict_a = cavs[modules.index(module_a)][concept_a.id]
            print('\n', module_a, concept_a)
            for k in cav_dict_a['val_stat']:
                print('--', k, cav_dict_a['val_stat'][k])

            for b_idx, node_b in enumerate(target_circuit[a_idx+1:]):
                module_b = node_b[0][0]
                concept_b = node_b[1]
                # Only if different from the module of the first node
                # and eventually recognize only the subsequent one
                if module_a != module_b and (next_a is None or
                                             module_b == next_a):
                    # Set next_module
                    next_a = module_b

                    # Load CAVs dict
                    cav_dict_a = cavs[modules.index(module_a)][concept_a.id]
                    cav_dict_b = cavs[modules.index(module_b)][concept_b.id]

                    # Load CAV from disk
                    cav_a = cav_dict_a['cav']
                    cav_b = cav_dict_b['cav']

                    # Numpy -> torch.Tensor
                    if isinstance(cav_a, np.ndarray):
                        cav_a = torch.from_numpy(cav_a)
                    if isinstance(cav_b, np.ndarray):
                        cav_b = torch.from_numpy(cav_b)

                    # Remove unit dimensions
                    cav_a = cav_a.squeeze()
                    cav_b = cav_b.squeeze()

                    # Eventually load on GPU
                    if gpu:
                        cav_a = cav_a.cuda()
                        cav_b = cav_b.cuda()

                    # Configure dataset
                    dataset.target_concept = concept_b
                    dataset.return_index = True

                    # Create subset
                    indices = cav_dict_b['train_set']
                    indices += cav_dict_b['val_set']
                    subset = torch.utils.data.Subset(dataset, indices)

                    # Compute TCAV
                    tcav = TCAV(model, subset, module_a, cav_a, module_b,
                                cav_b, verbose=False, gpu=gpu, nw=nw,
                                batch_size=batch_size)

                    # Print results
                    print(module_a, concept_a, '--->',
                          module_b, concept_b, tcav)

                    # Store result
                    average_cav.append(tcav)

    # Report average CAV
    average_cav = sum(average_cav) / len(average_cav)
    print(average_cav)
