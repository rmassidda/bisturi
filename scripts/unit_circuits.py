from bisturi import semalign as sa
from bisturi.circuits import overview_circuit
from bisturi.circuits import report_circuits
from bisturi.circuits import retrieve_circuits
from bisturi.circuits import sim_functions
from bisturi.circuits import smallest_dag
from bisturi.circuits import unique_concepts
from bisturi.dataset.broden import BrodenDataset
from bisturi.dataset.broden import BrodenOntology
from bisturi.dataset.imagenet import ImageNetDataset
from bisturi.dataset.imagenet import ImageNetOntology
from bisturi.influence import accuracy_drop
from bisturi.model import load_model
from bisturi.semalign import SigmaMeasure
from bisturi.util import top_k_accuracy
from collections import Counter
from matplotlib import pyplot as plt
from tabulate import tabulate
from torchvision import transforms, io
from torchvision.datasets.places365 import Places365
import argparse
import numpy as np
import os
import torch


def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)


if __name__ == '__main__':
    # Command line
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help='Location of the data cache')
    parser.add_argument('model_name', help='Name of the model to analyze')
    parser.add_argument('--tau', help='Threshold to filter out concepts',
                        default=0.2, type=float)
    parser.add_argument('--dataset', help='Dataset to use',
                        choices=['broden', 'imagenet'], default='broden')
    parser.add_argument('--gpu', help='Wheter to try to use GPU',
                        action='store_true')
    parser.add_argument('--sigma', help='Semantic alignment measure',
                        choices=['LIKELIHOOD', 'IOU'], default='LIKELIHOOD')
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
    measure = SigmaMeasure[args.sigma]
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

    # Avoid last layer ablation
    skip_module = modules[-1]

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

    # Compute thresholds
    print('Learning Bias as in NetDissect')
    thresholds = sa.compute_thresholds(activations,
                                       batch_size=batch_size,
                                       n_workers=nw,
                                       cache=out_folder)
    for module in thresholds:
        print('thresholds', module, thresholds[module].shape)

    # Semantic Alignment
    print(f'Semantic Alignment with {measure.name}')
    semalign = sa.compute_sigma(activations, dataset,
                                thresholds, ontology,
                                n_workers=nw, cache=out_folder,
                                batch_size=batch_size,
                                measure=measure)
    for module in semalign:
        print('semalign', module, semalign[module].shape)

    # Compute PSI
    psi = sa.compute_psi(semalign, ontology, tau)
    print(tabulate(sa.report_psi(psi), headers='keys'))

    def sim(a, b):
        return sim_functions[sim_e](a, b)

    # Compute circuits
    out_circuits = os.path.join(out_folder,
                                f'{sim_f}_{sim_t}_tau_{tau}_'
                                f'{measure.name}.json')
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

    # Visualize units
    for e in target_circuit:
        print(e)

    # Select units to ablate (Except from the last layer)
    ablate_units = {module: [u for (m, u), c in target_circuit if m == module]
                    for module in modules if module != skip_module}

    # Load Places365 dataset
    custom_places365_normalizer = transforms.Compose([
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        ])

    def custom_places365_loader(path):
        img_arr = io.read_image(path)

        # Repeat channels for gray images
        if (img_arr.shape[0] == 1):
            img_arr = torch.repeat_interleave(img_arr, 3, axis=0)

        # Remove channels for RGBA
        if (img_arr.shape[0] == 4):
            img_arr = img_arr[:3, :, :]

        img_arr = custom_places365_normalizer(img_arr)

        return img_arr

    # Check if already extracted
    download_places365 = not os.path.exists(os.path.join(base_dir,
                                                         'datasets/places365/',
                                                         'val_256'))

    places365 = Places365(root=os.path.join(base_dir, 'datasets/places365/'),
                          split='val',
                          small=True,
                          download=download_places365,
                          loader=custom_places365_loader)
    idx_to_class, class_to_idx = places365.load_categories()

    # Ablation study for a given circuit
    ad_cache = os.path.join(out_folder, 'places365_forward.pt')
    ad = accuracy_drop(model, places365, ablate_units, batch_size=batch_size,
                       nw=nw, cache=ad_cache, gpu=gpu, verbose=True)

    # Results
    y_gt, y, y_gt_a, y_a = ad

    print('y_gt', y_gt.shape)
    print('y', y.shape)
    print('y_gt_a', y_gt_a.shape)
    print('y_a', y_a.shape)

    # Check matching ground truths
    assert torch.all(y_gt == y_gt_a)

    # Overall accuracy
    accuracy = top_k_accuracy(y_gt, y, k=5)
    print('Original', accuracy)

    accuracy = top_k_accuracy(y_gt_a, y_a, k=5)
    print('Ablated', accuracy)

    # Per class accuracy
    classes = set([y_i.item() for y_i in y_gt])
    perclass_index = {c: torch.nonzero(y_gt == c).squeeze() for c in classes}
    perclass_acc = [{
                     'Class_ID': c,
                     'Class': idx_to_class[c],
                     'Original': top_k_accuracy(y_gt[perclass_index[c]],
                                                y[perclass_index[c]]),
                     'Ablated': top_k_accuracy(y_gt[perclass_index[c]],
                                               y_a[perclass_index[c]])
                     } for c in classes]

    perclass_acc = [{**entry, 'Drop': entry['Ablated'] - entry['Original']}
                    for entry in perclass_acc]
    perclass_acc = sorted(perclass_acc, key=lambda e: e['Drop'])

    print(tabulate(perclass_acc, headers='keys'))

    # Plot per class accuracy
    plt.rcParams.update({'font.size': 22})
    figure = plt.figure(figsize=(10, 8))
    signed_drop = np.array([entry['Drop'] for entry in perclass_acc])
    x_axis = np.arange(min(signed_drop), max(signed_drop), 0.01)
    plt.hist(signed_drop, bins=30)
    plt.xlabel('Accuracy variation', fontsize=30)
    plt.ylabel('Class count', fontsize=30)
    out_figure = os.path.join(out_folder,
                              f'{sim_f}_{sim_t}_tau_{tau}_'
                              f'{measure.name}_{target_c_idx}.png')
    plt.savefig(out_figure)
    print(f'Figure stored in {out_figure}')

    # Most popular concept in circuit
    concepts = [c for (m, u), c in target_circuit]
    target_concept = most_common(concepts)

    print('Most popular concept:', target_concept)
    print(concepts.count(target_concept), 'over', len(concepts))

    ablate_units = {module: [u for (m, u), c in target_circuit if m == module
                             and c == target_concept]
                    for module in modules if module != skip_module}

    # Ablation study for a single concept
    ad_cache = os.path.join(out_folder, 'places365_forward.pt')
    ad = accuracy_drop(model, places365, ablate_units, batch_size=batch_size,
                       nw=nw, cache=ad_cache, gpu=gpu, verbose=True)

    # Results
    y_gt, y, y_gt_a, y_a = ad

    print('y_gt', y_gt.shape)
    print('y', y.shape)
    print('y_gt_a', y_gt_a.shape)
    print('y_a', y_a.shape)

    # Check matching ground truths
    assert torch.all(y_gt == y_gt_a)

    # Overall accuracy
    accuracy = top_k_accuracy(y_gt, y, k=5)
    print('Original', accuracy)

    accuracy = top_k_accuracy(y_gt_a, y_a, k=5)
    print('Ablated', accuracy)

    # Per class accuracy
    classes = set([y_i.item() for y_i in y_gt])
    perclass_index = {c: torch.nonzero(y_gt == c).squeeze() for c in classes}
    perclass_acc = [{
                     'Class_ID': c,
                     'Class': idx_to_class[c],
                     'Original': top_k_accuracy(y_gt[perclass_index[c]],
                                                y[perclass_index[c]]),
                     'Ablated': top_k_accuracy(y_gt[perclass_index[c]],
                                               y_a[perclass_index[c]])
                     } for c in classes]

    perclass_acc = [{**entry, 'Drop': entry['Ablated'] - entry['Original']}
                    for entry in perclass_acc]
    perclass_acc = sorted(perclass_acc, key=lambda e: e['Drop'])

    print(tabulate(perclass_acc, headers='keys'))

    # Plot per class accuracy
    plt.rcParams.update({'font.size': 22})
    figure = plt.figure(figsize=(10, 8))
    signed_drop = np.array([entry['Drop'] for entry in perclass_acc])
    x_axis = np.arange(min(signed_drop), max(signed_drop), 0.01)
    plt.hist(signed_drop, bins=30)
    plt.xlabel('Accuracy variation', fontsize=30)
    plt.ylabel('Class count', fontsize=30)
    out_figure = os.path.join(out_folder,
                              f'{sim_f}_{sim_t}_tau_{tau}_'
                              f'{measure.name}_{target_c_idx}_control.png')
    plt.savefig(out_figure)
    print(f'Figure stored in {out_figure}')
