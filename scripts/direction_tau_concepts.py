from bisturi.dataset.broden import BrodenDataset
from bisturi.dataset.broden import BrodenOntology
from bisturi.dataset.imagenet import ImageNetDataset
from bisturi.dataset.imagenet import ImageNetOntology
from bisturi.model import load_model
from bisturi.semalign import moduleid_to_string
from matplotlib import pyplot as plt
from scipy.interpolate import pchip
from tqdm import tqdm
import argparse
import numpy as np
import pickle
import os


# Plot size
plt.rcParams["figure.figsize"] = (5.5, 4)


def clean_arrays(x, y):
    # Remove NaNs
    idx = ~np.isnan(y)
    x, y = x[idx], y[idx]
    # Remove quasi-zeros
    idx = y > 1e-3
    x, y = x[idx], y[idx]
    return x, y


def load_CAV_dict(path: str):
    with open(path, 'rb') as fp:
        return pickle.load(fp)[1]


def filter_CAVs(module, cav_dict, tau: float = 0.5, verbose=False):
    # Valid concepts in the first module
    cav_dict = {c: cav_dict[c] for c in cav_dict
                if cav_dict[c]['val_stat']['f1'] > tau}

    # $\Psi$ subset of the first module
    psi = {module: {idx: [(ontology.nodes[cid],
                           cav_dict[cid]['val_stat']['f1'])]
                    for idx, cid in enumerate(cav_dict)}}

    return psi


def figure_data(tau_series, modules):
    # Load cache
    cavs = [load_CAV_dict(os.path.join(
      out_folder, f'{moduleid_to_string(module)}_cavs.pt'))
            for module in modules]

    # Modules to analyze
    modules = modules

    results = {m: {
        'unique': [],
        'propagation': [],
        'total': []

    } for m in modules + ['Aggregated']}

    for tau in tqdm(tau_series):

        # Per module-psi
        psi = {}
        for module, cav in zip(modules, cavs):
            psi.update(filter_CAVs(module, cav, tau))

        # Compute aggregation
        aggregated = {}
        for module in psi:
            shift = max(aggregated.keys(), default=0) + 1
            extension = {u + shift: psi[module][u] for u in psi[module]}
            aggregated.update(extension)

        # Assign aggregated results
        psi['Aggregated'] = aggregated

        # Iterate all modules
        for module in psi:
            # Concepts aligned
            concepts = [[c for c, v in psi[module][u]] for u in psi[module]]
            concepts = sum(concepts, [])

            # Fig 1.a: number of unique concepts
            unique_concepts = set(concepts)
            results[module]['unique'].append(len(unique_concepts))

            # Fig 1.b: percentage of propagated
            if len(unique_concepts) == 0:
                results[module]['propagation'].append(np.nan)
            else:
                propagated = {c for c in unique_concepts
                              if c.is_propagated()}
                results[module]['propagation'].append(
                    len(propagated) / len(unique_concepts))

            # Fig 1.c: number of total concepts
            results[module]['total'].append(len(concepts))

    # Convert into numpy arrays
    for module in results:
        for key in results[module]:
            results[module][key] = np.array(results[module][key])

    return results


if __name__ == '__main__':
    # Command line
    parser = argparse.ArgumentParser(description='Plots for the paper')
    parser.add_argument('base_dir', help='Location of the Anonlib data cache')
    parser.add_argument('model_name', help='Name of the model to analyze')
    parser.add_argument('--samples', help='Number of samples',
                        default=20, type=int)
    parser.add_argument('--dataset', help='Dataset to use',
                        choices=['broden', 'imagenet'], default='broden')
    parser.add_argument('--gap', help='Initial gap after zero',
                        default=1e-2, type=float)
    parser.add_argument('--gpu', help='Use GPU for forward passes',
                        action='store_true')
    parser.add_argument('--nw', help='Number of workers in multiprocessing',
                        default=4, type=int)
    parser.add_argument('--bs', help='Batch size',
                        default=32, type=int)

    args = parser.parse_args()

    # Arguments
    base_dir = args.base_dir
    model_name = args.model_name
    gpu = args.gpu
    dataset_name = args.dataset
    gap = args.gap
    samples = args.samples
    nw = args.nw
    batch_size = args.bs

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
        ratio_lowest = 0.4817
    elif dataset_name == 'imagenet':
        dataset_path = os.path.join(base_dir, 'datasets/ilsvrc2011/out/')
        ontology = ImageNetOntology(dataset_path)
        dataset = ImageNetDataset(dataset_path, ontology=ontology)
        ratio_lowest = 0.4750

    tau_series = np.linspace(gap, 1, samples)

    results = figure_data(tau_series, modules)

    labels = [
        ('Tau', 'Concepts'),
        ('Tau', 'Ratio'),
        ('Tau', 'Total Concepts')
        ]

    for module in results:
        for measure_a, measure_b, lab in zip(results[module],
                                             results[module],
                                             labels):
            assert measure_a == measure_b
            measure = measure_a
            module_name = moduleid_to_string(module)
            fname = f'direction_tau_concepts_' \
                    f'{module_name.lower()}_{measure.strip().lower()}_' \
                    f'{gap}_{samples}.png'
            fname = os.path.join(out_folder, fname)

            # IoU
            x_a = tau_series.copy()
            y_a = results[module][measure]
            x_a, y_a = clean_arrays(x_a, y_a)
            plt.plot(x_a, y_a, 'o', label='F1', color='tab:blue')

            # Smoothen IoU
            if len(x_a) > 2:
                x_smooth_a = np.linspace(gap, max(x_a), 100)
                f_a = pchip(x_a, y_a)
                plt.plot(x_smooth_a, f_a(x_smooth_a), '--', color='tab:blue')

            # Half ratio
            if measure == 'propagation':
                plt.axhline(y=ratio_lowest, color='gray', linestyle='dashed')

            # Legend
            plt.legend(loc='best')

            # log scale
            if measure == 'total':
                plt.yscale('log')

            # axis info
            xlab, ylab = lab
            plt.xlabel(xlab)
            plt.ylabel(ylab)

            # Store to file
            plt.savefig(fname, bbox_inches='tight')
            print(fname, 'stored')

            # Clear
            plt.clf()
