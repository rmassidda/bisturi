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


# Plot size
plt.rcParams["figure.figsize"] = (7.5, 7.5)
plt.rcParams.update({'font.size': 30})


def clean_arrays(x, y):
    # Remove NaNs
    idx = ~np.isnan(y)
    x, y = x[idx], y[idx]
    # Remove quasi-zeros
    idx = y > 1e-3
    x, y = x[idx], y[idx]
    return x, y


def figure_1_data(tau_series, semalign, modules):
    # Modules to analyze
    modules = modules + ['Aggregated']

    results = {m: {
        'unique': [],
        'propagation': [],
        'total': []

    } for m in modules}

    for tau in tqdm(tau_series):

        # Precompute PSI
        psi = sa.compute_psi(semalign, ontology, tau)

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

    # IoU Semantic Alignment
    print('Semantic Alignment with IOU')
    semalign = sa.compute_sigma(activations, dataset,
                                thresholds, ontology,
                                n_workers=nw, cache=out_folder,
                                batch_size=batch_size,
                                measure=SigmaMeasure.IOU)
    semiou = semalign
    for module in semiou:
        print('semiou', module, semiou[module].shape)

    # Likelihood Semantic Alignment
    print('Semantic Alignment with Likelihood')
    semalign = sa.compute_sigma(activations, dataset,
                                thresholds, ontology,
                                n_workers=nw, cache=out_folder,
                                batch_size=batch_size,
                                measure=SigmaMeasure.LIKELIHOOD)
    semlik = semalign
    for module in semlik:
        print('semlik', module, semlik[module].shape)

    # Figure 1:
    #   - (Tau, Unique Concepts)
    #   - (Tau, % Propagation)
    #   - (Tau, Total Concepts)
    tau_series = np.linspace(gap, 1, samples)

    results_iou = figure_1_data(tau_series, semiou, modules)
    results_lik = figure_1_data(tau_series, semlik, modules)

    labels = [
        ('Tau', 'Concepts'),
        ('Tau', 'Ratio'),
        ('Tau', 'Total Concepts')
        ]

    for module_a, module_b in zip(results_iou, results_lik):
        assert module_a == module_b
        module = module_a
        for measure_a, measure_b, lab in zip(results_iou[module],
                                             results_iou[module],
                                             labels):
            assert measure_a == measure_b
            measure = measure_a
            module_name = moduleid_to_string(module)
            fname = f'unit_tau_concepts_' \
                    f'{module_name.lower()}_{measure.strip().lower()}_' \
                    f'{gap}_{samples}.png'
            fname = os.path.join(out_folder, fname)

            # IoU
            x_a = tau_series.copy()
            y_a = results_iou[module][measure]
            x_a, y_a = clean_arrays(x_a, y_a)
            plt.plot(x_a, y_a, 'o', label='IoU', color='tab:blue')

            # lik
            if measure != 'propagation':
                x_b = tau_series.copy()
                y_b = results_lik[module][measure]
                x_b, y_b = clean_arrays(x_b, y_b)
                plt.plot(x_b, y_b, 'o', label='L', color='tab:orange')

            # Smoothen IoU
            if len(x_a) > 2:
                x_smooth_a = np.linspace(gap, max(x_a), 100)
                f_a = pchip(x_a, y_a)
                plt.plot(x_smooth_a, f_a(x_smooth_a), '--', color='tab:blue')

            # Smoothen L
            if len(x_b) > 2 and measure != 'propagation':
                x_smooth_b = np.linspace(gap, max(x_b), 100)
                f_b = pchip(x_b, y_b)
                plt.plot(x_smooth_b, f_b(x_smooth_b), '--', color='tab:orange')

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
