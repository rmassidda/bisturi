from bisturi import semalign as sa
from bisturi.circuits import retrieve_circuits
from bisturi.circuits import sim_functions
from bisturi.circuits import unique_concepts
from bisturi.dataset.broden import BrodenDataset
from bisturi.dataset.broden import BrodenOntology
from bisturi.dataset.imagenet import ImageNetDataset
from bisturi.dataset.imagenet import ImageNetOntology
from bisturi.model import load_model
from bisturi.semalign import SigmaMeasure
from matplotlib import pyplot as plt
from scipy.interpolate import pchip
import argparse
import numpy as np
import os


# Plot size
plt.rcParams["figure.figsize"] = (9.5, 4.5)
plt.rcParams.update({'font.size': 20})


def clean_arrays(x, y):
    # Remove NaNs
    idx = ~np.isnan(y)
    x, y = x[idx], y[idx]
    # Remove quasi-zeros
    idx = y > 1e-3
    x, y = x[idx], y[idx]
    return x, y


if __name__ == '__main__':
    # Command line
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help='Location of the Anonlib data cache')
    parser.add_argument('model_name', help='Name of the model to analyze')
    parser.add_argument('--dataset', help='Dataset to use',
                        choices=['broden', 'imagenet'], default='broden')
    parser.add_argument('--gap', type=float, default=0.1)
    parser.add_argument('--samples', help='Number of samples',
                        default=8, type=int)
    parser.add_argument('--sim_f',
                        help='Similarity function to build circuits',
                        choices=['jcn', 'lin'], default='jcn')
    parser.add_argument('--tau_a', help='Tau threshold',
                        default=0.04, type=float)
    parser.add_argument('--tau_b', help='Tau threshold',
                        default=0.25, type=float)
    parser.add_argument('--gpu', help='Use GPU for forward passes',
                        action='store_true')
    parser.add_argument('--nw', help='Number of workers in multiprocessing',
                        default=4, type=int)
    parser.add_argument('--bs', help='Batch size',
                        default=32, type=int)

    args = parser.parse_args()

    # Arguments
    base_dir = args.base_dir
    dataset_name = args.dataset
    model_name = args.model_name
    gpu = args.gpu
    gap = args.gap
    tau_a = args.tau_a
    tau_b = args.tau_b
    sim_f = args.sim_f
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
    model_path = os.path.join(
        base_dir, f'models/{model_name}_places365.pth.tar')
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
    activations = sa.record_activations(model,
                                        modules,
                                        dataset,
                                        batch_size=batch_size,
                                        cache=out_folder,
                                        gpu=gpu)

    # Compute thresholds
    thresholds = sa.compute_thresholds(activations,
                                       batch_size=batch_size,
                                       n_workers=nw,
                                       cache=out_folder)

    # Empty results
    results = {
        'circuits': {},
        'nonmono_circuits': {}
    }

    # Sigma measures
    measures = [SigmaMeasure.IOU, SigmaMeasure.LIKELIHOOD]
    tau_list = [tau_a, tau_b]

    # Semantic Alignment
    for sigma_measure, tau in zip(measures, tau_list):

        # sft (similarity function threshold)
        sft = np.linspace(gap, 1, samples)

        # Init empty lists
        results['circuits'][sigma_measure.name] = []
        results['nonmono_circuits'][sigma_measure.name] = []

        # Semantic alignment
        semalign = sa.compute_sigma(activations, dataset,
                                    thresholds, ontology,
                                    n_workers=nw, cache=out_folder,
                                    batch_size=batch_size,
                                    measure=sigma_measure)

        # Compute psi
        psi = sa.compute_psi(semalign, ontology, tau)

        # Iterate over SFT
        for sim_t in sft:
            print(sigma_measure, sim_f, sim_t)

            # Compute circuits
            out_circuits = os.path.join(out_folder,
                                        f'{sim_f}_{sim_t}_tau_{tau}_'
                                        f'{sigma_measure.name}.json')
            circuits = retrieve_circuits(psi,
                                         lambda a, b: sim_functions[sim_f](
                                             a, b) > sim_t,
                                         verbose=True,
                                         cache=out_circuits,
                                         ontology=ontology)

            # Retrieve non monosemantic circuits
            nonmono_circuits = [
                c for c in circuits if len(unique_concepts(c)) > 1]

            # Number of circuits
            results['circuits'][sigma_measure.name].append(
                len(circuits)
            )

            # Number of non monosemantic circuits
            results['nonmono_circuits'][sigma_measure.name].append(
                len(nonmono_circuits)
            )

        # To numpy array
        results['circuits'][sigma_measure.name] = np.array(
            results['circuits'][sigma_measure.name])
        results['nonmono_circuits'][sigma_measure.name] = np.array(
            results['nonmono_circuits'][sigma_measure.name])

    # Plot figures
    ylab_list = ['Circuits', 'Polysemantic Circuits']
    for ylab, metric in zip(ylab_list, results):
        # Filename of the plot
        fname = f'unit_similarity_circuits_' \
                f'{metric.strip().lower()}_' \
                f'{sim_f}_{tau_a}_{tau_b}_' \
                f'{gap}_{samples}.png'
        fname = os.path.join(out_folder, fname)

        # Alignment measures
        measure_a, measure_b = measures
        measure_a = measure_a.name
        measure_b = measure_b.name

        # IoU
        x_a = np.linspace(gap, 1, samples)
        y_a = results[metric][measure_a]
        x_a, y_a = clean_arrays(x_a, y_a)
        plt.plot(x_a, y_a, 'o', label='IoU', color='tab:blue')

        # lik
        x_b = np.linspace(gap, 1, samples)
        y_b = results[metric][measure_b]
        x_b, y_b = clean_arrays(x_b, y_b)
        plt.plot(x_b, y_b, 'o', label='L', color='tab:orange')

        # axis info
        plt.xlabel('Similarity Threshold')
        plt.ylabel(ylab)

        # Smoothen IoU
        if len(x_a) > 2:
            x_smooth_a = np.linspace(gap, max(x_a), 100)
            f_a = pchip(x_a, y_a)
            plt.plot(x_smooth_a, f_a(x_smooth_a), '--', color='tab:blue')

        # Smoothen L
        if len(x_b) > 2:
            x_smooth_b = np.linspace(gap, max(x_b), 100)
            f_b = pchip(x_b, y_b)
            plt.plot(x_smooth_b, f_b(x_smooth_b), '--', color='tab:orange')

        # Legend
        plt.legend(loc='best')

        # Store to file
        plt.savefig(fname, bbox_inches='tight')
        print(fname, 'stored')

        # Clear
        plt.clf()
