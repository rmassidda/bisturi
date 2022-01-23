from bisturi.loss import FocalLoss
from bisturi.model import get_module, ModuleID
from bisturi.ontology import Concept
from bisturi.util import reshape_concept_mask
from multiprocessing import Queue
from tqdm import tqdm
from typing import Callable, Dict, List, Tuple, Union
import numpy as np
import torch


def CAV(concept: Concept,
        dataset: torch.utils.data.Dataset,
        activations: np.ndarray,
        init: torch.Tensor = None,
        epochs: int = 10,
        criterion: Callable = None,
        optimizer: torch.optim.Optimizer = None,
        batch_size: int = 32,
        nw: int = 4,
        gpu: bool = False,
        replace: bool = False,
        verbose: bool = False) -> torch.Tensor:
    """
    Compute the concept activation vector (CAV) of a concept.
    """

    # Number of samples
    n_samples = len(dataset)
    if verbose:
        print(f'Training CAV on concept {concept} with {n_samples} samples')

    # Feature size
    n_features = activations.shape[1]

    # Linear model
    if len(activations.shape) > 2:
        cav = torch.nn.Conv2d(n_features, 1, kernel_size=1,
                              stride=1, padding=0, bias=True)
        if init is not None:
            cav.weight.data = init.clone().reshape(1, n_features, 1, 1)
    else:
        cav = torch.nn.Linear(n_features, 1, bias=True)
        if init is not None:
            cav.weight.data = init.clone().reshape(1, n_features)

    if gpu:
        cav.cuda()

    # Load dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         num_workers=nw)

    # Loss function
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = criterion()

    # Optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(cav.parameters(), lr=1e-3)

    for epoch in range(epochs):
        # Progress bar
        if verbose:
            loader = tqdm(loader)

        for batch in loader:

            # Split batch
            idx, _, y = batch

            # Ensure even a single index
            # as a list of indices
            idx = list(idx)

            # Boolean mask as float
            y = y.float()

            # Select activations
            a = torch.from_numpy(activations[idx]).float()

            # Coherent activations/concept shapes
            y = reshape_concept_mask(y, a)

            if gpu:
                a = a.cuda()
                y = y.cuda()

            # Compute loss
            y_bar = cav(a)
            loss = criterion(y_bar, y)
            loss.backward()

            # Verbose
            if verbose:
                loader.set_description(f'Loss: {loss.item():.4f}')

            # Update cav
            optimizer.step()
            optimizer.zero_grad()

    return cav


def eval_CAV(model: torch.nn.Module,
             concept: Concept,
             dataset: torch.utils.data.Dataset,
             activations: np.ndarray,
             batch_size: int = 32,
             nw: int = 4,
             gpu: bool = False,
             replace: bool = False,
             verbose: bool = False) -> torch.Tensor:
    """
    Evaluate the concept activation vector (CAV) of a concept.
    """

    # Number of samples
    n_samples = len(dataset)
    if verbose:
        print(f'Testing CAV on concept {concept} with {n_samples} samples')

    # Feature size
    if gpu:
        model.cuda()

    # Load dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=nw)

    # Progress bar
    if verbose:
        loader = tqdm(loader)

    # Initialize Statistics
    stats = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'pos': 0, 'neg': 0}
    n = 0

    for batch in loader:

        # Split batch
        idx, _, y = batch

        # Ensure even a single index
        # as a list of indices
        idx = list(idx)

        # Boolean mask as float
        y = y.float()

        # Select activations
        a = torch.from_numpy(activations[idx]).float()

        # Coherent activations/concept shapes
        y = reshape_concept_mask(y, a)

        if gpu:
            a = a.cuda()
            y = y.cuda()

        # Compute loss
        y_bar = model(a)

        # Boolean results
        y_bar = y_bar > 0
        y = y.bool()

        # Compare results
        eq = y_bar == y
        neq = torch.logical_not(eq)
        pos = y == 1
        neg = torch.logical_not(pos)

        # Update counters
        stats['TP'] += torch.count_nonzero(eq * pos)
        stats['TN'] += torch.count_nonzero(eq * neg)
        stats['FP'] += torch.count_nonzero(neq * neg)
        stats['FN'] += torch.count_nonzero(neq * pos)
        stats['pos'] += torch.count_nonzero(pos)
        stats['neg'] += torch.count_nonzero(neg)

    n = stats['pos'] + stats['neg']

    stats['recall'] = stats['TP'] / \
        (stats['TP'] + stats['FN'] + 1e-6)
    stats['precision'] = stats['TP'] / \
        (stats['TP'] + stats['FP'] + 1e-6)
    stats['f1'] = 2 * stats['precision'] * \
        stats['recall'] / (stats['precision'] +
                           stats['recall'] + 1e-6)
    stats['accuracy'] = (stats['TP'] + stats['TN']) / n

    # No need for the torch.Tensor overhead
    for key in stats:
        stats[key] = stats[key].item()

    return stats


def model_to_CAV(cav: torch.nn.Module) -> torch.Tensor:
    # NOTE: why not just cav.weight.data.squeeze()?
    if isinstance(cav, torch.nn.Conv2d):
        cav = cav.weight.data
        cav = cav.reshape(cav.shape[1])
    elif isinstance(cav, torch.nn.Linear):
        cav = cav.weight.data
        cav = cav.reshape(cav.shape[1])
    else:
        raise ValueError('CAV must be extracted by either a convolutional '
                         'layer or a linear layer.')

    return cav


def TCAV(model: torch.nn.Module,
         dataset: torch.utils.data.Dataset,
         layer_a: ModuleID,
         cav_a: Union[torch.nn.Module, torch.Tensor],
         layer_b: ModuleID,
         cav_b: Union[torch.nn.Module, torch.Tensor],
         verbose: bool = False,
         batch_size: int = 32,
         nw: int = 0,
         gpu: bool = False) -> float:
    """
    Estimates the TCAV value described in "Interpretability Beyond
    Feature Attribution: Quantitative Testing with Concept Activation
    Vectors (TCAV)" by Kim et al. (2017).

    In addition to the original paper, this implementation
    enables the estimate of TCAV between any arbitrary
    directions in the neural model.
    """
    # TODO: can this function be generalized to compute TCAV
    #       over an arbitrary number of directions in parallel?
    # TODO: can we generalize all the convolutional wired-in stuff?

    # Retrieve influencing CAV
    if isinstance(cav_a, torch.nn.Module):
        cav_a = model_to_CAV(cav_a)

    # Retrieve influenced CAV
    if isinstance(cav_b, torch.nn.Module):
        cav_b = model_to_CAV(cav_b)

    # Evaluate model
    model.eval()

    # Decompose the influencing direction
    module_name_a, iter_a = layer_a
    module_a = get_module(model, module_name_a)

    # Decompose the target direction
    module_name_b, iter_b = layer_b
    module_b = get_module(model, module_name_b)

    # Create hooks
    gradients = []
    outputs = []

    # Init variables
    n = 0
    pos = 0

    # Target class hook
    def target_hook(module, input, output):
        outputs.append(output)

    # Backward CAV hook
    def gradient_tensor_hook(grad):
        gradients.append(grad)

    # Forward CAV hook
    def gradient_hook(module, input, output):
        output.register_hook(gradient_tensor_hook)

    # Hook for the retrieval of the gradient
    hook_a = module_a.register_forward_hook(gradient_hook)

    # Hook for the target logit
    hook_b = module_b.register_forward_hook(target_hook)

    # Iterate over the dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=nw)

    # Progress bar
    if verbose:
        loader = tqdm(loader)

    for batch in loader:
        # Clean tensors
        del gradients[:]
        del outputs[:]

        # Separate inputs and targets
        _, x, y = batch

        # Eventually move to GPU
        if gpu:
            x = x.cuda()
            y = y.cuda()

        # Forward pass
        _ = model.forward(x)

        # Eventually select output
        output = outputs[iter_b]

        # Compute the logit
        if len(output.shape) == 2:
            # Fully connected layer
            output = output @ cav_b
        elif len(output.shape) == 4:
            # Convolutional layer
            output = torch.einsum('bfhw, f -> bhw', output, cav_b)
        else:
            raise ValueError('Output must be either a 2D or 4D tensor.')

        # Backward pass
        output.backward(torch.ones_like(output))

        # Retrieve the gradients
        g = gradients[iter_a]

        # Compute "conceptual sensitivity"
        if len(g.shape) == 2:
            # Fully connected layer
            S = g @ cav_a
            S = S.reshape(S.shape[0], 1)
        elif len(g.shape) == 4:
            # Convolutional layer
            S = torch.einsum('bfhw, f -> bhw', g, cav_a)
            S = S.reshape(S.shape[0], 1, S.shape[1], S.shape[2])
        else:
            raise ValueError('Gradient must be either a 2D or 4D tensor.')

        # Boolean concept mask
        y = y.float()

        # Reshape the concept mask to match sensitivity
        y = reshape_concept_mask(y, S)

        # Binary mask positive sensitivity
        S = S > 0

        # Mask according to the concept mask
        S = S * y

        # Count positives in S
        pos += S.sum().item()

        # Count positives in y
        n += y.sum().item()

    # Remove hooks
    hook_a.remove()
    hook_b.remove()

    # Check if there was at least one positive example
    if n == 0:
        raise ValueError('No instances of the target concept were found in '
                         'the dataset')

    return (pos / n) - 0.5


def train_CAV(concepts: List[Concept],
              activations: Union[Tuple, np.ndarray],
              dataset: torch.utils.data.Dataset,
              train: List[List[int]],
              val: List[List[int]],
              gpu: bool,
              epochs: int,
              batch_size: int,
              nw: int,
              verbose: bool = False,
              queue: Queue = None) -> Dict:

    # Eventually reload the memmap
    if isinstance(activations, tuple):
        activations = np.memmap(activations[0],
                                dtype=float,
                                mode='r',
                                shape=activations[1])

    result = []
    for concept, train_idx, val_idx in zip(concepts, train, val):
        # Split dataset
        dataset.target_concept = concept
        dataset.return_index = True
        dataset.skip_image = True

        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)

        # Train CAV
        cav = CAV(concept, train_set,
                  activations,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=verbose,
                  gpu=gpu, nw=0,
                  criterion=FocalLoss)

        # Evaluate CAV (Train)
        train_stat = eval_CAV(cav, concept, train_set,
                              activations,
                              batch_size=batch_size,
                              verbose=verbose,
                              gpu=gpu, nw=0)

        # Evaluate CAV (Val)
        val_stat = eval_CAV(cav, concept, val_set,
                            activations,
                            batch_size=batch_size,
                            verbose=verbose,
                            gpu=gpu, nw=0)

        # Create concept entry
        result.append({'train_set': train_idx,
                       'val_set': val_idx,
                       'train_stat': train_stat,
                       'val_stat': val_stat,
                       'cav': np.array(cav.weight.data.detach()
                                       .cpu().numpy())})

        # Eventually notify progress
        if queue is not None:
            queue.put(1)

        if verbose:
            print('== Train')
            print(result['train_stat'])
            print('== Val')
            print(result['val_stat'])

    return result
