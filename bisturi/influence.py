from bisturi.model import get_module
from bisturi.model import ModuleID, Direction
from bisturi.util import top_k_accuracy
from functools import partial
from tqdm.auto import tqdm
from typing import Callable, Dict, List, Tuple
import torch


def contribution(model: torch.nn.Module,
                 dir_a: Direction,
                 dir_b: Direction,
                 input_shape: torch.Size = (1, 3, 224, 224),
                 basis: torch.Tensor = None,
                 magnitude: float = 1.0,
                 gpu: bool = False) -> float:
    """
    Estimates the contribution of the feature indicated
    by the first direction to the feature pointed by
    the second direction.
    """

    # Decompose direction
    module_a_id, dir_a = dir_a
    module_b_id, dir_b = dir_b

    # Retrieve modules
    module_a = get_module(model, module_a_id[0])
    module_b = get_module(model, module_b_id[0])

    # Forward hooks
    counter = 0

    def fun_a_hook(module, input, output):
        # Check if correct call of the module
        if counter == module_a_id[1]:
            out = torch.zeros(output.shape)
            # TODO: handle different basis
            # Check if convolutional
            if len(output.shape) == 4:
                out[:, dir_a, :, :] = magnitude
            else:
                out[:, dir_a] = magnitude
            return out

    result = []

    def fun_b_hook(module, input, output):
        out = output.detach().cpu().numpy()
        # TODO: handle different basis
        # Check if convolutional
        if len(output.shape) == 4:
            result.append(out[:, dir_a, :, :])
        else:
            result.append(out[:, dir_a])

    # Attach hooks
    hook_a = module_a.register_forward_hook(fun_a_hook)
    hook_b = module_b.register_forward_hook(fun_b_hook)

    # Keep track of the model status
    was_training = model.training
    model.eval()

    # Prepare input batch
    batch = torch.zeros(input_shape)
    if gpu:
        batch = batch.cuda()

    # Forward pass of the input
    with torch.no_grad():
        _ = model.forward(batch)

    # Revert model status
    if was_training:
        model.train()

    # Select index if module has been reused
    result = result[module_b_id[1]]

    # Remove hooks
    hook_a.remove()
    hook_b.remove()

    return result


def forward_model(model: torch.nn.Module,
                  dataset: torch.utils.data.Dataset,
                  nw: int = 8,
                  batch_size: int = 32,
                  gpu: bool = False,
                  verbose: bool = False,
                  pre_callback: Callable = None,
                  post_callback: Callable = None,
                  top_k: int = 5) \
                      -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a model and a datset it returns
    the ground truth and the output from
    the model.
    """
    # Evaluate model
    model.eval()

    # Iterate over the dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         num_workers=nw)

    # Progress bar
    if verbose:
        loader = tqdm(loader)

    # Output tensors
    y_gt = None
    y = None

    for batch in loader:
        # Separate inputs and targets
        x, y_gt_batch = batch

        # Eventually move to GPU
        if gpu:
            x = x.cuda()

        # Callback prior forward step
        if pre_callback:
            pre_callback()

        # Forward pass
        y_batch = model.forward(x)
        y_batch = y_batch.detach().cpu()

        # Callback after forward step
        if post_callback:
            post_callback()

        # Accumulate results (Ground truth)
        if y_gt is None:
            y_gt = y_gt_batch
        else:
            y_gt = torch.cat([y_gt, y_gt_batch])

        # Accumulate results (Output model)
        if y is None:
            y = y_batch
        else:
            y = torch.cat([y, y_batch])

        # Print overall classification accuracy
        if verbose:
            tmp_accuracy = top_k_accuracy(y_gt, y, k=top_k)
            loader.set_description(f'Accuracy: {tmp_accuracy:.4f}')

    return (y_gt, y)


def accuracy_drop(model: torch.nn.Module,
                  dataset: torch.utils.data.Dataset,
                  dirs: Dict[ModuleID, List[int]],
                  basis: Dict[ModuleID, torch.Tensor] = None,
                  mu: float = 0.0,
                  batch_size: int = 32,
                  nw: int = 8,
                  cache: str = None,
                  gpu: bool = False,
                  verbose: bool = False) \
        -> List[float]:
    """
    Estimates the accuracy drop as described in "Revisiting the
    Importance of Individual Units in CNNs via Ablation"
    by Zhou et al. (2018).

    In addition to the original paper, this implementation
    enables the estimate of the accuracy drop between any
    arbitrary directions in the neural model.

    The returned value contains the accuracy drop
    for each target class or concept in the
    provided direction.
    """

    # Eventually load original model forward
    compute_original = False
    if cache is not None:
        try:
            y_gt, y = torch.load(cache)
        except FileNotFoundError:
            compute_original = True
    else:
        compute_original = True

    # Eventually forward original model
    if compute_original:
        y_gt, y = forward_model(model, dataset, nw=nw,
                                batch_size=batch_size,
                                gpu=gpu, verbose=verbose)
        torch.save((y_gt, y), cache)

    # Forward counter per module
    unique_names = [module_name for module_name, _ in dirs]
    counter = {name: 0 for name in unique_names}

    # Define ablation forward hook
    def forward_hook(module, input, output, module_name=None):
        # Compose module id
        module_iter = counter[module_name]
        module_id = (module_name, module_iter)

        # Check if the module has to be altered
        if module_id in dirs:
            if basis is not None and module_id in basis:
                # TODO: handle different basis
                raise NotImplementedError

            # Zero out the directions
            mask = torch.ones_like(output)
            mask[:, dirs[module_id]] = 0.0
            output = output * mask

            # Eventualy replace the directions with constant mu
            if mu != 0.0:
                mask = torch.zeros_like(output)
                mask[:, dirs[module_id]] = mu
                output = output + mask

        # Update counter
        counter[module_name] += 1

        return output

    def pre_callback():
        # Zero the counter
        for name in counter:
            counter[name] = 0

    # Attach forward hooks
    modules = [get_module(model, module_name) for module_name in unique_names]
    hooks = [module.register_forward_hook(partial(forward_hook,
             module_name=module_name)) for module_name, module
             in zip(unique_names, modules)]

    # Accuracy for the ablated model
    y_gt_a, y_a = forward_model(model, dataset, nw=nw, batch_size=batch_size,
                                gpu=gpu, verbose=verbose,
                                pre_callback=pre_callback)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return y_gt, y, y_gt_a, y_a
