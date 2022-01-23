from functools import partial
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Union
import re
import torch
import torchvision


# Types
ModuleID = Union[str, Tuple[str, int]]
Direction = Tuple[ModuleID, int]


class LinearModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(input_size, output_size)])

    def forward(self, x):
        for module in self.linears:
            x = module(x)
        return x


def load_model(model_name, path=None, GPU=False):
    """
    Retrieves a model either from
    disk or by using torchvision.
    """
    # Retrieve the model from TorchVision
    if path is None:
        model = torchvision.models.__dict__[model_name](pretrained=True)
    else:
        # Load local model
        if GPU:
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))

        # Check if state_dict was serialized
        if isinstance(checkpoint, dict):
            # Retrieve state_dict
            if 'state_dict' in checkpoint:
                state_dict = {str.replace(
                    k, 'module.', ''): v for k, v
                    in checkpoint['state_dict'].items()}
            else:
                state_dict = checkpoint

            # Get last layer
            last_layer = list(state_dict)[-1]
            num_classes = state_dict[last_layer].size()[0]

            # Retrieve architecture
            model = torchvision.models.__dict__[
                model_name](num_classes=num_classes)

            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                # FIXME: DenseNet-161 keys might differ
                if model_name == 'densenet161':
                    pattern = re.compile(
                      r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.'
                      r'(?:weight|bias|running_mean|running_var))$')
                    for key in list(state_dict.keys()):
                        res = pattern.match(key)
                        if res:
                            new_key = res.group(1) + res.group(2)
                            state_dict[new_key] = state_dict[key]
                            del state_dict[key]
                    model.load_state_dict(state_dict)
                else:
                    raise RuntimeError(e)

        else:
            model = checkpoint

    # Finalize model loading
    if GPU:
        model.cuda()

    return model


def get_names(model):
    """
    Returns the list of names
    for the hookable modules
    in the given model.
    """
    return [e[0] for e in model.named_modules() if e[0]]


def module_to_name(module_info):
    """
    Converts the information about a module produced
    by a torchinfo summary into a string name.
    """
    if module_info.depth == 0:
        return ''

    name = module_info.get_layer_name(show_var_name=True, show_depth=False)
    tokens = name.split('(')
    name = tokens[-1].split(')')[0]
    if not module_info.parent_info or module_info.depth == 1:
        return name
    else:
        return module_to_name(module_info.parent_info) + '.' + name


def get_module(model, name):
    """
    Selects a module by name
    """
    tokens = [e for e in name.split('.') if e]
    module = model
    for token in tokens:
        if isinstance(module, torch.nn.Sequential):
            try:
                idx = int(token)
            except ValueError:
                idx = list(dict(module.named_children()).keys()).index(token)
            module = module[idx]
        else:
            module = module._modules[token]
    return module


def accuracy(y_ground_truth, y_predicted, k=None):
    """
    Computes the accuracy of the predictions
    for the given ground truth.
    """
    if k is None:
        idx = list(range(0, len(y_ground_truth)))
    else:
        idx = y_ground_truth == k
        idx = idx.nonzero()

    return (y_predicted[idx] == y_ground_truth[idx]).float().mean()


def accuracy_drop(y_ground_truth, y_a, y_b, max_k=None):
    if max_k is None:
        max_k = y_ground_truth.max() + 1

    acc_a = [accuracy(y_ground_truth, acc_a, k)
             for k in range(1, max_k)]
    acc_b = [accuracy(y_ground_truth, acc_b, k)
             for k in range(1, max_k)]
    drop = [acc_a[i] - acc_b[i] for i in range(0, len(acc_a))]

    return acc_a, acc_b, drop


def AblateForward(model: torch.nn.Module,
                  dataset: torch.utils.data.Dataset,
                  dirs: Dict[ModuleID, List[int]],
                  basis: Dict[ModuleID, torch.tensor] = None,
                  mu: float = 0.0,
                  batch_size: int = 32,
                  gpu: bool = False,
                  verbose: bool = False) \
                    -> Tuple[torch.tensor, torch.tensor]:
    """
    Given a model, it produces the output over
    a dataset by ablating a set of directions.
    """

    # Evaluate model
    model.eval()

    # Canonical basis
    if basis is None:
        basis = {module_id: None for module_id in dirs}

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
            # TODO: handle convolutional layers

            # TODO: use canonical basis
            if basis[module_id] is None:
                basis[module_id] = torch.identity(output)

            # Change basis
            output = output @ basis[module_id]

            # Zero out the directions
            mask = torch.ones_like(output)
            mask[:, dirs[module_id]] = 0
            output = output * mask

            # Replace the directions with constant mu
            mask = torch.zeros_like(output)
            mask[:, dirs[module_id]] = mu
            output = output + mask

        # Update counter
        counter[module_name] += 1

        return output

    # Attach forward hooks
    modules = [get_module(model, module_name) for module_name in unique_names]
    hooks = [module.register_forward_hook(partial(forward_hook,
             module_name=module_name)) for module_name, module
             in zip(unique_names, modules)]

    # Iterate over the dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Progress bar
    if verbose:
        loader = tqdm(loader)

    # Accumulate output
    y_tot = torch.zeros(len(dataset), dtype=torch.float)

    for batch_idx, batch in enumerate(loader):
        # Separate inputs and targets
        x, y = batch

        # Eventually move to GPU
        if gpu:
            x = x.cuda()

        # Zero the counter
        for name in counter:
            counter[name] = 0

        # Forward pass
        y_bar = model.forward(x)

        # Accumulate output
        y_tot[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y_bar

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Return output
    return y_tot
