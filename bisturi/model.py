from typing import Tuple
import numpy as np
import re
import torch
import torchvision


# Types
LayerID = Tuple[str, int]
Direction = Tuple[LayerID, np.ndarray, np.ndarray]


def load_model(model_name: str, path: str = None, GPU: bool = False):
    """
    Retrieves a model from TorchVision
    or from a path.

    Parameters
    ----------
    model_name: str
        The name of the model to load.
    path: str
        The path to the model.
    GPU: bool
        Whether to use the GPU.

    Returns
    -------
    model: torch.nn.Module
        The model.
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


def get_names(model: torch.nn.Module):
    """
    Returns the list of names
    for the hookable modules
    in the given model.

    Parameters
    ----------
    model: torch.nn.Module
        The model.

    Returns
    -------
    names: List[str]
        The list of names.
    """
    return [e[0] for e in model.named_modules() if e[0]]


def get_module(model: torch.nn.Module, name: str):
    """
    Selects a module by name

    Parameters
    ----------
    model: torch.nn.Module
        The model.
    name: str
        The name of the module.

    Returns
    -------
    module: torch.nn.Module
        The module.
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
