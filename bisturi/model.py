from bisturi.dataset import Dataset, collate_masks
from functools import partial
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import numpy as np
import os
import re
import torch
import torchvision


# Types
LayerID = Tuple[str, int]
Direction = Tuple[LayerID, np.ndarray, np.ndarray]


def layerid_to_string(layer_id: LayerID) -> str:
    """
    Converts a layer ID to a string.

    Parameters
    ----------
    layer_id: LayerID
        The layer ID.

    Returns
    -------
    string: str
        The string.
    """
    if isinstance(layer_id, tuple):
        return '%s#%d' % layer_id
    elif isinstance(layer_id, str):
        return layer_id
    else:
        raise TypeError('module_id must be a LayerID or str')


def is_convolutional(batch):
    '''
    Checks if the batch has been produced
    by a convolutional layer
    '''
    return len(batch.shape) == 4


def load_model(model_name: str, path: str = None,
               GPU: bool = False) -> torch.nn.Module:
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


def get_names(model: torch.nn.Module) -> List[str]:
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


def get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
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


def record_activations(model: torch.nn.Module,
                       modules_ids: List[LayerID],
                       dataset: Dataset,
                       batch_size: int = 128,
                       cache: str = None,
                       verbose: bool = True) -> Dict[LayerID, np.ndarray]:
    """
    Parameters
    ----------
    model: torch.nn.Module
        PyTorch model to analyze
    modules_id: List[ModuleID]
        Coordinates of the modules to analyze
    dataset: torch.utils.data.Dataset
        Dataset containing the inputs
    batch_size: int, optional
        Batch size for the forward pass
    cache: str, optional
        Path of the folder in which to
        eventually store the activations
    gpu: bool, optional
        Flag to handle GPU usage
    verbose: bool, optional
        When False disables the progress bar

    Returns
    -------
    activations: Dict[ModuleID, np.ndarray]
        Dictionary mapping the module
        id to either a NumPy array
        or a memmap containing the
        activations per input and
        per unit
    """

    activations = {}
    act_size = {}

    # check wheter model is on gpu
    gpu = next(model.parameters()).is_cuda

    # normalize module ids
    modules_ids = [(m, 0) if isinstance(m, str) else m for m in modules_ids]

    # module ids to string
    modules_str = [layerid_to_string(m) for m in modules_ids]

    # eventually load from file
    if cache:
        # skip network forward pass
        skip = True

        # shape filenames
        shape_filenames = {m_id: os.path.join(cache, "size_%s.npy" % m_str)
                           for m_id, m_str in zip(modules_ids, modules_str)}

        # activations filenames
        act_filenames = {m_id: os.path.join(cache, "act_%s.mmap" % m_str)
                         for m_id, m_str in zip(modules_ids, modules_str)}

        # load from file
        for m_id in modules_ids:
            s_fn = shape_filenames[m_id]
            a_fn = act_filenames[m_id]
            if os.path.exists(s_fn) and os.path.exists(a_fn):
                act_size[m_id] = np.load(s_fn)
                activations[m_id] = np.memmap(a_fn,
                                              dtype=float,
                                              mode='r',
                                              shape=tuple(act_size[m_id]))
            else:
                if verbose:
                    print("Activations for %s not found" % m_id[0])
                    print("Expected path was %s" % s_fn)
                skip = False

        # All the activations are on disk
        if skip:
            return activations

    # disable concept masks retrieval
    was_skipping_masks = dataset.skip_masks
    dataset.skip_masks = True

    # fix batch size for the image loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         collate_fn=collate_masks)

    # retrieve modules from the model
    modules_names = list({m_name for m_name, m_idx in modules_ids})
    modules = [get_module(model, m_name) for m_name in modules_names]

    # init recording variables
    recording = {m_name: [] for m_name, m_idx in modules_ids}

    # Define generic hook
    def hook_feature(module, input, output, module_name):
        recording[module_name].append(output.detach().cpu().numpy())

    # Register hooks
    hooks = [partial(hook_feature, module_name=m_name)
             for m_name in modules_names]
    hooks = [module.register_forward_hook(h)
             for module, h in zip(modules, hooks)]

    # keep track of the model status
    was_training = model.training
    model.eval()

    # batch iteration over the inputs
    first_batch = True
    for batch_idx, batch in enumerate(tqdm(loader, total=len(loader),
                                           disable=not verbose)):

        # Ignore masks and indices
        _, batch, _ = batch

        # Delete previous recording
        keys = list(recording.keys())
        for key in keys:
            del recording[key][:]

        # Prepare input batch
        if gpu:
            batch = batch.cuda()

        # Forward pass of the input
        with torch.no_grad():
            _ = model.forward(batch)

        # initialize tensors
        if first_batch:
            for m_id in modules_ids:
                m_name, m_idx = m_id
                act_size[m_id] = (len(dataset),
                                  *recording[m_name][m_idx].shape[1:])
                if cache:
                    s_fn = shape_filenames[m_id]
                    a_fn = act_filenames[m_id]
                    np.save(s_fn, act_size[m_id])
                    activations[m_id] = np.memmap(a_fn,
                                                  dtype=float,
                                                  mode='w+',
                                                  shape=act_size[m_id])
                else:
                    activations[m_id] = np.zeros(act_size[m_id])

            # Do not repeat the initialization
            first_batch = False

        # copy activations
        start_idx = batch_idx*loader.batch_size
        end_idx = min((batch_idx+1)*loader.batch_size, len(dataset))
        for m_id in modules_ids:
            m_name, m_idx = m_id
            activations[m_id][start_idx:end_idx] = recording[m_name][m_idx]

    # revert model status
    if was_training:
        model.train()

    # revert dataset preference
    dataset.skip_masks = was_skipping_masks

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations
