from tqdm.auto import tqdm
from bisturi.semalign import moduleid_to_string
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os


def activation_by_example(activations, top_n=10, style='max',
                          cache='', module_name=None, verbose=False):
    """
    Retrieve for each unit the indexes
    of the images that maximized its
    activations
    """

    if isinstance(activations, dict) and not module_name:
        if verbose:
            iterator = tqdm(activations)
        else:
            iterator = activations

        return {m: activation_by_example(activations, top_n, style, cache, m)
                for m in iterator}

    # Best or worst examples
    if style == 'max':
        f = np.max
        sign = 1
    elif style == 'min':
        f = np.min
        sign = -1
    else:
        raise NotImplementedError

    # Persistency
    if cache and module_name:
        examples_path = os.path.join(cache,
                                     style+'examples_%s.npy' %
                                     moduleid_to_string(module_name))
        try:
            return np.load(examples_path)
        except FileNotFoundError:
            pass

    # Select activations
    activations = activations[module_name]
    n_images = activations.shape[0]

    while len(activations.shape) != 2:
        activations = f(activations, axis=-1)

    activations_idx = np.argpartition(sign * activations.T,
                                      n_images-top_n, axis=1)[:, -top_n:]
    # Store the results
    if cache and module_name:
        np.save(examples_path, activations_idx)

    return activations_idx


def visualize_examples(images_idx, annotations, unit, figsize=(20, 10),
                       ncols=5, module_name=None):
    """
    Visualization of the provided
    examples by using matplotlib
    """

    if isinstance(images_idx, dict) and not module_name:
        for module_name in images_idx.keys():
            visualize_examples(images_idx, annotations, unit, figsize,
                               ncols, module_name)
        return

    examples = images_idx[module_name][unit]
    n_examples = len(examples)

    nrows = int(np.ceil(n_examples / ncols))

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    i = 0
    for row in ax:
        for col in row:
            if i < n_examples:
                e = examples[i]
                path = annotations[e]['path']
                img = mpimg.imread(path)
                col.imshow(img)
                i += 1

    plt.show()
