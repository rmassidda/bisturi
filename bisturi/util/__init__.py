import numpy as np
import torch
from nltk.corpus.reader.wordnet import Synset


def synset_to_wnid(synset: Synset):
    """
    Convert a wordnet synset to a WordNet ID.

    Parameters
    ----------
    synset : Synset
        The wordnet synset.

    Returns
    -------
    wnid : str
        The WordNet ID.
    """
    return synset.pos() + str(synset.offset()).zfill(8)


def top_k_accuracy(y_target: torch.Tensor, y_pred: torch.Tensor, k: int = 5):
    """
    Computes top-k accuracy between
    a target vector and a predicted
    one.
    """
    # Select top-k classes
    _, top_k_pred = y_pred.topk(k, dim=1)
    # Replicate the target labels k times
    y_target_k = y_target.unsqueeze(1).repeat(1, k)
    # Compare the top-k predictions with the target labels
    correct = top_k_pred == y_target_k
    # Compute the top-k accuracy
    top_k_acc = correct.sum(dim=1).float().mean()
    return top_k_acc


def is_convolutional(batch: torch.Tensor):
    return len(batch.shape) == 4


def reshape_concept_mask(y: torch.Tensor, a: torch.Tensor):
    """
    Reshape the concept mask to the shape of the
    input activation tensor.

    If a tensor comes from a convolutional layer,
    it has shape (batch_size, channels, height, width).
    Otherwise, it has shape (batch_size, channels).
    In a concept mask the channels stand for different
    concepts, in an activation tensor for each neuron.

    Different strategies for reshaping the concept mask
    are implemented here according to the shape of the
    original tensors.

    Parameters
    ----------
    y : torch.Tensor
        The concept mask.
    a : torch.Tensor
        The input activation tensor.
    """

    if is_convolutional(y) and not is_convolutional(a):
        # Pixel-level annotations && fully connected layer
        # NOTE: we implicitly assume that there are not distinct
        #       branches in the network, thus that the fully
        #       connected layer has a complete receptive field.
        y = torch.amax(y, dim=(2, 3))
    elif is_convolutional(y) and is_convolutional(a):
        # Pixel-level annotations && convolutional layer
        # TODO: don't scale masks but use max pooling
        #       on the receptive field of the module.
        y = torch.nn.functional.interpolate(y, size=a.shape[2:],
                                            mode='nearest')
    elif not is_convolutional(y) and not is_convolutional(a):
        # Image-level annotations && fully connected layer
        # NOTE: we implicitly assume that there are not distinct
        #       branches in the network, thus that the fully
        #       connected layer has a complete receptive field.
        pass
    elif not is_convolutional(y) and is_convolutional(a):
        # Image-level annotations && convolutional layer
        # NOTE: we assume that the visual concept is spread
        #       over the entire image.
        y = y.reshape(*y.shape, 1, 1)
        y = torch.repeat_interleave(y, a.shape[2], dim=2)
        y = torch.repeat_interleave(y, a.shape[3], dim=3)
    else:
        raise ValueError('Activation tensor and concept mask wrongly shaped.')

    return y


def filter_nan(x):
    return x[~np.isnan(x)]


def sigmoid(x, k=2.0):
    """
    Sigmoidal function.
    """
    return 1 / (1 + ((1/x) - 1)**k)
