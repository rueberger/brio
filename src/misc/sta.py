"""
This module contains utilities for computing the receptive fields of trained networks
"""
import numpy as np
import itertools
from misc.utils import roll_itr

def record_responses(net, stimuli):
    """ present stimuli to net and record which units in which layers respond to what

    :param net: a trained network
    :param stimuli: iterable of stimuli
    :returns: a dictionary recording the responses of each unit
    :rtype: dict((layer_idx, unit_idx) : array(responses))
    """
    # can run in parallel
    active_layers = range(1, len(net.layers))
    responses = [[] for _ in active_layers]
    response_dict = {}
    epoch_size = net.params.stimuli_per_epoch

    for epoch_idx, rolled_stimuli in enumerate(roll_itr(stimuli, epoch_size)):
        net.update_network(rolled_stimuli)
        sample_idx = np.arange(epoch_idx, epoch_idx + epoch_size).reshape(1, -1)
        for l_idx in active_layers:
            for response in (net.layers[l_idx].state * sample_idx).T:
                responses[l_idx - 1].append(response)
    for l_idx in active_layers:
        layer_responses = np.array(responses[l_idx - 1]).T
        for idx in xrange(net.layers[l_idx].n_dims):
            active_at_sample_idx = np.where(layer_responses[idx] != 0)[0]
            response_dict[(l_idx, idx)] = np.array(stimuli[active_at_sample_idx])
    return response_dict

def scalar_sta(net, n_samples=1E4, stim_gen=None):
    """ computes responses for visualizing the receptive field of layers
    computes a list of response for each neuron in each layer


    :param net: a trained network. the first layer must be a RasterInputLayer
    :param stim_gen: generator of stimuli. by default stimuli are generated at random
       for the relevant domain
    :returns: a dictionary of responses
    :rtype: dict((layer_idx, unit_idx): array(responses))
    """
    assert type(net.layers[0]).__name__ == 'RasterInputLayer'
    if stim_gen is None:
        stimuli = np.random.uniform(net.layers[0].lower_bnd, net.layers[0].upper_bnd, n_samples)
    else:
        # this sets stimuli to an array containing the first n_samples elements of stim_gen
        stimuli = np.array(list(itertools.islice(stim_gen, n_samples)))
        assert stimuli.ndim == 1
    return record_responses(net, stimuli)

def img_sta(net, n_samples=1E4, img_dim=None, stim_gen=None):
    """ computes spike triggered averages for visualizing the receptive field of layers

    :param net: a trained network.
    :param n_samples: the number of samples to draw
    :param img_dim: tuple with image dimensions. if None the size of the InputLayer must
      be a perfect square
    :param stim_gen: generator of stimuli. by default stimuli are generated at random
       for the relevant domain
    :returns: a dictionary of spike triggered averages
    :rtype: dict((layer_idx, unit_idx): array(sta))
    """
    var_range = (.5, 1.5)
    img_dim = factor(net.layers[0].n_dims)

    if stim_gen is None:
        # i think this should be white noise
        stimuli = np.zeros((n_samples, img_dim[0], img_dim[1]))
        x_idx = np.arange(img_dim[0])
        y_idx = np.arange(img_dim[1])
        for idx, (x_mean, y_mean) in enumerate(zip(np.random.uniform(-1, img_dim[0], n_samples),
                                                   np.random.uniform(-1, img_dim[1], n_samples))):
            stimuli[idx] = np.outer(gaussian_blob(x_idx, x_mean, var_range),
                                    gaussian_blob(y_idx, y_mean, var_range))
    else:
        # this sets stimuli to an array containing the first n_samples elements of stim_gen
        stimuli = np.array(list(itertools.islice(stim_gen, n_samples)))
        assert stimuli.shape[1] == img_dim[0] and stimuli.shape[2] == img_dim[1]
    return record_responses(net, stimuli)

def split_img_sta(net, n_samples=1E4, stim_gen=None):
    """ computes spike triggered averages for visualizing the receptive field of layers
    differs from img_sta in that this supports multiple image input for
      split or input layers

    :param net: a trained network.
    :param n_samples: the number of samples to draw
    :param stim_gen: generator of stimuli. by default stimuli are generated at random
       for the relevant domain
    :returns: a dictionary of spike triggered averages
    :rtype: dict((layer_idx, unit_idx): array(sta))
    """
    # can also just take the product with the first layer weight
    input_layer = net.layers[0]
    n_stim_dims = input_layer.children[0].n_dims
    n_children = len(input_layer.children)
    # all child layers must have the same dimension
    assert (np.array([c.n_dims for c in input_layer.children]) == n_stim_dims).all()
    img_dim = factor(n_stim_dims)

    if stim_gen is None:
        # white noise is the most general but wont' work well for disparity
        # i don't think it matters what axis the child images are tiled across
        stimuli = np.random.random(n_samples, img_dim[0] * n_children, img_dim[1])
    else:
        # this sets stimuli to an array containing the first n_samples elements of stim_gen
        stimuli = np.array(list(itertools.islice(stim_gen, n_samples)))
        assert stimuli.shape[1] == img_dim[0] * n_children and stimuli.shape[2] == img_dim[1]
    return record_responses(net, stimuli)


def auto_sta(net, n_samples=1E4, stim_gen=None):
    """ calls either img_sta, split_img_sta or scalar_sta depending on input layer type

    :param net: a trained network
    :param n_samples: the number of samples to base the sta off of
    :param stim_gen: generator of stimuli. by default stimuli are generated at random
       for the relevant domain
    :returns: a dictionary of spike triggered averages
    :rtype: dict((layer_idx, unit_idx): array(sta))

    """
    from blocks.layer import InputLayer, RasterInputLayer, GatedInput, SplitInput
    input_layer = net.layers[0]
    if isinstance(input_layer, (SplitInput, GatedInput)):
        split_img_sta(net, n_samples, stim_gen=stim_gen)
    elif isinstance(input_layer, RasterInputLayer):
        return scalar_sta(net, n_samples, stim_gen=stim_gen)
    elif isinstance(input_layer, InputLayer):
        return img_sta(net, n_samples, stim_gen=stim_gen)
    else:
        raise NotImplementedError(
            "STA method has not been specified for input layer type: {}".format(
                type(net.layers[0]).__name__))


def gaussian_blob(x_arr, mean, var_range):
    """ Utility method. returns a gaussian blob centered on mean
    Variance is drawn from var_range

    :param x_arr: 2 dimensional array to fill
    :param mean: 1d array of shape (2, ), probably
    :param var_range: tuple (min_var, max_var)
    :returns: array filled with the blob
    :rtype: array of shape x_arr.shape

    """
    var = np.random.uniform(*var_range)
    return np.exp(- ((x_arr - mean) ** 2) / (2 * var))

def factor(r):
    """ factor r as evenly as possible

    :param r: positive integer
    :returns: the two largest factors
    :rtype: tuple (p, q)
    """
    # prevent pylint from complaining about p,q being bad
    # pylint:disable=c0103
    assert int(r) == r
    # upper bound since int rounds down
    q_max = int(np.sqrt(r) + 1)
    for q in xrange(q_max, 0, -1):
        if r % q == 0:
            return (q, r / q)
