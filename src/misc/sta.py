"""
This module contains utilities for computing the receptive fields of trained networks
"""
import numpy as np
import itertools

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

    for sample_idx, stimulus in enumerate(stimuli):
        net.update_network(stimulus)
        for l_idx in active_layers:
            responses[l_idx - 1].append(net.layers[l_idx].state * sample_idx)
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
        stimuli = np.array(list(
            itertools.takewhile(lambda i, x: i < n_samples), enumerate(stim_gen)))
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
    scale = 1
    if img_dim is None:
        side_length = np.sqrt(net.layers[0].n_dims)
        img_dim = (side_length, side_length)
        assert int(side_length) == side_length
    else:
        assert len(img_dim) == 2
        assert img_dim[0] * img_dim[1] == net.layers[0].n_dims

    if stim_gen is None:
        def gauss(x_arr, mean):
            var = np.random.uniform(*var_range)
            return scale * np.exp(- ((x_arr - mean) ** 2) / (2 * var))

        stimuli = np.zeros((n_samples, img_dim[0], img_dim[1]))
        x_idx = np.arange(img_dim[0])
        y_idx = np.arange(img_dim[1])
        for idx, (x_mean, y_mean) in enumerate(zip(np.random.uniform(-1, img_dim[0], n_samples),
                                                   np.random.uniform(-1, img_dim[1], n_samples))):
            stimuli[idx] = np.outer(gauss(x_idx, x_mean), gauss(y_idx, y_mean))
    else:
        # this sets stimuli to an array containing the first n_samples elements of stim_gen
        stimuli = np.array(list(
            itertools.takewhile(lambda i, x: i < n_samples), enumerate(stim_gen)))
        assert stimuli.shape[1] == img_dim[0] and stimuli.shape[2] == img_dim[1]
    return record_responses(net, stimuli)

def auto_sta(net, n_samples=1E4, stim_gen=None):
    """ calls either img_sta or scalar_sta depending on input layer type

    :param net: a trained network
    :param n_samples: the number of samples to base the sta off of
    :param stim_gen: generator of stimuli. by default stimuli are generated at random
       for the relevant domain
    :returns: a dictionary of spike triggered averages
    :rtype: dict((layer_idx, unit_idx): array(sta))

    """
    if type(net.layers[0]).__name__ == 'InputLayer':
        return img_sta(net, n_samples, stim_gen)
    elif type(net.layers[0]).__name__ == 'RasterInputLayer':
        return scalar_sta(net, n_samples, stim_gen)
    else:
        raise NotImplementedError(
            "STA method has not been specified for input layer type: {}".format(
                type(net.layers[0]).__name__))
