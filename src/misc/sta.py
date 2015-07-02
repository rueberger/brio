"""
This module contains utilities for computing the receptive fields of trained networks
"""
import numpy as np

def scalar_sta(net, n_samples=1E4):
    """ computes responses for visualizing the receptive field of layers
    computes a list of response for each neuron in each layer


    :param net: a trained network. the first layer must be a RasterInputLayer
    :returns: a dictionary of responses
    :rtype: dict((layer_idx, unit_idx): array(responses))
    """
    assert type(net.layers[0]).__name__ == 'RasterInputLayer'
    lower_bnd = net.layers[0].lower_bnd
    upper_bnd = net.layers[0].upper_bnd

    active_layers = range(1, len(net.layers))
    responses = [[] for _ in active_layers]
    samples = np.random.uniform(lower_bnd, upper_bnd, n_samples)
    response_dict = {}

    for sample_idx, scalar in enumerate(samples):
        net.update_network(scalar)
        for l_idx in active_layers:
            responses[l_idx - 1].append(net.layers[l_idx].state * sample_idx)
    for l_idx in active_layers:
        layer_responses = np.array(responses[l_idx - 1]).T
        for idx in xrange(net.layers[l_idx].n_dims):
            active_at_sample_idx = np.where(layer_responses[idx] != 0)[0]
            response_dict[(l_idx, idx)] = np.array(samples[active_at_sample_idx])
    return response_dict
