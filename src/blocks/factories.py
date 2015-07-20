"""
Factories for networks
"""
from blocks import layer, connection, network
from blocks.aux import LayerType, NetworkParams

def rbm_factory(layer_sizes):
    """ Constructs a Restricted Boltzmann Machine

    :param layer_sizes: list of the size of each layer. Input layer size first element, output last
    :returns: the constructed Restricted Boltzmann Machine
    :rtype: Network
    """
    layers = []
    layers.append(layer.InputLayer(layer_sizes[0]))
    for lsize in layer_sizes[1:]:
        layers.append(layer.BoltzmannMachineLayer(lsize))
    for input_layer, output_layer in zip(layers[:-1], layers[1:]):
        connection.CMConnection(input_layer, output_layer)
    return network.Network(layers)

def mlp_factory(layer_sizes):
    """ Constructs a Multilayer Perceptron network

    :param layer_sizes: list of the size of each layer. Input layer size first element, output last
    :returns: the constructed MLP net
    :rtype: Network
    """
    layers = []
    layers.append(layer.InputLayer(layer_sizes[0]))
    for lsize in layer_sizes[1:]:
        layers.append(layer.PerceptronLayer(lsize))
    for input_layer, output_layer in zip(layers[:-1], layers[1:]):
        connection.CMConnection(input_layer, output_layer)
    return network.Network(layers)

def einet_factory(layer_sizes, params=NetworkParams()):
    """ Constructs EI-net, inspired from King and Deweese 2013 (with some differences)
    Use Boltzmann machine units for now

    :param layer_sizes: list of the layer sizes. Must be of length 3.
      format is [input_size, excitatory_size, inhibitory_size]
      in general the inhibitory layer should be much smaller than the excitatory layer
    :returns: the constructed EI-net
    :rtype: Network
    """
    assert len(layer_sizes) == 3
    layers = [
        layer.InputLayer(layer_sizes[0]),
        layer.LIFLayer(layer_sizes[1], LayerType.excitatory),
        layer.LIFLayer(layer_sizes[2], LayerType.inhibitory, allow_self_con=False)
    ]
    input_con = connection.OjaConnection(layers[0], layers[1], lrate_multiplier=0.1)
    input_con.weight_multiplier = 5
    connection.CMConnection(layers[1], layers[2], weight_scheme='uniform', lrate_multiplier=.7)
    connection.CMConnection(layers[2], layers[2],weight_scheme='zero', lrate_multiplier=1.5)
    connection.CMConnection(layers[2], layers[1], weight_scheme='uniform', lrate_multiplier=0.7)
    return network.Network(layers, params)

def sailnet_factory(layer_sizes, params=NetworkParams()):
    """ Constructs SAILnet as specified in Zylderberg 2011

    :param layer_sizes: list of the layer sizes. Must be of length 3.
      format is [input_size, excitatory_size, inhibitory_size]
      in general the inhibitory layer should be much smaller than the excitatory layer
    :returns: the constructed SAILnet
    :rtype: Network
    """

    assert len(layer_sizes) == 2
    layers = [
        layer.InputLayer(layer_sizes[0]),
        layer.LIFLayer(layer_sizes[1]),
    ]
    connection.OjaConnection(layers[0], layers[1], lrate_multiplier=0.2)
    connection.FoldiakConnection(layers[1], layers[1], weight_scheme='uniform', lrate_multiplier=.7)
    return network.Network(layers, params)

def perceptron_factory(layer_sizes, params=NetworkParams()):
    import numpy as np
    assert len(layer_sizes) == 2
    layers = [
        layer.InputLayer(layer_sizes[0]),
        layer.PerceptronLayer(layer_sizes[0], update_bias=False)
    ]
    layers[1].bias = np.zeros(layer_sizes[1])
    connection.OjaConnection(layers[0], layers[1])
    return network.Network(layers, params)

def gated_einet_factory(layer_sizes, n_input_stimuli, params=NetworkParams()):
    """ Constructs an EInet with multiplicatively gated input
    That is, the excitatory current receives current:
      I_in = W_1 x_1 cdot W_2 x_2 ... W_{n-1} x_{n-1} cdot W_n x_n
    where x_i is the ith input stimuli, W_i is the ith set of weights,
      n = n_input_stimuli and cdot indicates a dot product

    :param layer_sizes: list of the layer sizes. Must be of length 3.
      format is [input_size, excitatory_size, inhibitory_size]
      in general the inhibitory layer should be much smaller than the excitatory layer
    :param n_input_stimuli: the number of input stimuli that will be gated multiplicatively
    :param params: a NetworkParams object with global parameters for this network
    :returns: the gated EInet
    :rtype: Network
    """
    assert len(layer_sizes) == 3
    # might as well use standard EINet
    assert n_input_stimuli > 1
    layers = [
        layer.GatedInput(layer_sizes[1]),
        layer.LIFLayer(layer_sizes[1], LayerType.excitatory),
        layer.LIFLayer(layer_sizes[2], LayerType.inhibitory, allow_self_con=False),
    ]
    for child_layer in layers[0].children:
        layers.append(child_layer)
        input_con = connection.OjaConnection(child_layer, layers[0], lrate_multiplier=0.1)
        input_con.weight_multiplier = 5
    connection.ConstantConnection(layers[0], layers[1])
    connection.CMConnection(layers[1], layers[2], weight_scheme='uniform', lrate_multiplier=.7)
    connection.CMConnection(layers[2], layers[2],weight_scheme='zero', lrate_multiplier=1.5)
    connection.CMConnection(layers[2], layers[1], weight_scheme='uniform', lrate_multiplier=0.7)
    return network.Network(layers, params)