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
        layer.LIFLayer(layer_sizes[2], LayerType.inhibitory)
    ]
    connection.OjaConnection(layers[0], layers[1], lrate_multiplier=0.2)
    connection.CMConnection(layers[1], layers[2], weight_scheme='uniform', lrate_multiplier=.7)
    connection.CMConnection(layers[2], layers[2],weight_scheme='zero', lrate_multiplier=1.5)
    connection.CMConnection(layers[2], layers[1], weight_scheme='zero', lrate_multiplier=0.7)
    return network.Network(layers, params)
