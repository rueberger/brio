"""
This module holds the Network class
"""
import numpy as np
from blocks.aux import NetworkParams

class Network(object):
    """
    Top level class for networks
    The Network class contains pointers to the Layer and Connection classes
      that make up a network in addition to methods for running the network
    """

    def __init__(self, layers, params=NetworkParams()):
        """ Initalize Network object. Only layers are specified upon initalization
        Connections should already be instantiated

        :param layers: list of Layer objects. InputLayer must be the
           first element, output layer must be last
        :param presentations: the number of times to run the
           network for each stimulus. For async networks
        """
        self.layers = layers
        self.params = params
        self.__check_layers()
        self.__find_connections()
        self.node_idx = np.arange(np.sum([l.n_dims for l in layers[1:]]))
        self.idx_to_layer = self.__build_layer_dict()
        self.__set_parentage()

    def compute_sta(self, stimulus_generator, layer_idx, num_stim_to_avg=250):
        """ Computes the spike triggered averages for the layers

        :param stimulus_generator: a generator object. calling next on this generator must return
          an array that can be flatted to the shape of the input layer
        :param layer: idx to layer to compute STA's for
        :param num_stim_to_avg:
        :returns: a list containing the spike triggered averages for each layer
          list contains one array for each layer, in the same order self.layers
          averages are of the format (input_n_dims, layer_n_dims)
        :rtype: [array]

        """
        # changed my emacs bindings recently... still getting used to them
        activations = []
        stim = []
        for idx, stimulus in enumerate(stimulus_generator):
            if idx == num_stim_to_avg:
                break
            stim.append(stimulus)
            self.update_network(stimulus)
            # what this does is append the idx of the current stimulus to activations
            #  if the neuron at that idx fired in response to this stimulus
            #  (using binary neurons in {0, 1})
            activations.append(self.layers[layer_idx] * idx)
        stas = []
        for activation_idx in activations:
            sta_at_idx = np.mean(stim[activation_idx], axis=0)
            stas.append(sta_at_idx)
        return stas

    def train(self, stimulus_generator):
        """ Trains the network on the generated stimulus
        Reports progress

        :param stimulus_generator: a generator object. calling next on this generator must return
          an array that can be flatted to the shape of the input layer
        :returns: None
        :rtype: None
        """
        for idx, stimulus in enumerate(stimulus_generator):
            if idx % 25 == 0 and idx > 100:
                print "Now training on stimulus number {}".format(idx)
                print "Example firing rate: {}".format(self.layers[1].firing_rates()[0])
            self.run_network(stimulus)



    def run_network(self, stimulus):
        """ Presents the stimulus to the network
        Updates the state and performs a training iteration
        This is the method to call from external code
        training does not begin until the network has sufficient history to compute
          mean firing rates

        :param stimulus: array of shape (input_layer.ndims, )
        :returns: None
        :rtype: None
        """
        self.update_network(stimulus)
        # look to see if the layers have accumulated sufficient history
        if len(self.layers[0].history) >= self.params.layer_history_length:
            self.training_iteration()

    def update_network(self, stimulus):
        """ Present stimulus to the network and update the state

        :param stimulus: array of shape (input_layer.ndims, )
        """
        np.random.shuffle(self.node_idx)
        self.layers[0].set_state(stimulus)
        for _ in xrange(self.params.presentations):
            for idx in self.node_idx:
                layer, unit_idx = self.idx_to_layer[idx]
                layer.update_state(unit_idx)
        self.__update_layer_histories()

    def training_iteration(self):
        """ Calls the training method in each layer and connection
        Connection training method updates weights
        layer training method update biases

        :returns: None
        :rtype: None
        """
        for connection in self.connections:
            connection.apply_weight_rule()
        for layer in self.layers[1:]:
            layer.apply_bias_rule()


    def __check_layers(self):
        """ Checks that the input layer is the first element of layers
          and that all other layers have inputs and outputs (except for possibly the output layer)
        """
        # also check type of input layer
        assert len(self.layers[0].inputs) == 0
        assert len(self.layers[0].outputs) != 0
        for layer in self.layers[1:-1]:
            assert len(layer.inputs) != 0
            assert len(layer.outputs) != 0
        assert len(self.layers[-1].inputs) != 0

    def __build_layer_dict(self):
        """ Builds a dictionary from unit idx to layer for use in update method

        :returns: dictionary: (idx : layer object)
        :rtype: dictionary
        """
        unit_dict = {}
        start_idx = 0
        for layer in self.layers[1:]:
            for idx in xrange(layer.n_dims):
                unit_dict[idx + start_idx] = (layer, idx)
            start_idx = len(unit_dict)
        return unit_dict

    def __find_connections(self):
        """ Finds all the connections in the network by searching through the
          input and output lists in each layer

        :returns: None
        :rtype: None
        """
        self.connections = set()
        for layer in self.layers:
            for connection in layer.inputs + layer.outputs:
                self.connections.add(connection)

    def __update_layer_histories(self):
        """ calls the update history method in each layer
        intended to be called once during each network update
          after all units have been updated

        :returns: None
        :rtype: None
        """
        for layer in self.layers:
            layer.update_history()

    def __set_parentage(self):
        """ sets self as the parent network of all layers

        :returns: None
        :rtype: None
        """
        for layer in self.layers:
            layer.unpack_network_params(self)
        for connection in self.connections:
            connection.unpack_network_params(self)




    # burn in method