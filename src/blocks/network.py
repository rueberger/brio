"""
This module holds the Network class
"""
import numpy as np
from blocks.aux import NetworkParams
from misc.plotting import ParamPlot

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
        self.t_counter = 0
        if params.display:
            self.param_plot = ParamPlot(self)

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
        stas = []
        activations = []
        stim = []
        for idx, stimulus in enumerate(stimulus_generator):
            if idx == num_stim_to_avg:
                break
            stim.append(stimulus)
            self.update_network(stimulus)
            activations.append(self.layers[layer_idx].state * idx)
        filt_activations = np.array(activations, dtype=np.int32).T
        stim = np.array(stim)
        for idx in xrange(self.layers[layer_idx].n_dims):
            activation_idx = np.where(filt_activations[idx] != 0)[0]
            sta_at_idx = np.mean(stim[activation_idx], axis=0)
            stas.append(sta_at_idx)
        return stas

    def describe_progress(self):
        """ prints some information on how the training is progressing

        :returns: None
        :rtype: None
        """
        print "Training iteration: {}".format(self.t_counter)
        print "Example firing rate: {}".format(self.layers[1].firing_rates[0])
        if self.params.display:
            self.param_plot.update_plot()

    def train(self, stimulus_generator):
        """ Trains the network on the generated stimulus
        Reports progress

        :param stimulus_generator: a generator object. calling next on this generator must return
          an array that can be flatted to the shape of the input layer
        :returns: None
        :rtype: None
        """
        for idx, stimulus in enumerate(stimulus_generator):
            self.run_network(stimulus)
            self.t_counter += 1
            if idx % self.params.stimuli_per_epoch == 0 and idx != 0:
                self.training_iteration()


    def update_network(self, stimulus):
        """ Present stimulus to the network and update the state

        :param stimulus: array of shape (input_layer.ndims, )
        """
        self.layers[0].set_state(stimulus)
        for layer in self.layers[1:]:
            layer.reset()
        if self.params.async:
            np.random.shuffle(self.node_idx)
            for _ in xrange(self.params.presentations):
                for idx in self.node_idx:
                    layer, unit_idx = self.idx_to_layer[idx]
                    layer.async_update(unit_idx)
                for layer in self.layers:
                    layer.update_firing_rate()
                    layer.update_history()
        else:
            for _ in xrange(self.params.presentations):
                for layer in self.layers[1:]:
                    layer.sync_update()
                for layer in self.layers:
                    layer.update_firing_rates()
                    layer.update_history()



    def training_iteration(self):
        """ Calls the training method in each layer and connection
        Connection training method updates weights
        layer training method update biases

        :returns: None
        :rtype: None
        """
        for connection in self.connections:
            connection.weight_update()
        for layer in self.layers[1:]:
            layer.bias_update()


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

    def __set_parentage(self):
        """ sets self as the parent network of all layers

        :returns: None
        :rtype: None
        """
        for layer in self.layers:
            layer.unpack_network_params(self)
        for connection in self.connections:
            connection.unpack_network_params(self)
