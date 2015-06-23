A
class Network(object):
    """
    Top level class for networks
    The Network class contains pointers to the Layer and Connection classes
      that make up a network in addition to methods for running the network
    """

    def __init__(layers, presentations=5):
        """ Initalize Network object. Only layers are specified upon initalization
        Connections should already be instantiated

        :param layers: list of Layer objects. InputLayer must be the first element, output layer must be last
        :param presentations: the number of times to run the network for each stimulus. For async networks
        """
        self.input_layer = input_layer
        self.layers = layers
        self.__check_layers()
        self.presentations = 5
        self.node_idx = np.arange(np.sum[l.n_dims for l in layers])
        self.idx_to_layer = self.__build_layer_dict()


    def __check_layers():
        """ Checks that the input layer is the first element of layers
          and that all other layers have inputs and outputs (except for possibly the output layer)
        """
        assert len(self.layers[0].inputs) == 0
        assert len(self.layers[0].outputs) != 0
        for layer in self.layers[1:-1]:
            assert len(layer.inputs) != 0
            assert len(layer.outputs) != 0
        assert len(self.layers[-1].inputs) != 0

    def __build_layer_dict():
        """ Builds a dictionary from unit idx to layer for use in update method

        :returns: dictionary: (idx : layer object)
        :rtype: dictionary
        """
        unit_dict = {}
        start_idx = 0
        for layer in layers:
            for idx in xrange(layer.n_dims):
                unit_dict[idx + start_idx] = layer
            start_idx = len(unit_dict)
        return unit_dict


    def update_network(self, stimulus):
        """ Present stimulus to the network and update the state

        :param stimulus: array of shape (input_layer.ndims, )
        """
        np.random.shuffle(self.node_idx)
        self.input_layer.set_state(stimulus)
        for _ in xrange(self.presentations):
            for idx in node_idx:
                self.idx_to_layer[idx].update(idx)