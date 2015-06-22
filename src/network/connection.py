
import numpy as np

class Connection(object):
    """
    Base class for connection between network layers
    holds network weights
    """

    def __init__(self, input_layer, output_layer):
        # to do: add excitatory flag
        self.input_layer = input_layer
        self.output_layer = output_layer
        # allow for specification of input method
        self.weights = np.random.randn(input_layer.ndims, output_layer.ndims)
        self.output_layer.add_input(self)


    def get_energy(self, idx):
        """
        Returns the output into the unit at idx of the output layer
        """
        return self.weights[:, idx] * self.input_layer.state




# subclass: constrainedconnection