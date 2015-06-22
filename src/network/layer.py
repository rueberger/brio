
import numpy as np

class Layer(object):
    """
    Base class for a network layer
    Holds the state, binary only for now
    """

    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.state = np.ones(n_dims)
        # to do allow for specification of init method
        self.bias = np.random.randn(n_dims)
        self.inputs = []

    def activation(self, energy):
        """
        Inheriting classes must specify the activation function for this layer
        EG a sigmoid
        returns the update state: 1 or -1
        """
        raise NotImplementedError

    def add_input(self, input_connection):
        """
        add input_connection to the list of connections feeding into this layer
        """
        self.inputs.append(input_connection)
        # assert the inputs have the right dimension

    def update(self, idx):
        """
        update the state at idx
        """
        energy = self.bias[idx]
        for input_layer in self.inputs:
            energy += input_layer.get_energy(idx)
        self.state[idx] = self.activation(energy)



# subclassing for input layer may not be necessary
