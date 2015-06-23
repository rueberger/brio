
from misc.utils import overrides
import numpy as np

# to do: add a synchronous layer (eg for rbms)

class Layer(object):
    """
    Base class for a network layer
    Holds the state, binary only for now
    """

    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.state = np.ones(n_dims)
        # to do allow for specification of init method
        # implement mean histories
        self.bias = np.random.randn(n_dims)
        self.inputs = []
        self.outputs = []

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

    def add_output(self, output_connection):
        """
        add output_connection to the list of connections feeding out of this layer
        """
        self.outputs.append(output_connection)

    def input_energy(self, idx):
        """
        returns the energy fed into the the unit at idx by all input layers
        for feedforward networks this is the only relevant energy method
        """
        energy = 0
        for input_layer in self.inputs:
            energy += input_layer.feedforward_energy(idx)
        return energy

    def output_energy(self, idx):
        """
        returns the energy this unit feeds into its output layers
        for use in calculating the energy difference of a bitflip for boltzmann machines
        """
        energy = 0
        for output_layer in self.outputs:
            energy += output_layer.energy_shadow(idx)

    def update(self, idx):
        """
        update the state at idx
        """
        raise NotImplementedError


class BoltzmannMachineLayer(Layer):
    """
    Implements the Boltzman Machine actiation function
    """

    @overrides(Layer)
    def activation(self, energy):
        # might want an extra factor of two here to account
        # for the energy difference
        p_on = 1. / (1 + np.exp(-energy))
        if np.random.random() < p_on:
            return 1
        else:
            return -1

    @overrides(Layer)
    def update(self, idx):
        e_diff = self.bias[idx]
        e_diff += self.input_energy(idx)
        e_diff += self.output_energy(idx)
        self.state[idx] = self.activation(e_diff)

class PerceptronLayer(Layer):
    """
    Simple feedforward perceptron with a hard threshold activation function
    """

    @overrides(Layer)
    def activation(self, energy):
        if energy > 0:
            return 1
        else:
            return -1

    @overrides(Layer)
    def update(self, idx):
        energy = self.bias[idx] + self.input_energy(idx)
        self.state[idx] = self.activation(energy)

class InputLayer(Layer):
    """
    Input layer. Lacks update methods
    """

    def set_state(self, state):
        """
        set state as the current state of the layer
        state must be an array of shape (ndims, )
        """
        assert state.shape == self.state.shape
        self.state = state.copy()