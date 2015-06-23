"""
This module holds the Layer class and its various subclasses:
Layer: Base class for layers. Defines interface and share methods
PerceptronLayer: Layer subclass with a perceptron activation function
BoltzmannMachineLayer: Layer subclass with a Boltzmann Machine activation function
"""
from misc.utils import overrides
import numpy as np

# to do: add a synchronous layer (eg for rbms)

class Layer(object):
    """
    Base class for network layers.
    Defines the interface for layers and implements some common functionality
    To use, inheriting classes must override the activation and update methods
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
        """ The activation function determines the nonlinearity of units in this layer
        Must be implemented by inheriting classes
        Sets the state of a unit for a given input energy

        :param energy: the input energy to a unit.
           Generally the weighted outputs of units in the previous layer
        :returns: the updated state of the unit in {-1, 1}
        :rtype: int

        """

        raise NotImplementedError

    def update(self, idx):
        """ Update the unit at idx by summing the weighted contributions of its input units
        and running the activation function

        :param idx: idx of the unit to update. in range(self.n_dims)
        :returns: None
        :rtype: None

        """
        raise NotImplementedError

    def add_input(self, input_connection):
        """ add input_connection to the list of connections feeding into this layer
        This method is called when Connections are initialized

        :param input_connection: Connection object
        :returns: None
        :rtype: None

        """
        self.inputs.append(input_connection)

    def add_output(self, output_connection):
        """ add output_connection to the list of connections feeding out of this layer
        This method is called when Connections are initialized

        :param output_connection: Connection object
        :returns: None
        :rtype: None

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


class BoltzmannMachineLayer(Layer):
    """
    Implements the Boltzman Machine activation function
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