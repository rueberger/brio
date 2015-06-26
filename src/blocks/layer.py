"""
This module holds the Layer class and its various subclasses:
Layer: Base class for layers. Defines interface and shared methods
PerceptronLayer: Layer subclass with a perceptron activation function
BoltzmannMachineLayer: Layer subclass with a Boltzmann Machine activation function
"""

from misc.utils import overrides
from blocks.aux import LayerType
import numpy as np

# to do: add a synchronous layer (eg for rbms)

class Layer(object):
    """
    Base class for network layers.
    Defines the interface for layers and implements some common functionality
    To use, inheriting classes must override the activation and update_state methods
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, n_dims, ltype=LayerType.unconstrained):
        self.n_dims = n_dims
        self.state = np.ones(n_dims)
        # to do allow for specification of init method
        self.bias = np.random.randn(n_dims)
        self.inputs = []
        self.outputs = []
        self.history = []
        self.ltype = ltype


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

    def update_state(self, idx):
        """ Update the unit at idx by summing the weighted contributions of its input units
        and running the activation function
        Must be implemented by inheriting class

        :param idx: idx of the unit to update. in range(self.n_dims)
        :returns: None
        :rtype: None

        """
        raise NotImplementedError

    def apply_bias_rule(self):
        """ Update the unit biases for this layer
        By default uses the homeostatic threshold rule from
          Foldiak 1990

        :returns: None
        :rtype: None
        """
        # to do add a non windowed mean to see how this does
        delta = self.firing_rates() - self.target_firing_rate
        self.bias += self.learning_rate * delta


    def add_input(self, input_connection):
        """ add input_connection to the list of connections feeding into this layer
        This method is called when Connections are initialized

        :param input_connection: Connection object
        :returns: None
        :rtype: None

        """
        assert input_connection not in self.inputs
        self.inputs.append(input_connection)

    def add_output(self, output_connection):
        """ add output_connection to the list of connections feeding out of this layer
        This method is called when Connections are initialized

        :param output_connection: Connection object
        :returns: None
        :rtype: None

        """
        assert output_connection not in self.outputs
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
        return energy

    def unpack_network_params(self, network):
        """ adds an attribute pointing to the parent network and sets up the
        weighting used for computing firing rates
        Also sets the target firing rate and the max history length

        :param network: Network object. The parent network
        :returns: None
        :rtype: None
        """
        time_constant = 1./ network.params.presentations
        normalizing_constant = (np.sqrt(np.pi) /  (2 * time_constant))
        self.avg_weighting = normalizing_constant * np.exp(
            - time_constant * np.arange(2 * self.max_history_length))
        self.target_firing_rate = (self.ltype.firing_rate_multiplier *
                                   network.params.baseline_firing_rate)
        self.max_history_length = network.params.layer_history_length
        self.learning_rate = network.params.bias_learning_rate



    def update_history(self):
        """ appends the current state to the history
        additionally truncates the history if it grows too long

        :returns: None
        :rtype: None
        """
        self.history.insert(0, self.state)
        if len(self.history) > 2 * self.max_history_length:
            self.history = self.history[:self.max_history_length]

    def firing_rates(self):
        """ returns the mean firing rate for the units in this layer
          weighted by a decaying exponential.
        The time constant is set as the inverse of the number of presentations
          for each stimulus for the parent network.

        :returns: weighted firing rates
        :rtype: float array
        """
        return np.sum(self.history * self.avg_weighting[:len(self.history)], axis=0)


class BoltzmannMachineLayer(Layer):
    """
    Implements the Boltzman Machine activation function
    """

    @overrides(Layer)
    def activation(self, energy):
        """ The Boltzmann Machine activation function
        Returns the state of the unit selected stochastically from a sigmoid

        :param energy: actually energy difference in this case between unit up and down
        :returns: the updated state of the unit in {-1, 1}
        :rtype: int
        """

        p_on = 1. / (1 + np.exp(-energy))
        if np.random.random() < p_on:
            return 1
        else:
            return -1

    @overrides(Layer)
    def update_state(self, idx):
        """ Updates the state of the unit at idx according to the Boltzmann Machine scheme
        Calculates the global energy difference between the unit at idx being up and down
        Sets the unit stochastically according the Boltzmann Machine activation function

        :param idx: idx of the unit to update. in range(self.n_dims)
        :returns: None
        :rtype: None
        """
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
        """ Perceptron activation rule
        A simple hard threshold

        :param energy: feedforward contributions
        :returns: the updated state of the unit in {-1, 1}
        :rtype: int
        """

        if energy > 0:
            return 1
        else:
            return -1

    @overrides(Layer)
    def update_state(self, idx):
        """ Updates the state of the unit at idx according to the Perceptron scheme
        Computes the feedforward contributions to the unit at idx and uses the hard threshold
          in the activation function to compute the update

        :param idx: idx of the unit to update. in range(self.n_dims)
        :returns: None
        :rtype: None
        """
        energy = self.bias[idx] + self.input_energy(idx)
        self.state[idx] = self.activation(energy)

class InputLayer(Layer):
    """
    Input layer. Lacks update_state methods
    """

    def set_state(self, state):
        """
        set flat_state as the current flat_state of the layer
        flat_state must be an array of shape (ndims, )
        """
        flat_state = np.ravel(state)
        assert flat_state.shape == self.state.shape
        self.state = flat_state.copy()
