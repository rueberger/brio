"""
This module holds the Layer class and its various subclasses:
Layer: Base class for layers. Defines interface and shared methods
PerceptronLayer: Layer subclass with a perceptron activation function
BoltzmannMachineLayer: Layer subclass with a Boltzmann Machine activation function
"""

from misc.utils import overrides
from blocks.aux import LayerType
import numpy as np
np.seterr('raise')

# to do: add a synchronous layer (eg for rbms)


class Layer(object):
    """
    Base class for network layers.
    Defines the interface for layers and implements some common functionality
    To use, inheriting classes must override the async_activation and async_update methods
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, n_dims, ltype=LayerType.unconstrained):
        self.n_dims = n_dims
        # randomly initialize state
        self.state = np.ones(n_dims)
        self.state[np.random.random(n_dims) < .5] = 0
        # to do allow for specification of init method
        self.bias = np.zeros(n_dims)
        self.inputs = []
        self.outputs = []
        self.history = [self.state.copy()]
        self.ltype = ltype


    def sync_update(self):
        """ Synchronously updates the state of all of the units in this layer
        Must be implemented by inheriting class

        :returns: None
        :rtype: None
        """
        raise NotImplementedError


    def async_activation(self, energy):
        """ The activation function determines the nonlinearity of units in this layer
        Must be implemented by inheriting classes
        Sets the state of a unit for a given input energy

        :param energy: the input energy to a unit.
           Generally the weighted outputs of units in the previous layer
        :returns: the updated state of the unit in {0, 1}
        :rtype: int
        """
        raise NotImplementedError

    def async_update(self, idx):
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
        # is this the right sign convention for all layers?
        # delta = self.firing_rates() - self.target_firing_rate
        delta = self.target_firing_rate - self.firing_rates()
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
        # avoids double counting for recurrent connections
        if output_connection not in self.inputs:
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
        # time_constant = 1./ network.params.presentations
        time_constant = 1./ (network.params.presentations * network.params.char_itrs)
        self.max_history_length = network.params.layer_history_length
        # I don't think normalization matters
        self.avg_weighting = np.exp(
            - time_constant * np.arange(self.max_history_length))[:, np.newaxis]
        self.avg_weighting *= 1. / (np.sum(self.avg_weighting))
        self.target_firing_rate = (self.ltype.firing_rate_multiplier *
                                   network.params.baseline_firing_rate)
        self.learning_rate = network.params.bias_learning_rate



    def update_history(self):
        """ appends the current state to the history
        additionally truncates the history if it grows too long

        :returns: None
        :rtype: None
        """
        self.history.insert(0, self.state.copy())
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
        rectified_hist = np.array(self.history[:self.max_history_length])
        return np.sum(rectified_hist * self.avg_weighting, axis=0)

    def __repr__(self):
        """
        A nicer string for this class
        """
        return "{} layer of size {}".format(self.ltype.name, self.n_dims)


class BoltzmannMachineLayer(Layer):
    """
    Implements the Boltzman Machine async_activation function
    """

    @overrides(Layer)
    def sync_update(self):
        """ Implements synchronous state update for Boltzmann Machines

        :returns: None
        :rtype: None
        """
        # need to check and write tests for this
        delta_e = self.bias.copy()
        for input_connection in self.inputs:
            multiplier = input_connection.weight_multiplier
            weights = input_connection.weights.T
            state = input_connection.presynaptic_layer.history[0]
            delta_e += multiplier * np.dot(weights, state)
        for output_connection in self.outputs:
            multiplier = output_connection.weight_multiplier
            weights = output_connection.weights
            state = output_connection.postsynaptic_layer.history[0]
            delta_e += multiplier * np.dot(weights, state)

        p_on = 1. / (1 + np.exp(-delta_e))
        update_idxs = np.where(np.random.random(self.n_dims) < p_on)[0]
        self.state = np.zeros(self.n_dims)
        self.state[update_idxs] = 1


    @overrides(Layer)
    def async_activation(self, energy):
        """ The Boltzmann Machine activation function
        Returns the state of the unit selected stochastically from a sigmoid

        :param energy: actually energy difference in this case between unit up and down
        :returns: the updated state of the unit in {-1, 1}
        :rtype: int
        """
        if energy > 200:
            return 1
        elif energy < -200:
            return 0
        else:
            p_on = 1. / (1 + np.exp(-energy))
            if np.random.random() < p_on:
                return 1
            else:
                return 0

    @overrides(Layer)
    def async_update(self, idx):
        """ Updates the state of the unit at idx according to the Boltzmann Machine scheme
        Calculates the global energy difference between the unit at idx being up and down
        Sets the unit stochastically according the Boltzmann Machine activation function

        :param idx: idx of the unit to update. in range(self.n_dims)
        :returns: None
        :rtype: None
        """
        delta_e = self.bias[idx]
        delta_e += self.input_energy(idx)
        delta_e += self.output_energy(idx)
        self.state[idx] = self.async_activation(delta_e)

class PerceptronLayer(Layer):
    """
    Simple feedforward perceptron with a hard threshold activation function
    """

    @overrides(Layer)
    def async_activation(self, energy):
        """ Perceptron activation rule
        A simple hard threshold

        :param energy: feedforward contributions
        :returns: the updated state of the unit in {-1, 1}
        :rtype: int
        """

        if energy > 0:
            return 1
        else:
            return 0

    @overrides(Layer)
    def async_update(self, idx):
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
    Input layer. Lacks async_update methods
    """

    def set_state(self, state):
        """
        set flat_state as the current flat_state of the layer
        flat_state must be an array of shape (ndims, )
        """
        # will want to record input data shape for plotting purposesx
        flat_state = np.ravel(state)
        assert flat_state.shape == self.state.shape
        self.state = flat_state.copy()

class RasterInputLayer(Layer):
    """
    An input layer that contains methods to rasterize scalar variables into spike trains
    The range of the scalar variable is partitioned into equal bins from specified bounds and n_dims
    Each bin is represented by a single neuron with independent poisson spiking behavior
    For each stimulus value, the rate at which a particular neuron fires is computed as the integral
       of a gaussian centered around the stimulus value across the bin that neuron codes for
    """

    def __init__(self, n_dims, min_range, max_range,
                 ltype=LayerType.unconstrained):
        super(RasterInputLayer, self).__init__(n_dims, ltype)
        assert min_range < max_range
        self.lower_bnd = min_range
        self.upper_bnd = max_range
        self.sample_points = np.linspace(min_range, max_range, n_dims)
        # dviding by 1E4 produces a pretty wide distribution of rates
        # probably a good starting point for
        # current variance of gaussian
        self.var = (max_range - min_range) / 1E5
        # overall scale of gaussian. 1 is normalized
        self.scale = 3
        # how long in each time bin
        self.timestep = 0.5
        # also need to represent cooling schedule somehow

    def set_state(self, scalar_value):
        """ sets the state of this layer probabilistically according to the scheme
          described in the class header doc

        :param scalar_value: a scalar. must be in (min_range, max_range)
        :returns:  None
        :rtype: None
        """
        assert self.lower_bnd < scalar_value < self.upper_bnd
        rates = self.rate_at_points(scalar_value)
        p_fire_in_bin = 1 - np.exp(-rates * self.timestep)
        firing_idx = (np.random.random(self.n_dims) < p_fire_in_bin)
        self.state = np.zeros(self.n_dims)
        self.state[firing_idx] = np.ones(self.n_dims)[firing_idx]


    def rate_at_points(self, scalar_value):
        """ returns an array with the rates at each sample point

        :param scalar_value: the value being coded for
        :returns: rate array
        :rtype: array

        """
        # right now not normalizing at all which means that larger variance vastly increases
        # firing rate of whole population
        # normalizing the gaussian will not translate to keeping constant firing rates across gaussian
        # need to normalize wrt the poisson cdf
        return self.scale * np.exp(- ((self.sample_points - scalar_value) ** 2) / (2 * self.var))

    def avg_activation(self, scalar_value):
        """ returns an array with the average activation of each neuron
        for testing purposes

        :param scalar_value: the value to test for activation for
        :returns: array of average activation
        :rtype: array
        """
        hist = []
        for _ in xrange(1000):
            self.set_state(scalar_value)
            hist.append(self.state)
        return np.mean(hist, axis=0)
