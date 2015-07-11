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



class Layer(object):
    """
    Base class for network layers.
    Defines the interface for layers and implements some common functionality
    To use, inheriting classes must override the async_activation and async_update methods
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, n_dims, ltype=LayerType.unconstrained,
                 update_bias=True, allow_self_con=True):
        """ Initialize Layer object

        :param n_dims: the number of neurons in the layer
        :param ltype: Enum holding constants particular to layer type.
           unconstrainted, excitatory or inhibitory
        :param update_bias: True if this layer should update its biases following foldiak's rule during traing
        :param allow_self_con: True if neurons in this layer are allowed to connect to themselves
        :returns: a Layer object
        :rtype: Layer
        """

        self.n_dims = n_dims
        self.state = np.zeros(n_dims)
        self.state[np.random.random(n_dims) < .5] = 0
        self.bias = np.ones(self.n_dims) * 2
        self.bias_updates = []
        self.inputs = []
        self.outputs = []
        self.ltype = ltype
        self.update_sign = 1
        # hodge podge of firing rate attributes
        self.history = [self.state.copy()]
        self.firing_rates = np.zeros(self.n_dims)
        self.fr_max = 1
        self.fr_history = []
        # initialized when target firing rate is imported
        self.lfr_mean = None
        self.update_bias = update_bias
        self.allow_self_con = allow_self_con

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

    def bias_update(self):
        """ Update the unit biases for this layer
        By default uses the homeostatic threshold rule from
          Foldiak 1990

        :returns: None
        :rtype: None
        """
        # incredibly confusing, but the bias update does NOT use the exponential
        # moving average used for the firing rate everywhere else....
        # inelegant and I hope not necessary but this comes directly out of the
        # EI net implementation
        if self.update_bias:
            epoch_time_units = self.params.update_batch_size * self.params.timestep
            delta = self.target_firing_rate - self.epoch_fr
            self.bias += (self.update_sign * self.learning_rate * delta * epoch_time_units)
            # sensible for LIF neurons but will want to change for others....
            self.bias[self.bias < 0] = 0

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
        time_constant = 1. / (network.params.steps_per_fr_time)
        self.max_history_length = network.params.layer_history_length
        self.avg_weighting = np.exp(
            - time_constant * np.arange(self.max_history_length))[:, np.newaxis]
        self.avg_weighting *= 1. / (np.sum(self.avg_weighting))
        self.target_firing_rate = (self.ltype.firing_rate_multiplier *
                                   network.params.baseline_firing_rate)
        self.learning_rate = network.params.bias_learning_rate * network.params.baseline_lrate
        self.params = network.params
        self.lfr_mean = np.ones(self.n_dims) * self.target_firing_rate
        self.fr_bias = np.ones(self.n_dims) * self.target_firing_rate


    def update_lifetime_mean(self):
        """ Updates the lifetime mean firing rate for this layer

        :returns: None
        :rtype: None
        """
        self.epoch_fr = (np.mean(self.fr_history[:self.params.layer_history_length], axis=0) /
                         self.params.timestep)
        self.lfr_mean += self.params.ema_lfr * (np.mean(self.firing_rates, axis=0) / self.params.timestep
                                                - self.lfr_mean)



    def update_history(self):
        """ appends the current state to the history
        additionally truncates the history if it grows too long

        :returns: None
        :rtype: None
        """
        self.history.insert(0, self.state.copy())
        self.fr_history.append(self.firing_rates)
        if len(self.history) > 2 * self.max_history_length:
            self.history = self.history[:self.max_history_length]
            self.fr_history = self.fr_history[self.max_history_length:]

    def update_firing_rates(self):
        """ Compute the current firing rates
          weighted by a decaying exponential.
        The time constant is set as the inverse of the number of presentations
          for each stimulus for the parent network.
        Should only be called once per simulation step

        :returns: None
        :rtype: None
        """
        self.fr_bias += self.params.ema_curr * (self.state - self.fr_bias)
        self.fr_max += self.params.ema_curr * (1 - self.fr_max)
        # normalize
        self.firing_rates = self.fr_bias / self.fr_max

    def reset(self):
        """ reset the state for this layer

        :returns: None
        :rtype: None
        """
        self.state = np.zeros(self.n_dims)

    def __repr__(self):
        """
        A nicer string for this class
        """
        return "{} layer of size {}".format(self.ltype.name, self.n_dims)


class LIFLayer(Layer):
    """
    Implements a layer of leaky integrate and fire neurons
    """

    def __init__(self, *args, **kwargs):
        super(LIFLayer, self).__init__(*args, **kwargs)
        # now state represents spikes and still works with everything else
        self.potentials = np.zeros(self.n_dims)
        self.update_sign = -1
        self.pot_history = []
        # messy trick: inhibitory neurons have a faster firing rate and this gives them
        # a faster rc time constant too
        self.decay_scale = self.ltype.firing_rate_multiplier

    @overrides(Layer)

    def sync_update(self):
        """ Implements synchronous state update for leaky integrate and fire neurons

        :returns: None
        :rtype: None
        """
        # update mebrane potentials
        self.potentials *= np.exp(-self.decay_scale / float(self.params.steps_per_rc_time))
        for input_connection in self.inputs:
            multiplier = input_connection.weight_multiplier
            weights = input_connection.weights.T
            state = input_connection.presynaptic_layer.history[0]
            self.potentials += multiplier * np.dot(weights, state)
        # set state for neurons that cross threshold
        fire_idxs = np.where(self.potentials >= self.bias)
        self.state = np.zeros(self.n_dims)
        self.state[fire_idxs] = 1
        # reset membrane potential
        self.potentials[fire_idxs] = 0
        if self.params.keep_extra_history:
            if len(self.pot_history) >= self.params.presentations:
                self.pot_history = []
            self.pot_history.append(self.potentials.copy())



    @overrides(Layer)
    def reset(self):
        self.state = np.zeros(self.n_dims)
#        self.potentials = np.random.random(self.n_dims) * self.bias
        self.potentials = np.zeros(self.n_dims)


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
    def sync_update(self):
        """ Implemented a synchronous state update for a perceptron layer

        :returns: None
        :rtype: None
        """
        energy = self.bias.copy()
        for input_connection in self.inputs:
            multiplier = input_connection.weight_multiplier
            weights = input_connection.weights.T
            state = input_connection.presynaptic_layer.history[0]
            energy += multiplier * np.dot(weights, state)
        update_idxs = np.where(energy > 0)[0]
        self.state = np.zeros(self.n_dims)
        self.state[update_idxs] = 1


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
        # current injection per unit time
        # unit of current is in membrane rc time
        self.state = flat_state.copy() / float(self.params.steps_per_rc_time)


class RasterInputLayer(Layer):
    """
    An input layer that contains methods to rasterize scalar variables into spike trains
    The range of the scalar variable is partitioned into equal bins from specified bounds and n_dims
    Each bin is represented by a single neuron with independent poisson spiking behavior
    For each stimulus value, the rate at which a particular neuron fires is computed as the integral
       of a gaussian centered around the stimulus value across the bin that neuron codes for
    """

    def __init__(self, n_dims, min_range, max_range, **kwargs):
        super(RasterInputLayer, self).__init__(n_dims, **kwargs)
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
        p_fire_in_bin = 1 - np.exp(-rates)
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
