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
                 update_bias=True, allow_self_con=False):
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
        self.bias = np.ones((self.n_dims, 1))
        self.inputs = []
        self.outputs = []
        self.ltype = ltype
        self.update_sign = 1
        self.update_bias = update_bias
        self.allow_self_con = allow_self_con
        self.update_cap = 0.5


    def set_up(self, network):
        """ adds an attribute pointing to the parent network and sets up the
        weighting used for computing firing rates
        Also sets the target firing rate and the max history length

        :param network: Network object. The parent network
        :returns: None
        :rtype: None
        """
        self.params = network.params
        # import params
        self.max_history_length = network.params.layer_history_length
        self.target_firing_rate = (self.ltype.firing_rate_multiplier *
                                   network.params.baseline_firing_rate)
        self.learning_rate = network.params.bias_learning_rate * network.params.baseline_lrate
        self.stim_per_epoch = self.params.stimuli_per_epoch

        # initialize attributes
        self.state = np.zeros((self.n_dims, self.stim_per_epoch))
        self._history = [self.state.copy()]
        self._firing_rates = self.state.copy()
        self._fr_history = []
        self._lfr_mean = np.ones(self.n_dims) * self.target_firing_rate
        self._epoch_fr = np.zeros((self.n_dims, self.stim_per_epoch))
        # additional set up for inheriting layers (if necessary)
        self.aux_set_up()

    @property
    def firing_rates(self):
        """ Property for firing rates

        :returns _firing_rates
        :rtype: array
        """
        return self._firing_rates

    @property
    def fr_history(self):
        """ Property for firing rate history

        :returns: _fr_history trimmed to proper length
        :rtype: array
        """
        return np.array(self._fr_history[:self.params.presentations])

    @property
    def lfr_mean(self):
        """ Property for lfr_mean

        :returns: _lfr_mean
        :rtype: array
        """
        return self._lfr_mean

    @property
    def epoch_fr(self):
        """ Property for epoch_fr

        :returns: _epoch_fr
        :rtype: array
        """
        return self._epoch_fr

    @property
    def prev_state(self):
        """ Property that returns the previous state

        :returns: history[0]
        :rtype: array
        """
        return self._history[0]


    def aux_set_up(self):
        """ This method is called after the main set up has finished executing
        Intended to be overrided by inheriting classes and used to set up
           Layer type specific state variables (such as potential)

        :returns: None
        :rtype: None
        """
        pass


    def sync_update(self):
        """ Synchronously updates the state of all of the units in this layer
        Must be implemented by inheriting class

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
        if self.update_bias:
            epoch_time_units = self.params.update_batch_size * self.params.timestep
            delta = (self.target_firing_rate - self.epoch_fr).reshape(-1, 1)
            delta_b = (self.update_sign * self.learning_rate * delta * epoch_time_units)
            if self.update_cap is not None:
                delta_b[np.where(delta_b > self.update_cap)] = self.update_cap
                delta_b[np.where(delta_b < -self.update_cap)] = -self.update_cap
            self.bias += delta_b

    def update_lifetime_mean(self):
        """ Updates the lifetime mean firing rate for this layer

        :returns: None
        :rtype: None
        """
        act_mean = np.mean(self._history[:self.params.layer_history_length], axis=(0, 2))
        fr_mean = np.mean(self._fr_history[:self.params.layer_history_length], axis=(0, 1))
        self._epoch_fr = (act_mean / self.params.timestep)
        self._lfr_mean += self.params.ema_lfr * ((fr_mean / self.params.timestep) - self._lfr_mean)

    def update_history(self):
        """ appends the current state to the history
        additionally updates the firing rates

        :returns: None
        :rtype: None
        """
        self._firing_rates += self.params.ema_curr * (self.state - self._firing_rates)
        self._history.insert(0, self.state.copy())
        self._fr_history.append(self._firing_rates.copy().T)

    def reset(self):
        """ Reset the layer in anticipation of running
          the next batch of stimuli
        Clears the history and calls a method which resets the Layer type specific
          state variables

        :returns: None
        :rtype: None
        """
        self.reset_state_vars()
        self._firing_rates = np.zeros((self.n_dims, self.stim_per_epoch))
        self._fr_history = []
        self._history = [self.state.copy()]


    def reset_state_vars(self):
        """ Reset the state variables for this layer such as state
           or membrane potential
        override this method when inheriting - NOT reset

        :returns: None
        :rtype: None
        """
        self.state = np.zeros((self.n_dims, self.stim_per_epoch))


    def __repr__(self):
        """
        A nicer string for this class
        """
        return "{} layer of size {}".format(self.ltype.name, self.n_dims)


class LIFLayer(Layer):
    """
    Implements a layer of leaky integrate and fire neurons

    Manages to remain compatible with everything else by using state to reinterpret spikes
      and defining auxiliary potential variables
    """

    def __init__(self, *args, **kwargs):
        super(LIFLayer, self).__init__(*args, **kwargs)
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
            state = input_connection.presynaptic_layer.prev_state
            self.potentials += multiplier * np.dot(weights, state)
        # set state for neurons that cross threshold
        fire_idxs = np.where(self.potentials >= self.bias)
        self.state = np.zeros((self.n_dims, self.stim_per_epoch))
        self.state[fire_idxs] = 1
        # reset membrane potential
        self.potentials[fire_idxs] = 0
        if self.params.keep_extra_history:
            if len(self.pot_history) >= self.params.presentations:
                self.pot_history = []
            self.pot_history.append(self.potentials.copy())

    @overrides(Layer)
    def aux_set_up(self):
        self.potentials = np.zeros((self.n_dims, self.stim_per_epoch))

    @overrides(Layer)
    def reset_state_vars(self):
        self.state = np.zeros((self.n_dims, self.stim_per_epoch))
        self.potentials = np.zeros((self.n_dims, self.stim_per_epoch))
        self.pot_history = []

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
        delta_e = np.tile(self.bias, self.stim_per_epoch).reshape(-1, self.stim_per_epoch)
        for input_connection in self.inputs:
            multiplier = input_connection.weight_multiplier
            weights = input_connection.weights.T
            state = input_connection.presynaptic_layer.prev_state
            delta_e += multiplier * np.dot(weights, state)
        for output_connection in self.outputs:
            multiplier = output_connection.weight_multiplier
            weights = output_connection.weights
            state = output_connection.postsynaptic_layer.prev_state
            delta_e += multiplier * np.dot(weights, state)

        p_on = 1. / (1 + np.exp(-delta_e))
        rand_p = np.random.random((self.n_dims, self.stim_per_epoch))
        update_idxs = np.where(rand_p < p_on)
        self.state = np.zeros((self.n_dims, self.stim_per_epoch))
        self.state[update_idxs] = 1



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
        energy = np.tile(self.bias, self.stim_per_epoch).reshape(-1, self.stim_per_epoch)
        for input_connection in self.inputs:
            multiplier = input_connection.weight_multiplier
            weights = input_connection.weights.T
            state = input_connection.presynaptic_layer.prev_state
            energy += multiplier * np.dot(weights, state)
        update_idxs = np.where(energy > 0)
        self.state = np.zeros((self.n_dims, self.stim_per_epoch))
        self.state[update_idxs] = 1


#######################
#   Input layers      #
#######################


class InputLayer(Layer):
    """
    Input layer. Lacks async_update methods
    """

    def __init__(self, *args, **kwargs):
        super(InputLayer, self).__init__(*args, **kwargs)
        self.update_bias = False

    def set_state(self, state):
        """ set state as the state of the input layer

        :param state: array of shape (self.n_dims, self.stimuli_per_epoch)
        :returns: None
        :rtype: None
        """
        assert state.shape == (self.n_dims, self.stim_per_epoch)
        # current injection per unit time
        # unit of current is in membrane rc time
        self.state = state.copy() / float(self.params.steps_per_rc_time)
        self._history.insert(0, self.state.copy())


    @overrides(Layer)
    def sync_update(self):
        pass

class RasterInputLayer(InputLayer):
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
        # dviding by 1E4 produces a pretty wide distribution of rates
        # probably a good starting point for
        # current variance of gaussian
        self.var = (max_range - min_range) / 1E5
        # overall scale of gaussian. 1 is normalized
        self.scale = 3
        # how long in each time bin
        # also need to represent cooling schedule somehow


    @overrides(InputLayer)
    def aux_set_up(self):
        self.sample_points = np.tile(np.linspace(self.lower_bnd, self.upper_bnd, self.n_dims),
                                     self.stim_per_epoch).reshape(
                                         self.n_dims, self.stim_per_epoch)
    @overrides(InputLayer)
    def set_state(self, scalar_value):
        """ sets the state of this layer probabilistically according to the scheme
          described in the class header doc

        :param scalar_value: a scalar. must be in (min_range, max_range)
        :returns:  None
        :rtype: None
        """
        assert (self.lower_bnd < scalar_value).all()
        assert (scalar_value < self.upper_bnd).all()
        rates = self.rate_at_points(scalar_value)
        p_fire_in_bin = 1 - np.exp(-rates)
        rand_p = np.random.random((self.n_dims, self.stim_per_epoch))
        firing_idx = np.where(rand_p < p_fire_in_bin)
        self.state = np.zeros((self.n_dims, self.stim_per_epoch))
        self.state[firing_idx] = 1
        self._history.insert(0, self.state.copy())


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


class SplitInput(InputLayer):
    """
    An input layer that accepts multiple stimuli simultaneously
    Must not have any outputs
    """

    # pylint:disable=too-many-instance-attributes

    def __init__(self, n_dims, n_children, **kwargs):
        super(SplitInput, self).__init__(n_dims, **kwargs)
        self.update_bias = False
        self.children = [InputLayer(n_dims, **kwargs) for _ in xrange(n_children)]

    @overrides(InputLayer)
    def set_state(self, rolled_stimuli_set):
        """ Set the state of all the child layers

        :param rolled_stimuli_set: a list of stimuli of len n_children,
         each element of the list is an array of rolled stimuli of shape (n_dims, stimuli_per_epoch)
        :returns: None
        :rtype: None
        """
        if len(rolled_stimuli_set) == len(self.children):
            for stimulus, child_layer in zip(rolled_stimuli_set, self.children):
                child_layer.set_state(stimulus)
        elif rolled_stimuli_set.shape[0] % n_stim_dims == 0:
            # all children must have the same dimension if stimulus must be implicitly decoded
            n_stim_dims = self.children[0].n_dims
            assert (np.array([c.n_dims for c in self.children]) == n_stim_dims).all()
            for idx, child_layer in enumerate(rolled_stimuli_set):
                child_layer.set_state(rolled_stimuli_set[n_stim_dims * idx: n_stim_dims * (idx + 1)])
        else:
            raise ValueError("Stimuli provided could not be implicitly decoded")
        self.aux_update()

    def aux_update(self):
        """ auxiliary updates to perform after setting child state
        to be implemented by inheriting classes

        :returns: None
        :rtype: None
        """
        pass

    @overrides(InputLayer)
    def sync_update(self):
        pass


    @overrides(InputLayer)
    def aux_set_up(self):
        """
        Check that this layer is disconnected
        """
        assert len(self.outputs) == 0
        assert len(self.inputs) == 0


class GatedInput(SplitInput):
    """
    An input layer that accepts multiple stimuli simultaneously
    State of this layer is set by gating inptu stimuli multiplicatively
    Output must be a ConstantConnection
    Sets firing rates to those of the postsynaptic  to
      preserve weight rule functionality
    """

    # pylint:disable=too-many-instance-attributes

    def __init__(self, n_dims, input_n_dims, n_children, **kwargs):
        super(GatedInput, self).__init__(n_dims, **kwargs)
        self.update_bias = False
        self.children = [InputLayer(input_n_dims, **kwargs) for _ in xrange(n_children)]

    @overrides(SplitInput)
    def aux_update(self)
        update_state = np.ones((self.n_dims, self.stim_per_epoch))
        for input_connection in self.inputs:
            multiplier = input_connection.weight_multiplier
            weights = input_connection.weights.T
            state = input_connection.presynaptic_layer.prev_state
            update_state *= multiplier * np.dot(weights, state)
        self.state = update_state
        self._history.insert(0, self.state.copy())

    @overrides(InputLayer)
    def aux_set_up(self):
        """
        Set the firing rate and history as those of the postsynaptic neuron
        """
        from blocks.connection import ConstantConnection

        assert len(self.outputs) == 1
        assert len(self.inputs) == len(self.children)
        assert isinstance(self.outputs[0], ConstantConnection)
        self.parent_layer = self.outputs[0].postsynaptic_layer

    @overrides(InputLayer)
    def update_lifetime_mean(self):
        pass

    @overrides(InputLayer)
    def update_history(self):
        self._history.insert(0, self.state.copy())

    @property
    @overrides(InputLayer)
    def firing_rates(self):
        """ Property for firing rates

        :returns _firing_rates
        :rtype: array
        """
        return self.parent_layer.firing_rates

    @property
    @overrides(InputLayer)
    def fr_history(self):
        """ Property for firing rate history

        :returns: _fr_history trimmed to proper length
        :rtype: array
        """
        return self.parent_layer.fr_history

    @property
    @overrides(InputLayer)
    def lfr_mean(self):
        """ Property for lfr_mean

        :returns: _lfr_mean
        :rtype: array
        """
        return self.parent_layer.lfr_mean

    @property
    @overrides(InputLayer)
    def epoch_fr(self):
        """ Property for epoch_fr

        :returns: _epoch_fr
        :rtype: array
        """
        return self.parent_layer.epoch_fr
