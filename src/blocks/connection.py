"""
This module holds the connection class and its subclasses:
Connection: Base class for connections. Defines interface and shared methods
"""
from misc.utils import overrides
import numpy as np
np.seterr('raise')

class Connection(object):
    """
    Base class for connection between network layers
    holds network weights
    """

    def __init__(self, input_layer, output_layer,
                 learning_rate_multiplier=1):
        self.presynaptic_layer = input_layer
        self.postsynaptic_layer = output_layer
        self.weights = np.random.randn(input_layer.n_dims, output_layer.n_dims) * 0.01
        self.postsynaptic_layer.add_input(self)
        self.presynaptic_layer.add_output(self)
        self.weight_multiplier = self.presynaptic_layer.ltype.weight_multiplier
        self.learning_rate_multiplier = learning_rate_multiplier

    def weight_rule(self):
        """ Local update rule for the weights in this connection
        Must be implemented by inheriting class

        :returns: None
        """
        raise NotImplementedError

    def apply_weight_rule(self):
        """ Updates the weights in this connection according to
          the update rule (which must be specified by inheriting classes)
        Constrains the weights to be non-negative if the input layer is inhibitory or excitatory

        :returns: None
        """
        self.weight_rule()
        if self.presynaptic_layer.ltype.constrain_weights:
            self.__impose_constraint()

    def feedforward_energy(self, idx):
        """
        Returns the output into the unit at idx of the output layer
        """
        return np.sum(self.weight_multiplier * self.weights[:, idx] * self.presynaptic_layer.state)

    # to do: better name needed: use pre and post synaptic
    def energy_shadow(self, input_idx):
        """
        return the energy of the output states 'shadowed' by the input unit
          at input_idx
        for use in calculating energy difference for boltzmann machines
        """
        return np.sum(self.weight_multiplier * self.weights[input_idx, :] * self.postsynaptic_layer.state)

    def unpack_network_params(self, network):
        """ unpacks parameters from parent network
        For now only unpacks the learning rate

        :param network: Network object. The parent network
        :returns: None
        :rtype: None
        """

        self.learning_rate = network.params.weight_learning_rate * self.learning_rate_multiplier

    def __impose_constraint(self):
        """
        Constrain the weights according to the constraint multiplier
        """
        out_of_bounds_idx = (self.weights < 0)
        self.weights[out_of_bounds_idx] = 0


class OjaConnection(Connection):
    """
    Connection class that uses Oja's rule to iteratively update the weights
    """

    # pylint: disable=too-few-public-methods

    @overrides(Connection)
    def weight_rule(self):
        pre_syn_state = self.presynaptic_layer.history[-1]
        post_syn_state = self.postsynaptic_layer.history[-1]
        delta = np.outer(pre_syn_state, post_syn_state) - (post_syn_state ** 2) * self.weights
        self.weights += self.learning_rate * delta

class FoldiakConnection(Connection):
    """
    Connection class that uses Foldiak's rule to iteratively update the weights
    """

    # pylint: disable=too-few-public-methods

    @overrides(Connection)
    def weight_rule(self):
        pre_syn_state = self.presynaptic_layer.history[-1]
        post_syn_state = self.postsynaptic_layer.history[-1]
        pre_syn_avg_rates = self.presynaptic_layer.firing_rates()
        post_syn_avg_rates = self.postsynaptic_layer.firing_rates()
        delta = (np.outer(pre_syn_state, post_syn_state) -
                 np.outer(pre_syn_avg_rates, post_syn_avg_rates))
        self.weights += self.learning_rate * delta

class CMConnection(Connection):
    """
    Connection class that uses the Correlation Measuring rule to iteratively update the weights
    """


    # pylint: disable=too-few-public-methods

    @overrides(Connection)
    def weight_rule(self):
        pre_syn_state = self.presynaptic_layer.history[-1]
        post_syn_state = self.postsynaptic_layer.history[-1]
        pre_syn_avg_rates = self.presynaptic_layer.firing_rates()
        post_syn_avg_rates = self.postsynaptic_layer.firing_rates()
        delta = (np.outer(pre_syn_state, post_syn_state) -
                 np.outer(pre_syn_avg_rates, post_syn_avg_rates)) * (1 + self.weights)
        self.weights += self.learning_rate * delta
