"""
This module holds the connection class and its subclasses:
Connection: Base class for connections. Defines interface and shared methods
"""
from misc.utils import overrides, normalize_by_row
import numpy as np
np.seterr('raise')

class Connection(object):
    """
    Base class for connection between network layers
    holds network weights
    """

    def __init__(self, input_layer, output_layer,
                 lrate_multiplier=1, weight_scheme='uniform'):
        self.presynaptic_layer = input_layer
        self.postsynaptic_layer = output_layer
        self.postsynaptic_layer.add_input(self)
        self.presynaptic_layer.add_output(self)
        self.weight_multiplier = self.presynaptic_layer.ltype.weight_multiplier
        self.lrate_multiplier = lrate_multiplier
        self.__init_weights(weight_scheme)


    def __init_weights(self, scheme):
        if scheme == 'uniform':
            self.weights = np.random.random((self.presynaptic_layer.n_dims,
                                             self.postsynaptic_layer.n_dims))
            if not self.presynaptic_layer.ltype.constrain_weights:
                self.weights = 2 * self.weights - 1
        elif scheme == 'gaussian':
            self.weights = np.random.randn((self.presynaptic_layer.n_dims,
                                            self.postsynaptic_layer.n_dims))
            if not self.presynaptic_layer.ltype.constrain_weights:
                self.weights = np.abs(self.weights)
        elif scheme == 'zero':
            self.weights = np.zeros((self.presynaptic_layer.n_dims,
                                     self.postsynaptic_layer.n_dims))
        else:
            raise NotImplementedError("please choose on of the implemented weight schemes")


    def weight_update(self):
        """ accumulate weight updates and apply them as specified
        by the network parameters
        This is the method to call from external code

        :returns: None
        :rtype: None
        """
        self.bulk_weight_update()
        if self.presynaptic_layer.ltype.constrain_weights:
            self.__impose_constraint()


    def bulk_weight_update(self):
        """ Local update rule for the weights in this connection
        performs a bulk weight update for the last update_batch_size
          number of state updates
        Must be implemented by inheriting class

        :returns: None
        """
        raise NotImplementedError

    def feedforward_energy(self, idx):
        """
        Returns the output into the unit at idx of the output layer
        """
        return np.sum(self.weight_multiplier * self.weights[:, idx] * self.presynaptic_layer.state)

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
        self.params = network.params
        self.learning_rate = self.lrate_multiplier *  network.params.baseline_lrate
        self.epoch_size = self.params.update_batch_size

    def __impose_constraint(self):
        """
        Constrain the weights according to the constraint multiplier
        """
        out_of_bounds_idx = (self.weights < 0)
        self.weights[out_of_bounds_idx] = 0

    def __repr__(self):
        """ overrides str for more useful info about connections

        :returns: descriptive string
        :rtype: string
        """
        return "{}: In: {}; Out: {}".format(type(self).__name__,
                                            self.presynaptic_layer.__str__(),
                                            self.postsynaptic_layer.__str__())


class OjaConnection(Connection):
    """
    Connection class that uses Oja's rule to iteratively update the weights
    """

    def __init__(self, input_layer, output_layer,
                 lrate_multiplier=1, weight_scheme='uniform'):
        super(OjaConnection, self).__init__(input_layer, output_layer,
                                            lrate_multiplier, weight_scheme)
        # stupid trick to normalize columns
        self.weights = normalize_by_row(self.weights.T).T


    # pylint: disable=too-few-public-methods

    @overrides(Connection)
    def bulk_weight_update(self):
        pre_syn_rates = np.array(self.presynaptic_layer.fr_history[:self.params.update_batch_size])
        post_syn_rates = np.array(self.postsynaptic_layer.fr_history[:self.params.update_batch_size])
        delta = (np.dot(pre_syn_rates.T, post_syn_rates) -
                 np.sum(post_syn_rates ** 2, axis=0) * self.weights)
        self.weights += self.learning_rate * delta

class FoldiakConnection(Connection):
    """
    Connection class that uses Foldiak's rule to iteratively update the weights
    """

    # pylint: disable=too-few-public-methods

    @overrides(Connection)
    def bulk_weight_update(self):
        pre_syn_rates = np.array(self.presynaptic_layer.fr_history[:self.epoch_size])
        post_syn_rates = np.array(self.postsynaptic_layer.fr_history[:self.epoch_size])
        pre_syn_avg_rates = self.presynaptic_layer.lfr_mean
        post_syn_avg_rates = self.postsynaptic_layer.lfr_mean
        delta = (np.dot(pre_syn_rates.T, post_syn_rates) -
                 np.outer(pre_syn_avg_rates, post_syn_avg_rates) * self.epoch_size)
        self.weights += self.learning_rate * delta



class CMConnection(Connection):
    """
    Connection class that uses the Correlation Measuring rule to iteratively update the weights
    """


    # pylint: disable=too-few-public-methods

    @overrides(Connection)
    def bulk_weight_update(self):
        pre_syn_rates = np.array(self.presynaptic_layer.fr_history[:self.epoch_size])
        post_syn_rates = np.array(self.postsynaptic_layer.fr_history[:self.epoch_size])
        # pre_syn_avg_rates = self.presynaptic_layer.lfr_mean
        # post_syn_avg_rates = self.postsynaptic_layer.lfr_mean
        pre_syn_avg_rates = self.presynaptic_layer.target_firing_rate
        post_syn_avg_rates = self.postsynaptic_layer.target_firing_rate
        delta = (np.dot(pre_syn_rates.T, post_syn_rates) -
                 np.outer(pre_syn_avg_rates, post_syn_avg_rates) *
                 self.epoch_size * (1 + self.weights))
        self.weights += self.learning_rate * delta

class ConstantConnection(Connection):
    """
    A connection class with no learning rule
    Intended to be use as simple feedforward weights from the input layer to the first layer
    """

    def __init__(self, input_layer, output_layer):
        super(ConstantConnection, self).__init__(input_layer, output_layer)
        assert input_layer.n_dims == output_layer.n_dims
        self.weights = np.diag(np.ones(input_layer.n_dims))

    @overrides(Connection)
    def bulk_weight_update(self):
        pass