"""
This module holds the connection class and its subclasses:
Connection: Base class for connections. Defines interface and shared methods
"""
import numpy as np

class Connection(object):
    """
    Base class for connection between network layers
    holds network weights
    """

    def __init__(self, input_layer, output_layer,
                 learning_rate_multiplier=1):
        self.input_layer = input_layer
        self.output_layer = output_layer
        # allow for specification of input method
        self.weights = np.random.randn(input_layer.n_dims, output_layer.n_dims)
        self.output_layer.add_input(self)
        self.input_layer.add_output(self)
        self.weight_multiplier = self.input_layer.ltype.weight_multiplier
        self.learning_rate_multiplier = learning_rate_multiplier

    def __weight_rule(self):
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
        self.__weight_rule()
        if self.input_layer.ltype.constrain_weights:
            self.__impose_constraint()

    def feedforward_energy(self, idx):
        """
        Returns the output into the unit at idx of the output layer
        """
        return np.sum(self.weight_multiplier * self.weights[:, idx] * self.input_layer.state)

    # to do: better name needed: use pre and post synaptic
    def energy_shadow(self, input_idx):
        """
        return the energy of the output states 'shadowed' by the input unit
          at input_idx
        for use in calculating energy difference for boltzmann machines
        """
        return np.sum(self.weight_multiplier * self.weights[input_idx, :] * self.output_layer.state)

    def __impose_constraint(self):
        """
        Constrain the weights according to the constraint multiplier
        """
        out_of_bounds_idx = (self.weights < 0)
        self.weights[out_of_bounds_idx] = 0

class OjaConnection(object):
    """
    A layer that implements Oja's rule as the training
    """