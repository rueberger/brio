
import numpy as np

class Connection(object):
    """
    Base class for connection between network layers
    holds network weights
    """

    def __init__(self, input_layer, output_layer, constraint=None):
        # to do: add excitatory flag
        self.input_layer = input_layer
        self.output_layer = output_layer
        # allow for specification of input method
        self.weights = np.random.randn(input_layer.ndims, output_layer.ndims)
        self.output_layer.add_input(self)
        self.input_layer.add_output(self)
        if constraint is not None:
            self.weight_multiplier = {
                'excitatory': 1,
                'inhibitory': -1
            }[constraint]


    # input energy and output energy methods needed

    def feedforward_energy(self, idx):
        """
        Returns the output into the unit at idx of the output layer
        """
        return np.sum(self.weight_multiplier * self.weights[:, idx] * self.input_layer.state)

    # to do: better name needed
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
