# interface for network classes
# shared exposed methods: run network
# shared hidden methods: update state (asynchronous for now, which rules out the LIF neurons used in King et al - a major question to be answered is why they chose that scheme and if it will,
# work with boltzmann machines)
# shared variables: weights (input to E, E to I, I to I, I to E),
# to do: implement inhibitory pooling in the input


import numpy as np
from exceptions import NotImplementedError

class Network(object):
    """
    Generic interface class for networks that will be trained with the correlation measuring rule
    Implementing networks will share the same topology
    The difference will lie in the neuron representation
    Intended subclasses: McCullochPitts and Boltzmann at least, possibly LIF
    """

    def __init__(self, n_input, n_E, n_I):
        self.w_inp_E = np.random.randn(n_input, n_E)
        self.w_E_I = np.random.randn(n_E, n_I)
        self.w_I_I = np.random.randn(n_I, n_I)
        self.w_I_E = np.random.randn(n_I, n_E)
        self.theta_E = np.random.randn(n_E)
        self.theta_I = np.random.randn(n_I)

    def update_network(self, input):
        """
        Update all nodes according to the relevant update rule
        input: shape (n_input, )
        """
        raise NotImplementedError()

class DiscreteNet(Network):
    """
    Superclass for discrete state networks (McCullochPitts and Boltzmann)
    Contains network representation and share code for asynchronous updates
    """

    # async is really inefficient
    # can get away with sync updates?

    def __init__(self, *args):
        super(DiscreteNet, self).__init__(*args)
        # {-1, 1}^n
        self.y_E = np.ones(n_E)
        self.y_I = np.ones(n_I)

    def unit_energy(self, input, idx, E):
        """
        Returns the energy of the unit at idx
        E toggles excited or inhibitory
        """
        if E:
            E_input = np.sum(self.w_inp_E[:, idx] * input)
            E_inhib = -np.sum(self.w_I_E[:, idx] * self.y_I)
            # sign error here?
            E_bias = self.theta_E[idx] * self.y_E[idx]
            return E_inhib + E_input + E_bias
        else:
            E_excit = np.sum(self.w_E_I[:, idx] * self.y_E[idx])
            E_inhib = np.sum(self.w_I_I[:, idx] * self.y_I[idx])
            E_bias = np.sum(self.theta_I[idx] * self.y_I[idx])
            return E_inhib + E_excit + E_bias


    def update_unit(self, input, idx, E):
        """
        update the state of the unit at idx
        E toggles between excitatory and inhibitory neurons
        """
        # probably want to switch to some indexing scheme between the arrays
        raise NotImplementedError

    def update_async(self, input):
        """
        Asynchronously update the state of the network given presentation
         of stimulus input
        """
        # potentially override update_net unless there are other intervening steps
        # maybe run several times for each stimulus
        async_idx = np.random.permutation(self.n_E + self.n_I)
        for idx in async_idx:
            self.update_unit(input, idx, idx < self.n_E)

class BoltzmannMachine(DiscreteNet):
    """
    Boltzmann Machine
    """

    @overrides(DiscreteNet)
    def update_unit(input, idx, E):
        if E:
            pass
