# interface for network classes
# shared exposed methods: run network
# shared hidden methods: update state (asynchronous for now, which rules out the LIF neurons used in King et al - a major question to be answered is why they chose that scheme and if it will,
# work with boltzmann machines)
# shared variables: weights (input to E, E to I, I to I, I to E),
# to do: implement inhibitory pooling in the input


import numpy as np

class Network(object):
    """
    Generic interface class for networks that will be trained with the correlation measuring rule
    Implementing networks will share the same topology
    The difference will lie in the neuron representation
    Intended subclasses: McCullochPitts and Boltzmann at least, possibly LIF
    """

    def __init__(self, n_input, n_E, n_I):
        self.n_in = n_input
        self.n_i = n_I
        self.n_e = n_E
        self.w_inp_e = np.random.randn(n_input, n_E)
        self.w_e_i = np.random.randn(n_E, n_I)
        self.w_i_i = np.random.randn(n_I, n_I)
        self.w_i_e = np.random.randn(n_I, n_E)
        self.theta_e = np.random.randn(n_E)
        self.theta_i = np.random.randn(n_I)

    def update_network(self, in):
        """
        Update all nodes according to the relevant update rule
        in: shape (n_input, )
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
        self.y_e = np.ones(self.n_e)
        self.y_i = np.ones(self.n_i)

    def unit_energy(self, input, idx, excitatory):
        """
        Returns the energy of the unit at idx
        E toggles excited or inhibitory
        """
        if excitatory:
            e_input = np.sum(self.w_inp_e[:, idx] * input)
            e_inhib = -np.sum(self.w_i_e[:, idx] * self.y_i)

            # sign error here?
            e_bias = self.theta_e[idx] * self.y_e[idx]
            return e_inhib + e_input + e_bias
        else:
            e_excit = np.sum(self.w_e_i[:, idx] * self.y_e[idx])
            e_inhib = np.sum(self.w_i_i[:, idx] * self.y_i[idx])
            e_bias = np.sum(self.theta_i[idx] * self.y_i[idx])
            return e_inhib + e_excit + e_bias


    def update_unit(self, input, idx, excitatory):
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
        async_idx = np.random.permutation(self.n_e + self.n_i)
        for idx in async_idx:
            self.update_unit(input, idx, idx < self.n_e)

class BoltzmannMachine(DiscreteNet):
    """
    Boltzmann Machine
    """

    @overrides(DiscreteNet)
    def update_unit(input, idx, E):
        if E:
            pass
