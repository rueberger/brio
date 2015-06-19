# interface for network classes
# shared exposed methods: run network
# shared hidden methods: update state (asynchronous for now, which rules out the LIF neurons used in King et al - a major question to be answered is why they chose that scheme and if it will,
# work with boltzmann machines)
# shared variables: weights (input to E, E to I, I to I, I to E),
# note: following yimengs paper directly, won't need to compute input weights - set by infomax tuning curves with hypercolumns - I worry constrained connectivity will cause problems though


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