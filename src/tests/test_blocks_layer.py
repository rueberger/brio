import numpy as np
import unittest
from blocks.layers import Layer, BoltzmanMachineLayer, PerceptronLayer, InputLayer

N_SEED = 1337
N_DIMS = 10

class TestLayer(unittest.TestCase):

    def test_add_input(self):
        # layer = Layer(N_DIMS, ltype)
        # self.assertEqual(expected, layer.add_input(input_connection))
        assert False # TODO: implement your test here

    def test_add_output(self):
        # layer = Layer(N_DIMS, ltype)
        # self.assertEqual(expected, layer.add_output(output_connection))
        assert False # TODO: implement your test here

    def test_apply_bias_rule(self):
        # layer = Layer(N_DIMS, ltype)
        # self.assertEqual(expected, layer.apply_bias_rule())
        assert False # TODO: implement your test here

    def test_firing_rates(self):
        # layer = Layer(N_DIMS, ltype)
        # self.assertEqual(expected, layer.firing_rates())
        assert False # TODO: implement your test here

    def test_input_energy(self):
        # layer = Layer(N_DIMS, ltype)
        # self.assertEqual(expected, layer.input_energy(idx))
        assert False # TODO: implement your test here

    def test_output_energy(self):
        # layer = Layer(N_DIMS, ltype)
        # self.assertEqual(expected, layer.output_energy(idx))
        assert False # TODO: implement your test here

    def test_unpack_network_params(self):
        # layer = Layer(N_DIMS, ltype)
        # self.assertEqual(expected, layer.unpack_network_params(network))
        assert False # TODO: implement your test here

    def test_update_history(self):
        # layer = Layer(N_DIMS, ltype)
        # self.assertEqual(expected, layer.update_history())
        assert False # TODO: implement your test here

    def test_update_state(self):
        # layer = Layer(N_DIMS, ltype)
        # self.assertEqual(expected, layer.update_state(idx))
        assert False # TODO: implement your test here

class TestBoltzmannMachineLayer(unittest.TestCase):

    def setUp(self):
        np.random.seed(N_SEED)
        self.test_layer = BoltzmanMachineLayer(N_DIMS)

    def test_activation(self):
        self.assertEqual(1, self.test_layer.activation(np.inf))

class TestPerceptronLayer(unittest.TestCase):

    def setUp(self):
        np.random.seed(N_SEED)
        self.test_layer = PerceptronLayer(N_DIMS)

    def test_activation(self):
        self.assertEqual(1, self.test_layer.activation(np.inf))

class TestInputLayer(unittest.TestCase):

    def setUp(self):
        np.random.seed(N_SEED)
        self.test_layer = InputLayer(N_DIMS)

    def test_set_state(self):
        test_state = np.ones(N_DIMS)
        self.assertEqual(self.test_layer.state, self.test_layer.set_state(test_state))
