"""
unit test for blocks.layer
"""
import numpy as np
import unittest
from blocks.layer import Layer, BoltzmannMachineLayer, PerceptronLayer, InputLayer

N_SEED = 1337
N_DIMS = 10

# pylint: disable=missing-docstring

class TestLayer(unittest.TestCase):

    def setUp(self):
        np.random.seed(N_SEED)
        self.test_layer = Layer(N_DIMS)

    def test_update_history(self):
        self.test_layer.max_history_length = 5
        self.test_layer.state = 5 * np.ones(N_DIMS)
        self.test_layer.update_history()
        self.test_layer.state = np.ones(N_DIMS)
        self.test_layer.update_history()
        self.assertEqual(self.test_layer.history[-1], np.ones(N_DIMS))
        self.test_layer.state = 2 * np.ones(N_DIMS)
        for _ in xrange(5):
            self.test_layer.update_history()
        self.test_layer.state = 6 * np.ones(N_DIMS)
        for _ in xrange(5):
            self.test_layer.update_history()
        self.assertEqual(self.test_layer.history[-1], 6 * np.ones(N_DIMS))

class TestBoltzmannMachineLayer(unittest.TestCase):

    def setUp(self):
        np.random.seed(N_SEED)
        self.test_layer = BoltzmannMachineLayer(N_DIMS)

    def test_activation(self):
        # test a particular value
        self.assertEqual(1, self.test_layer.async_activation(np.inf))
        # test validity of returned values
        allowed_updates = {1, -1}
        for energy in 5 * np.random.randn(250):
            self.assertTrue(self.test_layer.async_activation(energy) in allowed_updates)



class TestPerceptronLayer(unittest.TestCase):

    def setUp(self):
        np.random.seed(N_SEED)
        self.test_layer = PerceptronLayer(N_DIMS)

    def test_activation(self):
        # test a particular value
        self.assertEqual(1, self.test_layer.activation(np.inf))
        # test validity of returned values
        allowed_updates = {1, -1}
        for energy in 5 * np.random.randn(250):
            self.assertTrue(self.test_layer.activation(energy) in allowed_updates)



class TestInputLayer(unittest.TestCase):

    def setUp(self):
        np.random.seed(N_SEED)
        self.test_layer = InputLayer(N_DIMS)

    def test_set_state(self):
        test_state = np.ones(N_DIMS)
        self.assertEqual(self.test_layer.state, self.test_layer.set_state(test_state))
