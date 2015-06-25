import unittest

class TestLayer(unittest.TestCase):
    def test___init__(self):
        # layer = Layer(n_dims, ltype)
        assert False # TODO: implement your test here

    def test_activation(self):
        # layer = Layer(n_dims, ltype)
        # self.assertEqual(expected, layer.activation(energy))
        assert False # TODO: implement your test here

    def test_add_input(self):
        # layer = Layer(n_dims, ltype)
        # self.assertEqual(expected, layer.add_input(input_connection))
        assert False # TODO: implement your test here

    def test_add_output(self):
        # layer = Layer(n_dims, ltype)
        # self.assertEqual(expected, layer.add_output(output_connection))
        assert False # TODO: implement your test here

    def test_apply_bias_rule(self):
        # layer = Layer(n_dims, ltype)
        # self.assertEqual(expected, layer.apply_bias_rule())
        assert False # TODO: implement your test here

    def test_firing_rates(self):
        # layer = Layer(n_dims, ltype)
        # self.assertEqual(expected, layer.firing_rates())
        assert False # TODO: implement your test here

    def test_input_energy(self):
        # layer = Layer(n_dims, ltype)
        # self.assertEqual(expected, layer.input_energy(idx))
        assert False # TODO: implement your test here

    def test_output_energy(self):
        # layer = Layer(n_dims, ltype)
        # self.assertEqual(expected, layer.output_energy(idx))
        assert False # TODO: implement your test here

    def test_unpack_network_params(self):
        # layer = Layer(n_dims, ltype)
        # self.assertEqual(expected, layer.unpack_network_params(network))
        assert False # TODO: implement your test here

    def test_update_history(self):
        # layer = Layer(n_dims, ltype)
        # self.assertEqual(expected, layer.update_history())
        assert False # TODO: implement your test here

    def test_update_state(self):
        # layer = Layer(n_dims, ltype)
        # self.assertEqual(expected, layer.update_state(idx))
        assert False # TODO: implement your test here

class TestBoltzmannMachineLayer(unittest.TestCase):
    def test_activation(self):
        # boltzmann_machine_layer = BoltzmannMachineLayer()
        # self.assertEqual(expected, boltzmann_machine_layer.activation(energy))
        assert False # TODO: implement your test here

    def test_update_state(self):
        # boltzmann_machine_layer = BoltzmannMachineLayer()
        # self.assertEqual(expected, boltzmann_machine_layer.update_state(idx))
        assert False # TODO: implement your test here

class TestPerceptronLayer(unittest.TestCase):
    def test_activation(self):
        # perceptron_layer = PerceptronLayer()
        # self.assertEqual(expected, perceptron_layer.activation(energy))
        assert False # TODO: implement your test here

    def test_update_state(self):
        # perceptron_layer = PerceptronLayer()
        # self.assertEqual(expected, perceptron_layer.update_state(idx))
        assert False # TODO: implement your test here

class TestInputLayer(unittest.TestCase):
    def test_set_state(self):
        # input_layer = InputLayer()
        # self.assertEqual(expected, input_layer.set_state(state))
        assert False # TODO: implement your test here

if __name__ == '__main__':
    unittest.main()
