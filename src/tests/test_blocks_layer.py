import unittest

class TestLayer(unittest.TestCase):
    def test___init__(self):
        # layer = Layer(n_dims, ltype, update_bias, allow_self_con)
        assert False # TODO: implement your test here

    def test___repr__(self):
        # layer = Layer(n_dims, ltype, update_bias, allow_self_con)
        # self.assertEqual(expected, layer.__repr__())
        assert False # TODO: implement your test here

    def test_async_activation(self):
        # layer = Layer(n_dims, ltype, update_bias, allow_self_con)
        # self.assertEqual(expected, layer.async_activation(energy))
        assert False # TODO: implement your test here

    def test_async_update(self):
        # layer = Layer(n_dims, ltype, update_bias, allow_self_con)
        # self.assertEqual(expected, layer.async_update(idx))
        assert False # TODO: implement your test here

    def test_bias_update(self):
        # layer = Layer(n_dims, ltype, update_bias, allow_self_con)
        # self.assertEqual(expected, layer.bias_update())
        assert False # TODO: implement your test here

    def test_input_energy(self):
        # layer = Layer(n_dims, ltype, update_bias, allow_self_con)
        # self.assertEqual(expected, layer.input_energy(idx))
        assert False # TODO: implement your test here

    def test_output_energy(self):
        # layer = Layer(n_dims, ltype, update_bias, allow_self_con)
        # self.assertEqual(expected, layer.output_energy(idx))
        assert False # TODO: implement your test here

    def test_reset(self):
        # layer = Layer(n_dims, ltype, update_bias, allow_self_con)
        # self.assertEqual(expected, layer.reset())
        assert False # TODO: implement your test here

    def test_sync_update(self):
        # layer = Layer(n_dims, ltype, update_bias, allow_self_con)
        # self.assertEqual(expected, layer.sync_update())
        assert False # TODO: implement your test here

    def test_unpack_network_params(self):
        # layer = Layer(n_dims, ltype, update_bias, allow_self_con)
        # self.assertEqual(expected, layer.unpack_network_params(network))
        assert False # TODO: implement your test here

    def test_update_firing_rates(self):
        # layer = Layer(n_dims, ltype, update_bias, allow_self_con)
        # self.assertEqual(expected, layer.update_firing_rates())
        assert False # TODO: implement your test here

    def test_update_history(self):
        # layer = Layer(n_dims, ltype, update_bias, allow_self_con)
        # self.assertEqual(expected, layer.update_history())
        assert False # TODO: implement your test here

    def test_update_lifetime_mean(self):
        # layer = Layer(n_dims, ltype, update_bias, allow_self_con)
        # self.assertEqual(expected, layer.update_lifetime_mean())
        assert False # TODO: implement your test here

class TestLIFLayer(unittest.TestCase):
    def test___init__(self):
        # l_if_layer = LIFLayer(*args, **kwargs)
        assert False # TODO: implement your test here

    def test_reset(self):
        # l_if_layer = LIFLayer(*args, **kwargs)
        # self.assertEqual(expected, l_if_layer.reset())
        assert False # TODO: implement your test here

    def test_sync_update(self):
        # l_if_layer = LIFLayer(*args, **kwargs)
        # self.assertEqual(expected, l_if_layer.sync_update())
        assert False # TODO: implement your test here

class TestBoltzmannMachineLayer(unittest.TestCase):
    def test_async_activation(self):
        # boltzmann_machine_layer = BoltzmannMachineLayer()
        # self.assertEqual(expected, boltzmann_machine_layer.async_activation(energy))
        assert False # TODO: implement your test here

    def test_async_update(self):
        # boltzmann_machine_layer = BoltzmannMachineLayer()
        # self.assertEqual(expected, boltzmann_machine_layer.async_update(idx))
        assert False # TODO: implement your test here

    def test_sync_update(self):
        # boltzmann_machine_layer = BoltzmannMachineLayer()
        # self.assertEqual(expected, boltzmann_machine_layer.sync_update())
        assert False # TODO: implement your test here

class TestPerceptronLayer(unittest.TestCase):
    def test_async_activation(self):
        # perceptron_layer = PerceptronLayer()
        # self.assertEqual(expected, perceptron_layer.async_activation(energy))
        assert False # TODO: implement your test here

    def test_async_update(self):
        # perceptron_layer = PerceptronLayer()
        # self.assertEqual(expected, perceptron_layer.async_update(idx))
        assert False # TODO: implement your test here

    def test_sync_update(self):
        # perceptron_layer = PerceptronLayer()
        # self.assertEqual(expected, perceptron_layer.sync_update())
        assert False # TODO: implement your test here

class TestInputLayer(unittest.TestCase):
    def test_set_state(self):
        # input_layer = InputLayer()
        # self.assertEqual(expected, input_layer.set_state(state))
        assert False # TODO: implement your test here

class TestRasterInputLayer(unittest.TestCase):
    def test___init__(self):
        # raster_input_layer = RasterInputLayer(n_dims, min_range, max_range, **kwargs)
        assert False # TODO: implement your test here

    def test_avg_activation(self):
        # raster_input_layer = RasterInputLayer(n_dims, min_range, max_range, **kwargs)
        # self.assertEqual(expected, raster_input_layer.avg_activation(scalar_value))
        assert False # TODO: implement your test here

    def test_rate_at_points(self):
        # raster_input_layer = RasterInputLayer(n_dims, min_range, max_range, **kwargs)
        # self.assertEqual(expected, raster_input_layer.rate_at_points(scalar_value))
        assert False # TODO: implement your test here

    def test_set_state(self):
        # raster_input_layer = RasterInputLayer(n_dims, min_range, max_range, **kwargs)
        # self.assertEqual(expected, raster_input_layer.set_state(scalar_value))
        assert False # TODO: implement your test here

if __name__ == '__main__':
    unittest.main()
