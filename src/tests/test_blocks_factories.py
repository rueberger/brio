import unittest
from blocks.factories import rbm_factory
from blocks.factories import einet_factory

class TestRbmFactory(unittest.TestCase):
    def test_rbm_factory(self):
        # self.assertEqual(expected, rbm_factory(layer_sizes))
        assert False # TODO: implement your test here

    def test_rbm_factory_returns_network_instance_for_list(self):
        self.assertEqual(Network([InputLayer(), BoltzmannMachineLayer(), BoltzmannMachineLayer(), BoltzmannMachineLayer(), BoltzmannMachineLayer()], 5), rbm_factory([5, 10, 50, 10, 5]))

class TestMlpFactory(unittest.TestCase):
    def test_mlp_factory(self):
        # self.assertEqual(expected, mlp_factory(layer_sizes))
        assert False # TODO: implement your test here

class TestEinetFactory(unittest.TestCase):
    def test_einet_factory(self):
        # self.assertEqual(expected, einet_factory(layer_sizes))
        assert False # TODO: implement your test here

    def test_einet_factory_raises_key_error_for_list(self):
        self.assertRaises(KeyError, lambda: einet_factory([30, 100, 20]))

    def test_einet_factory_returns_network_instance_for_list(self):
        self.assertEqual(Network([InputLayer(), BoltzmannMachineLayer(), BoltzmannMachineLayer()], 5), einet_factory([30, 100, 20]))

if __name__ == '__main__':
    unittest.main()
