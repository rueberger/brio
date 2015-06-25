from blocks.connection import Connection
import unittest

class TestConnection(unittest.TestCase):
    def test_creation_with_constraint_equal_None_and_input_layer_equal_boltzmann_machine_layer_instance_and_output_layer_equal_boltzmann_machine_layer_instance(self):
        connection = Connection(BoltzmannMachineLayer(), BoltzmannMachineLayer(), None)
        # Make sure it doesn't raise any exceptions.

    def test_creation_with_constraint_equal_None_and_input_layer_equal_boltzmann_machine_layer_instance_and_output_layer_equal_boltzmann_machine_layer_instance_case_2(self):
        connection = Connection(BoltzmannMachineLayer(), BoltzmannMachineLayer(), None)
        # Make sure it doesn't raise any exceptions.

    def test_creation_with_constraint_equal_None_and_input_layer_equal_boltzmann_machine_layer_instance_and_output_layer_equal_boltzmann_machine_layer_instance_case_3(self):
        connection = Connection(BoltzmannMachineLayer(), BoltzmannMachineLayer(), None)
        # Make sure it doesn't raise any exceptions.

    def test_creation_with_constraint_equal_None_and_input_layer_equal_input_layer_instance_and_output_layer_equal_boltzmann_machine_layer_instance(self):
        connection = Connection(InputLayer(), BoltzmannMachineLayer(), None)
        # Make sure it doesn't raise any exceptions.

    def test_energy_shadow(self):
        # connection = Connection(input_layer, output_layer, constraint)
        # self.assertEqual(expected, connection.energy_shadow(input_idx))
        assert False # TODO: implement your test here

    def test_feedforward_energy(self):
        # connection = Connection(input_layer, output_layer, constraint)
        # self.assertEqual(expected, connection.feedforward_energy(idx))
        assert False # TODO: implement your test here

    def test_creation_with_constraint_equal_excitatory_and_input_layer_equal_input_layer_instance_and_output_layer_equal_perceptron_layer_instance(self):
        connection = Connection(InputLayer(), PerceptronLayer(), 'excitatory')
#Makesureitdoesn'traiseanyexceptions.

    def test___init__(self):
        # connection = Connection(input_layer, output_layer, learning_rate_multiplier)
        assert False # TODO: implement your test here

    def test_apply_weight_rule(self):
        # connection = Connection(input_layer, output_layer, learning_rate_multiplier)
        # self.assertEqual(expected, connection.apply_weight_rule())
        assert False # TODO: implement your test here

    def test_unpack_network_params(self):
        # connection = Connection(input_layer, output_layer, learning_rate_multiplier)
        # self.assertEqual(expected, connection.unpack_network_params(network))
        assert False # TODO: implement your test here

if __name__ == '__main__':
    unittest.main()
