"""
poe for checking that weights are properly set with constraints
"""
from blocks import layer, connection
INPUT_LAYER = layer.InputLayer(10)
OTHER_LAYER = layer.PerceptronLayer(20)
connection.Connection(INPUT_LAYER, OTHER_LAYER, constraint='excitatory')
