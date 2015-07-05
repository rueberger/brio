"""
Auxiliary objects for blocks. Contents:
LayerType: Enum that holds type information for layers
NetworkParams: container
"""
from enum import Enum, unique

#to do: look into replacing this with protobuf

@unique
class LayerType(Enum):
    """
    This Enumerated class defines the basic types of layers
    The purpose of using an Enum object instead of simply passing the type as a string
      when intializing Layer is that Enum objects come with data attached and
      are more robust (to mispelling).
    This class reduces overall module complexity by allowing high level distinctions between
      the types of layers to not interfere with Layer subclasses, sidestepping the need to use
      factories for Layer subclasses or modify __metaclass__
    """
    # pylint: disable=too-few-public-methods

    unconstrained = (1, 1, False)
    excitatory = (1, 1, True)
    inhibitory = (2, -1, True)

    def __init__(self, firing_rate_multiplier, weight_multiplier, constrain_weights):
        self.firing_rate_multiplier = firing_rate_multiplier
        self.weight_multiplier = weight_multiplier
        self.constrain_weights = constrain_weights

class NetworkParams(object):
    """
    This class is a container for global network parameters
    It is used when initializing Network objects and parameters are propagated
      to layers and connections
    The use of a class as a parameter is somewhat experimental, but I think preferable
      to a long list of kwargs in Network
    """
    # pylint: disable=too-few-public-methods
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments

    def __init__(self, baseline_firing_rate=0.02, bias_learning_rate=0.1,
                 weight_learning_rate=0.028, presentations=50, async=False):
        self.presentations = presentations
        # sets number of iterations for characeteristic scale of exponential moving
        #   average
        self.char_itrs = 1. / 5
        self.baseline_firing_rate = baseline_firing_rate
        self.bias_learning_rate = bias_learning_rate
        self.weight_learning_rate = weight_learning_rate
        self.layer_history_length = 100
        self.async = async
        self.update_batch_size = 100
        self.display = False
