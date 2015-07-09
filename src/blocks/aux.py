"""
Auxiliary objects for blocks. Contents:
LayerType: Enum that holds type information for layers
NetworkParams: container
"""
from enum import Enum, unique
import numpy as np
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

    def __init__(self, baseline_firing_rate=0.02, bias_learning_rate=0.0001,
                 weight_learning_rate=0.005, presentations=50, async=False,
                 display=False):
        self.presentations = presentations
        self.stimuli_per_epoch = 100
        self.update_batch_size = presentations * self.stimuli_per_epoch
        # sets number of iterations for characeteristic scale of exponential moving
        #   average
        self.baseline_firing_rate = baseline_firing_rate
        self.bias_learning_rate = bias_learning_rate
        self.baseline_weight_lrate = weight_learning_rate
        # how many firing rates to keep in computing the average
        self.layer_history_length = self.update_batch_size
        self.keep_extra_history = True
        self.async = async
        self.display = display
        # the number of simulation steps corresponding to the characteristic time of the membrane
        #  rc constant
        # this is is less meaningful for non-LIF neurons
        self.steps_per_rc_time = 10
        self.steps_per_fr_time = 10
        # in number of epochs
        self.lfr_char_time = 1
        # for now the characteristic time for the ema history is the update batch size
        self.ema_lfr = 1 - np.exp(- 1. / self.lfr_char_time)
        self.ema_curr = 1 - np.exp(-1. / self.steps_per_fr_time)
