"""
Examples usage of brio
Demonstrates training einet on whitened image patches from the van hateren natural image database
Original images available from the bethge lab: http://bethgelab.org/datasets/vanhateren/
Whitened images used here available from Bruno Olshausen at the
  redwood center here: https://redwood.berkeley.edu/bruno/sparsenet/
"""

from blocks.factories import einet_factory
from misc.patches import patch_generator
from scipy.io import loadmat
from blocks.aux import NetworkParams
from misc.plotting import plot_receptive_fields, plot_param_distr
import matplotlib.pyplot as plt
plt.ion()

DATA_PATH = "../data/whitened_van_hateren.mat"
PATCH_SIZE = 8

images = loadmat(DATA_PATH)['IMAGES']


def run_example():
    """
    Creates an EI-net as described in Deweese and King 2013
    trains it on whitened image patches from the van hateren database
    demonstrates the learned receptive fields
    """
    # input layer with 256 ndoes (same as image patches)
    # excitatory layer with 256 nodes
    # inhibitory layer with 49 nodes
    einet = einet_factory([PATCH_SIZE ** 2, 121, 36], NetworkParams(async=False, display=True))

    # this method iterates through the stimuli
    # presents each stimulus to the network a number of times (5 by default)
    #  and updates the state asynchronously
    # next, weights and biases are updated according to the learning rules defined in
    #  their class
    einet.train(patch_generator(images, PATCH_SIZE, 1000))

    plot_param_distr(einet)

    print "Now plotting receptive fields"
    plot_receptive_fields(einet, 1, slideshow=False, n_samples=1000,
                          stimulus_generator=patch_generator(images, PATCH_SIZE, int(1E5)))

    # that's it!
    return einet
