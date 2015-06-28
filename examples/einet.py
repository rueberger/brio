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

DATA_PATH = "../data/whitened_van_hateren.mat"
PATCH_SIZE = 16

# creates an EI-net as described in Deweese and King 2013
# input layer with 256 ndoes (same as image patches)
# excitatory layer with 256 nodes
# inhibitory layer with 49 nodes
einet = einet_factory([PATCH_SIZE ** 2, 256, 49])

# trains the network
for patch in patch_generator:
    einet.run_network(patch)

# that's it!