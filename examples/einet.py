"""
Examples usage of brio
Demonstrates training einet on whitened image patches from the van hateren natural image database
Original images available from the bethge lab: http://bethgelab.org/datasets/vanhateren/
Whitened images used here available from Bruno Olshausen at the
  redwood center here: https://redwood.berkeley.edu/bruno/sparsenet/
"""

from brio.blocks.factories import einet_factory, sailnet_factory
from brio.misc.patches import patch_generator, mean_zero_patch
from scipy.io import loadmat
from brio.blocks.aux import NetworkParams
from brio.misc.plotting import plot_receptive_fields, plot_param_distr, plot_concat_imgs
import matplotlib.pyplot as plt
plt.ion()

DATA_PATH = "../data/whitened_van_hateren.mat"
PATCH_SIZE = 17

images = loadmat(DATA_PATH)['IMAGES']


def run_example():
    """
    Creates an EI-net as described in Deweese and King 2013
    trains it on whitened image patches from the van hateren database
    demonstrates the learned receptive fields
    """
    params = NetworkParams(async=False, display=False)
    params.update_cap = 0.1
    params.steps_per_rc_time = 10
    params.steps_per_fr_time = 10
    params.bias_learning_rate = 0.2
    params.baseline_lrate = 0.02
    params.baseline_firing_rate = 0.02
    params.lfr_char_time = 5p
    einet = einet_factory([PATCH_SIZE ** 2, 400, 49], params)
    fig, ax = plt.subplots(figsize=(10,10))
    for _ in xrange(40):
        einet.train(patch_generator(images, PATCH_SIZE, 1000))
        oja_w = einet.layers[0].outputs[0].weights.T
        imgs = [w.reshape(PATCH_SIZE, PATCH_SIZE) for w in oja_w]
        plot_concat_imgs(imgs, axis=ax)
        plt.draw()
    return einet

def sailnet():
    params = NetworkParams(async=False, display=False)
    params.update_cap = 0.25
    params.steps_per_rc_time = 10
    params.steps_per_fr_time = 1
    params.bias_learning_rate = 0.1
    params.baseline_lrate = 0.1
    params.lfr_char_time = 5
    snet = sailnet_factory([PATCH_SIZE **2, 400], params)
    fig, ax = plt.subplots(figsize=(10,10))
    for _ in xrange(20):
        snet.train(patch_generator(images, PATCH_SIZE, 1000))
        oja_w = snet.layers[0].outputs[0].weights.T
        imgs = [w.reshape(PATCH_SIZE, PATCH_SIZE) for w in oja_w]
        plot_concat_imgs(imgs, axis=ax)
        plt.draw()
    return snet
