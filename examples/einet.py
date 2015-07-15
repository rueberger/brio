"""
Examples usage of brio
Demonstrates training einet on whitened image patches from the van hateren natural image database
Original images available from the bethge lab: http://bethgelab.org/datasets/vanhateren/
Whitened images used here available from Bruno Olshausen at the
  redwood center here: https://redwood.berkeley.edu/bruno/sparsenet/
"""

from blocks.factories import einet_factory, sailnet_factory, perceptron_factory
from misc.patches import patch_generator, mean_zero_patch
from scipy.io import loadmat
from blocks.aux import NetworkParams
from misc.plotting import plot_receptive_fields, plot_param_distr, plot_concat_imgs
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
    einet = einet_factory([PATCH_SIZE ** 2, 256, 36], NetworkParams(async=False, display=False))
    fig, ax = plt.subplots(figsize=(10,10))
    for _ in xrange(40):
        einet.train(patch_generator(images, PATCH_SIZE, 250))
        oja_w = einet.layers[0].outputs[0].weights.T
        imgs = [w.reshape(PATCH_SIZE, PATCH_SIZE) for w in oja_w]
        plot_concat_imgs(imgs, ax=ax)
        plt.draw()
    return einet

def sailnet():
    snet = sailnet_factory([PATCH_SIZE **2, 256], NetworkParams(async=False, display=True))
    snet.train(patch_generator(images, PATCH_SIZE, 10000))
    return snet

def oja():
    onet = perceptron_factory([PATCH_SIZE **2, 64], NetworkParams(async=False, display=False))
    onet.train(mean_zero_patch(images, PATCH_SIZE, 5000))
    plot_receptive_fields(onet, 1, slideshow=False, n_samples=1000,
                          stimulus_generator=mean_zero_patch(images, PATCH_SIZE, int(1E5)))

    return onet
