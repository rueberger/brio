"""
 This model contains various plotting utilities
"""
import time
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from misc.sta import auto_sta
plt.ion()

SEAMAP = mpl.colors.ListedColormap(sns.cubehelix_palette(256, start=.5, rot=-.75))

def img_slideshow(imgs):
    """ plots a slideshow of the imgs

    :param imgs: list or array of images
    :returns: None
    :rtype: None
    """
    fig = plt.figure()
    for img in imgs:
        plt.clf()
        plt.imshow(img, cmap=SEAMAP)
        fig.canvas.draw()
        time.sleep(.1)

def hist_slideshow(arr):
    """ plots a histogram or density plot of the scalar distribution data

    :param arr: list of array of draws from scalar distribution
    :returns: None
    :rtype: None
    """
    fig = plt.figure()
    for distr in arr:
        plt.clf()
        plt.hist(distr, bins=250, normed=True)
        fig.canvas.draw()
        time.sleep(.1)

def plot_param_distr(net):
    """ plots histograms of the weight distributions in the different layers

    :param net: a network that is currently being trained
    :param update_interval: time interval in seconds for the plot to update
    :param n_updates: number of times to update the plot
    :returns: None
    :rtype: None

    """
    cons = list(net.connections)
    nrows = max(len(cons), len(net.layers[1:]))
    fig, ax_arr = plt.subplots(nrows=nrows, ncols=2)
    for con, axis in zip(cons, ax_arr[:, 0]):
        axis.hist(np.ravel(con.weights), bins=250, normed=True)
        axis.set_title("Weight distribution for {}".format(str(con)))
    for layer, axis in zip(net.layers[1:], ax_arr[:, 1]):
        axis.hist(np.ravel(layer.bias), bins=250, normed=True)
        axis.set_title("Bias distribution for {}".format(str(layer)))
    plt.show()



def plot_receptive_fields(net, layer_idx, slideshow=True, n_samples=1E5, stimulus_generator=None):
    """ Make a plot of the receptive field of network

    :param net: trained network to plot the receptive fields of
    :param layer_idx: idx of the layer you would like to plot. keys into network.layers
    :param unit_idx: idx of the receptive field you'd like to look at
    :param stimulus_generator: a generator object. calling next on this generator must return
          an array that can be flatted to the shape of the input layer.
          By default uniform random stimuli are generated for the relevant domain
    :returns: None
    :rtype: None
    """
    response_dict = auto_sta(net, n_samples, stimulus_generator)
    are_imgs = (response_dict.values()[0].ndim == 3)
    if are_imgs:
        imgs = [np.mean(response_dict[layer_idx, unit_idx], axis=0) for
                unit_idx in xrange(net.layers[layer_idx].n_dims)]
        if slideshow:
            img_slideshow(imgs)
        else:
            plot_concat_imgs(imgs)
    else:
        distrs = [response_dict[layer_idx, unit_idx] for unit_idx in xrange(net.layers[layer_idx].n_dims)]
        if slideshow:
            hist_slideshow(distrs)
        else:
            raise NotImplementedError("Have yet to implement multiple scalar distr")




def plot_concat_imgs(imgs, border_thickness=2):
    """ concatenate the imgs together into one big image separated by borders

    :param imgs: list or array of images. total number of images must be a perfect square and
       images must be square
    :returns: array containing all receptive fields
    :rtype: array

    """
    assert int(np.sqrt(len(imgs))) == np.sqrt(len(imgs))
    assert imgs[0].shape[0] == imgs[0].shape[1]
    img_length = imgs[0].shape[0]
    layer_length = int(np.sqrt(len(imgs)))
    concat_length = layer_length * img_length + (layer_length - 1) * border_thickness
    border_color = np.max(imgs)
    concat_rf = np.ones((concat_length, concat_length)) * border_color
    for x_idx, y_idx in itertools.product(xrange(layer_length),
                                          xrange(layer_length)):
        # this keys into imgs
        flat_idx = x_idx * layer_length + y_idx
        x_offset = border_thickness * x_idx
        y_offset = border_thickness * y_idx
        # not sure how to do a continuation line cleanly here
        concat_rf[x_idx * img_length + x_offset: (x_idx + 1) * img_length + x_offset,
                  y_idx * img_length + y_offset: (y_idx + 1) * img_length + y_offset] = imgs[flat_idx]
    plt.imshow(concat_rf, cmap=SEAMAP)
