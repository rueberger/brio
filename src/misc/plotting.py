"""
 This model contains various plotting utilities
"""
import time
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import string
from misc.sta import auto_sta, factor
from blocks.layer import LIFLayer
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
        plt.imshow(img, cmap=SEAMAP, interpolation='none')
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
        if len(distr) != 0:
            plt.clf()
            plt.hist(distr, bins=250, normed=True)
            fig.canvas.draw()
            time.sleep(.1)


class ParamPlot(object):
    """
    This class provides a plot to visualize network parameters that can be updated on the fly
    """

    #pylint: disable=too-few-public-methods

    def __init__(self, net, layers=None, show_all=False, shape=None):
        """ Initialize this class

        :param net: network is a ycurrently in training network
        :param layers: layers to display. list of indices. by default all
        :param show_all: Show all parameters. Default of False only shows weight distributions
        :returns: the initialized ParamPlot object
        :rtype: ParamPlot

        """
        if layers is not None:
            self.layers = [net.layers[idx] for idx in layers]
        else:
            self.layers = net.layers[1:]
        self.net = net
        self.cons = net.connections.values()
        self.show_all = show_all
        if shape is None:
            if show_all:
                nrows = max(len(self.cons), len(self.layers) * 2)
                ncols=3
            else:
                # need something else: works terrible for primes
                nrows, ncols = factor(len(self.cons))
        else:
            nrows, ncols = shape
        self.fig, self.ax_arr = plt.subplots(nrows=nrows,
                                             ncols=ncols, figsize=(16, 10))
        self.t = np.arange(self.net.params.presentations)

    def update_plot(self):
        """ updates the plot without creating a new figure

        :returns: None
        :rtype: None
        """
        sns.set_style("whitegrid")
        self.fig.suptitle("Parameter distributions at timestep {}".format(self.net.t_counter))
        for axis in np.ravel(self.ax_arr):
            axis.clear()

        if self.show_all:
            for con, axis in zip(self.cons, self.ax_arr[:, 0]):
                axis.hist(np.ravel(con.weights), bins=250, normed=True)
                axis.set_title("Weight distribution for {}".format(str(con)))

            for layer, axis in zip(self.layers, self.ax_arr[:, 1]):
                axis.hist(np.ravel(layer.bias), bins=20, normed=True)
                axis.set_title("Bias distribution for {}".format(str(layer)))

            for layer, axis in zip(self.layers, self.ax_arr[len(self.layers):, 1]):
                axis.hist(np.ravel(layer.lfr_mean), bins=20, normed=True)
                axis.set_title("Firing rate distribution for {}".format(str(layer)))

            for layer, axis in zip(self.layers, self.ax_arr[len(self.layers):, 2]):
                if isinstance(layer, LIFLayer):
                    potentials = np.array(layer.pot_history)[:, :, -1].T
                    for u_t in potentials:
                        axis.plot(self.t, u_t)
                    axis.set_title("Potential history for one stimulus {}".format(str(layer)))
        else:
            for con, axis in zip(self.cons, self.ax_arr.ravel()):
                axis.hist(np.ravel(con.weights), bins=250, normed=True)
                axis.set_title("Weight distribution for {}".format(str(con)), fontsize=6)
        self.fig.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.draw()




def plot_param_distr(net):
    """ plots histograms of the weight distributions in the different layers

    :param net: a network that is currently being trained
    :param update_interval: time interval in seconds for the plot to update
    :param n_updates: number of times to update the plot
    :returns: None
    :rtype: None

    """
    cons = list(net.connections)
    nrows = max(len(cons), len(net.layers[1:]) * 2)
    fig, ax_arr = plt.subplots(nrows=nrows, ncols=2, figsize=(16, 10))
    for con, axis in zip(cons, ax_arr[:, 0]):
        axis.hist(np.ravel(con.weights), bins=250, normed=True)
        axis.set_title("Weight distribution for {}".format(str(con)))
    for layer, axis in zip(net.layers[1:], ax_arr[:, 1]):
        axis.hist(np.ravel(layer.bias), bins=250, normed=True)
        axis.set_title("Bias distribution for {}".format(str(layer)))
    for layer, axis in zip(net.layers[1:], ax_arr[len(net.layers[1:]):, 1]):
        axis.hist(np.ravel(layer.firing_rates), bins=250, normed=True)
        axis.set_title("Firing rate distibution for {}".format(str(layer)))
    fig.subplots_adjust(hspace=0.4)
    plt.draw()



def plot_receptive_fields(net, layer_idx, slideshow=True,
                          n_samples=1E5, stimulus_generator=None, stereo=False):
    """ Make a plot of the receptive field of network

    :param net: trained network to plot the receptive fields of
    :param layer_idx: idx of the layer you would like to plot. keys into network.layers
    :param slideshow: if True show receptive fields one at at time.
       Otherwise show them all at the same time
    :param n_samples: number of samples to compute STAs wtih
    :param stimulus_generator: a generator object. calling next on this generator must return
          an array that can be flatted to the shape of the input layer.
          By default uniform random stimuli are generated for the relevant domain
    :param stereo: if True split stereo images into two parts
    :returns: None
    :rtype: None

    """
    assert isinstance(layer_idx, int)
    response_dict, stimuli = auto_sta(net, n_samples, stimulus_generator, layer_idx=[layer_idx])
    are_imgs = (stimuli.ndim == 3)
    if are_imgs:
        imgs = np.zeros((net.layers[layer_idx].n_dims, stimuli.shape[1], stimuli.shape[2]))
        for unit_idx in xrange(net.layers[layer_idx].n_dims):
            response_idx = response_dict[(layer_idx, unit_idx)]
            imgs[unit_idx] = np.mean(stimuli[response_idx], axis=0)
        if slideshow:
            img_slideshow(imgs)
        else:
            if stereo:
                assert imgs.shape[1] == 2 * imgs.shape[2]
                side_len = imgs.shape[2]
                l_imgs = imgs[:, :side_len, :]
                r_imgs = imgs[:, side_len:, :]
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 14))
                plot_concat_imgs(l_imgs, axis=axes[0])
                plot_concat_imgs(r_imgs, axis=axes[1])
            else:
                plot_concat_imgs(imgs)
    else:
        distrs = [response_dict[layer_idx, unit_idx] for unit_idx in xrange(net.layers[layer_idx].n_dims)]
        if slideshow:
            hist_slideshow(distrs)
        else:
            raise NotImplementedError("Have yet to implement multiple scalar distr")




def plot_concat_imgs(imgs, border_thickness=2, axis=None, normalize=False):
    """ concatenate the imgs together into one big image separated by borders

    :param imgs: list or array of images. total number of images must be a perfect square and
       images must be square
    :param border_thickness: how many pixels of border between
    :param axis: optional matplotlib axis object to plot on
    :returns: array containing all receptive fields
    :rtype: array
    """
    sns.set_style('dark')
    assert isinstance(border_thickness, int)
    assert int(np.sqrt(len(imgs))) == np.sqrt(len(imgs))
    assert imgs[0].shape[0] == imgs[0].shape[1]
    if normalize:
        imgs = np.array(imgs)
        imgs /= np.sum(imgs ** 2, axis=(1,2)).reshape(-1, 1, 1)
    img_length = imgs[0].shape[0]
    layer_length = int(np.sqrt(len(imgs)))
    concat_length = layer_length * img_length + (layer_length - 1) * border_thickness
    border_color = np.nan
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
    if axis is not None:
        axis.imshow(concat_rf, interpolation='none', aspect='auto')
    else:
        plt.imshow(concat_rf, interpolation='none', aspect='auto')


def write_hist_to_stdout(data, n_bins=25, lines=10):
    """ Write a text based histogram to stdout

    :param data: the data to visualize
    :param bins: number of bins
    :param lines: how many vertical lines for the histogram to span
    :returns: None
    :rtype: None
    """
    pdf, bins = np.histogram(data, bins=n_bins, density=True)
    scaled_pdf = pdf * lines
    col_width = 4
    x_axis_width = col_width * n_bins
    y_label_width = 5
    # might want to use a string writer?
    for height in range(lines)[::-1]:
        line = string.join(['*** ' if d > height else '    ' for d in scaled_pdf], '')
        print "{:3.1f} | {}".format(height / float(lines), line)
    axis = '-' * n_bins * col_width
    mid_sep = (x_axis_width - 5) / 2
    x_labels = ['{:5.2f}'.format(bins[0]),
                ' ' * mid_sep,
                '{:5.2f}'.format(bins[n_bins / 2]),
                ' ' * mid_sep,
                '{:5.2f}'.format(bins[-1])
            ]
    print "    {}".format(axis)
    print string.join(x_labels, '')





def visualize_inhibition(einet, unit_idx=0,  n_show=9):
    """Shows the receptive field of the excitatory cells that the inhibitory cell
    at unit_idx inhibits the most

    :param einet: must be an einet
    :param unit_idx: unit_idx of inhibitory layer
    :returns: None
    :rtype: None
    """
    from misc.sta import factor
    inhib_weights = einet.layers[1].outputs[0].weights.T
    oja_weights = einet.layers[0].outputs[0].weights.T
    e_idx = np.argsort(inhib_weights[unit_idx])[::-1][:n_show]

    img_dims = factor(einet.layers[0].n_dims)
    imgs = [w.reshape(*img_dims) for w in oja_weights[e_idx]]
    print inhib_weights[unit_idx][e_idx]
    plot_concat_imgs(imgs)
