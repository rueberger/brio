"""
 This model contains various plotting utilities
"""
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.ion()

def plot_receptive_fields(net, layer_idx, unit_idx, stimulus_generator):
    """ Make a plot of the receptive field of network

    :param net: trained network to plot the receptive fields of
    :param layer_idx: idx of the layer you would like to plot. keys into network.layers
    :param unit_idx: idx of the receptive field you'd like to look at
    :param stimulus_generator: a generator object. calling next on this generator must return
          an array that can be flatted to the shape of the input layer
    :returns: None
    :rtype: None

    """
    receptive_field = net.compute_sta(stimulus_generator, layer_idx, 2000)[unit_idx]
    seamap = mpl.colors.ListedColormap(sns.cubehelix_palette(256, start=.5, rot=-.75))
    plt.imshow(receptive_field, cmap=seamap)

def concat_receptive_fields(net, layer_idx, stimulus_generator):
    """ concatenate the receptive fields of each neuron together into a block array
    so that they can all be plotted at once

    layer_size must be a perfect square for ease of plotting

    :param net: trained network to plot the receptive fields of
    :param layer_idx: idx of the layer you would like to plot. keys into network.layers
    :param stimulus_generator: a generator object. calling next on this generator must return
          an array that can be flatted to the shape of the input layer
    :returns: array containing all receptive fields
    :rtype: array
    """
    receptive_fields = net.compute_sta(stimulus_generator, layer_idx)
    img_size = len(stimulus_generator.next())
    layer_size = net.layers[layer_idx].n_dims
    assert int(np.sqrt(layer_size)) == np.sqrt(layer_size)
    assert int(np.sqrt(img_size)) == np.sqrt(img_size)
    layer_length = np.sqrt(layer_size)
    img_length = np.sqrt(img_size)
    border_thickness = 2
    border_color = 10
    concat_length = layer_length * img_length + (layer_length - 1) * border_thickness
    concat_rf = np.ones(concat_length, concat_length) * border_color
    for x_idx, y_idx in itertools.product(xrange(layer_length),
                                          xrange(layer_length)):
        # this keys into receptive_fields
        flat_idx = x_idx * layer_length + y_idx
        x_offset = border_thickness * x_idx
        y_offset = border_thickness * y_idx
        # not sure how to do a continuation line cleanly here
        concat_rf[x_idx * img_length + x_offset: (x_idx + 1) * img_length + x_offset,
                  y_idx * img_length + y_offset: (y_idx + 1) * img_length + y_offset] = receptive_fields[flat_idx]
    return concat_rf

    # now if I'm exceptionally lucky this will just work