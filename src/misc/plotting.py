"""
 This model contains various plotting utilities
"""
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.ion()

def plot_receptive_fields(net, layer_idx, unit_idx, stimulus_generator):
    """ Make a plot of the receptive field of network

    :param net: trained network to plot the receptive fields of
    # my doc generator got this from the other method!!! win! (it is called sphinx)
    :param layer_idx: idx of the layer you would like to plot. keys into network.layers
    :param unit_idx: idx of the receptive field you'd like to look at
    # lets see if it nabs the description for this one if I rename it. awww
    :param stimulus_generator: a generator object. calling next on this generator must return
          an array that can be flatted to the shape of the input layer
    :returns: None
    :rtype: None

    """
    receptive_field = net.compute_sta(stimulus_generator, layer_idx)[unit_idx]
    seamap = mpl.colors.ListedColormap(sns.cubehelix_palette(256, start=.5, rot=-.75))
    plt.imshow(receptive_field, cmap=seamap)
