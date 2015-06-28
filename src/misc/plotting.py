"""
 This model contains various plotting utilities
"""
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_receptive_fields(network, layer_idx=0):
    """ Make a plot of the receptive field of network
    by default the receptive field of the input layer is shown

    :param network: network object
    :param layer_idx: idx of the layer you would like to plot. keys into network.layers
    :returns: None
    :rtype: None
    """
    seamap = mpl.colors.ListedColormap(sns.cubehelix_palette(256, start=.5, rot=-.75))
    plt.imshow(network.layers[0].outputs[0].weights[:, 0].reshape(11, 11), cmap=seamap)
