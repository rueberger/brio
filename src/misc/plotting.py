"""
 This model contains various plotting utilities
"""
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.ion()

def plot_receptive_fields(weights):
    """ Make a plot of the receptive field of network
    by default the receptive field of the input layer is shown

    :param network: network object
    :param layer_idx: idx of the layer you would like to plot. keys into network.layers
    :returns: None
    :rtype: None
    """
    seamap = mpl.colors.ListedColormap(sns.cubehelix_palette(256, start=.5, rot=-.75))
    plt.imshow(weights, cmap=seamap)
