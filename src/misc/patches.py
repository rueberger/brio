"""
Utils for generating and processing patches from images
"""
import numpy as np

def patch_generator(images, patch_size, n_patches=1000):
    """ A generator that produces patches of the desired size from images

    :param images: an array of images. expected shape (img_height, img_width,  n_imgs)
    :param patch_size: the desired size of a patch size. patches are square
    :param n_patches: the number of patches for this generator to generate
    :returns: a patch generator
    :rtype: generator

    """
    height_max_draw = images.shape[0] - patch_size
    width_max_draw = images.shape[1] - patch_size
    n_imgs = images.shape[2]
    for _ in xrange(n_patches):
        hgt_idx = np.random.randint(height_max_draw)
        wdt_idx = np.random.randint(width_max_draw)
        img_idx = np.random.randint(n_imgs)
        yield images[hgt_idx : hgt_idx + patch_size, wdt_idx : wdt_idx + patch_size, img_idx]
