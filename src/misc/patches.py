"""
Utils for generating and processing patches from images
"""
import numpy as np

def patch_generator(images, patch_size, n_patches=1000, crop=4, normalize=True):
    """ A generator that produces patches of the desired size from images

    :param images: an array of images. expected shape (img_height, img_width,  n_imgs)
    :param patch_size: the desired size of a patch size. patches are square
    :param n_patches: the number of patches for this generator to generate
    :param crop: number of pixels to crop away from the borders. used to avoid FFT
       induced distortions along the boundaries
    :returns: a patch generator
    :rtype: generator

    """
    height_max_draw = images.shape[0] - patch_size - crop * 2
    width_max_draw = images.shape[1] - patch_size - crop * 2
    n_imgs = images.shape[2]
    for _ in xrange(n_patches):
        hgt_idx = np.random.randint(height_max_draw)  + crop
        wdt_idx = np.random.randint(width_max_draw) + crop
        img_idx = np.random.randint(n_imgs)
        img = images[hgt_idx : hgt_idx + patch_size, wdt_idx : wdt_idx + patch_size, img_idx]
        if normalize:
            img -= np.mean(img)
            img *= 1. / np.std(img)
            # divide by five as prescribed in the EINet paper
            yield img / 5.
        else:
            yield img / 5.

def mean_zero_patch(images, patch_size, n_patches, crop=4):
    patches = np.array(list(patch_generator(images, patch_size, n_patches, crop, True)))
    pixel_mean = np.mean(patches, axis=0)
    for patch in patches:
        yield patch - pixel_mean