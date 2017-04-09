#! /usr/bin/env python
#
# Program: visaulize_dstl
#
# Author: Nick Lee
#
# Description: Code to visualize data from DSTL kaggle competition
#
# Date: Feb 3, 2017
#

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from load_dstl import Load_DSTL
import seaborn as sns


# Set colors
class_colors = {
                1: 'red wine',  # buildings
                2: 'ruby',  # Small structures
                3: 'slate grey',  # good roads
                4: 'dark orange',  # Dirt roads
                5: 'pine',  # Trees
                6: 'ocre',  # Crops
                7: 'turquoise',  # Water
                8: 'blue',  # Standing water
                9: 'vivid purple',  # Large vehicle
                10: 'rich purple'  # Small vehicle
                }


# General display functions
def display_three_band(image, mask, true_mask=None, colors='black', true_colors='gray', title=None):
    sns.set_style('white')
    fig, ax, _ = tiff.imshow(255 * scale_percentile(image))
    if true_mask is not None:
        ax.contour(true_mask, colors=true_colors)
    ax.contour(mask, colors=colors)
    if title is not None:
        ax.set_title(title)
    plt.show()


def display_three_band_image(image_id='6120_2_2', overlay=True, classes=None, showplot=True):
    """ Display a given map """
    sns.set_style('white')
    dstl = Load_DSTL()
    image = dstl._get_image(image_id)
    fig, ax, _ = tiff.imshow(255 * scale_percentile(image))
    if overlay:
        # Find all masks
        image_shapes = dstl.training_set.loc[dstl.training_set['ImageId'] == image_id]
        for idx, row in image_shapes.iterrows():
            if (classes is None or row['ClassType'] in classes) and row['MultipolygonWKT'] != 'MULTIPOLYGON EMPTY':
                mask = dstl._create_mask(row)
                ax.contour(mask, colors=sns.xkcd_rgb[class_colors[row['ClassType']]])
    ax.set_title(image_id)
    if showplot:
        plt.show()
    return ax


def scale_percentile(image):
    """
    Fixes the pixel value range to 2%-98% original distribution of values

    As seen at:
    https://www.kaggle.com/chatcat/dstl-satellite-imagery-feature-detection/load-a-3-band-tif-image-and-overlay-dirt-tracks
    """
    orig_shape = image.shape
    image = np.reshape(image, [image.shape[0] * image.shape[1], 3]).astype(float)
    # Get 2nd and 98th percentile
    mins = np.percentile(image, 1, axis=0)
    maxs = np.percentile(image, 99, axis=0) - mins
    image = (image - mins[None, :]) / maxs[None, :]
    image = np.reshape(image, orig_shape)
    image = image.clip(0, 1)
    return image


############
# Run Code #
############
if __name__ == '__main__':
    pass
