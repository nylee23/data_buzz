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


class Visualize_DSTL(Load_DSTL):
    """
    Class to hold methods for training detection of objects in DSTL images
    """
    def __init__(self):
        # Get tables from Load_DSTL
        super().__init__()

    def display_three_band(self, image_id='6120_2_2'):
        """ Display a given map """
        image = self._get_image(image_id)
        fig, ax, _ = tiff.imshow(255 * self._scale_percentile(image))
        masks = self._get_objects(image_id, ax=ax)
        plt.show()
        return image, masks

    def _get_objects(self, image_id, ax=None):
        """
        Find all the objects in a particular image in the training set
        """
        if ax is None:
            fig, ax = plt.subplots()
        # Loop over all classes
        objects = self.training_set.loc[self.training_set['ImageId'] == image_id]
        masks = []
        for idx, row in objects.iterrows():
            masks.append(self._create_mask(row, ax=ax))
        return masks

    def _scale_percentile(self, image):
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
    dstl_plots = Visualize_DSTL()
    img, masks = dstl_plots.display_three_band()
