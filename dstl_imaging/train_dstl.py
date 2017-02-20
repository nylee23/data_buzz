#! /usr/bin/env python
#
# Program: train_dstl
#
# Author: Nick Lee
#
# Description: Code to extract and train the training set.
#
# Date: Feb 3, 2017
#

# Import Libraries
import numpy as np
import pandas as pd
import tifffile as tiff
from shapely.wkt import loads
from PIL import Image, ImageDraw


class Train_DSTL(object):
    """
    Class to hold methods for training detection of objects in DSTL images
    """
    def __init__(self):
        # Save useful tables
        self.training_set = pd.read_csv('train_wkt_v4.csv')
        self.grid_sizes = pd.read_csv('grid_sizes.csv')
        self.grid_sizes.columns = pd.Index(['ImageID', 'Xmax', 'Ymin'])
        self.object_colors = {1: 'gray', 2: 'blue', 3: 'black', 4: 'brown', 5: 'green', 6: 'yellow', 7: 'turquoise', 8: 'blue', 9: 'red', 10: 'orange'}

    def split_training_set(self, object_class=1):
        """
        Create a training, CV, and test set using a 60/20/20 split

        Since most of these are skewed classes, we will pay some special attention to make sure that positive class is properly represented in each category.
        """
        np.random.seed = 0
        X, Y = self.get_training_set(object_class=object_class)
        all_data = np.hstack((X, Y.reshape(-1, 1)))  # One array to make sure X and Y pairs aren't separated when permuting
        pos_train, pos_cv, pos_test = self._split_data(all_data[Y == 1], shuffle=True)
        neg_train, neg_cv, neg_test = self._split_data(all_data[Y == 0], shuffle=True)
        train = np.vstack((pos_train, neg_train))
        cv = np.vstack((pos_cv, neg_cv))
        test = np.vstack((pos_test, neg_test))
        return train, cv, test

    def get_training_set(self, object_class=1):
        """
        Create a training set for a given object class based on individual pixels
        Returns:
            training_set - A (m x n) sized numpy array corresponding to the pixel values in each band for the training examples.
            training_answers - A (m x 1) sized numpy array corresponding to the answer (1 = classified as part of class, 0 = not)
        """
        object_df = self.training_set.query('ClassType == {:d}'.format(object_class))
        train_examples = []
        train_answers = []
        for idx, row in object_df.iterrows():
            image = self._get_image(row['ImageId'])
            mask = self._create_mask(row)
            train_examples.append(image.reshape(-1, 3))
            train_answers.append(mask.ravel().astype('b'))
        training_set = np.concatenate((train_examples))
        training_answers = np.concatenate((train_answers))
        return training_set, training_answers

    def _create_mask(self, row, ax=None):
        """
        Given a row in the training set dataframe, return the mask corresponding to the given Multi-Polygon.
        """
        img_size, xy_limits, _ = self._get_image_properties(row['ImageId'])
        class_type = row['ClassType']
        if row['MultipolygonWKT'] != 'MULTIPOLYGON EMPTY':
            # Translate wkt to Shapely Polygon
            shapes = loads(row['MultipolygonWKT'])
            # Make a True/False image using PIL
            img = Image.new('1', img_size[-1::-1], 0)
            # Fill in the Image with the polygons
            for shape in shapes:
                xy_perimeter = np.array(shape.exterior.coords)
                xy_scaled = self._convert_xy_to_img_scale(xy_perimeter, img_size, xy_limits)
                poly = [(coord[0], coord[1]) for coord in xy_scaled]
                ImageDraw.Draw(img).polygon(poly, fill=1)
            mask = np.array(img)
            # Plot mask
            if ax is not None:
                ax.contour(mask, colors=self.object_colors[class_type])
        else:
            mask = np.zeros(img_size, dtype='bool')  # Make mask with no TRUE
        return mask

    # Helper Methods
    def _get_image(self, image_id):
        img_filename = 'three_band/{:}.tif'.format(image_id)
        image = tiff.imread(img_filename).transpose([1, 2, 0])
        return image

    def _get_image_properties(self, image_id='6120_2_2'):
        """ Return the img_size and xy_limits of a given image """
        image = self._get_image(image_id)
        img_size = image.shape[0:2]
        xy_limits = self.grid_sizes.loc[self.grid_sizes['ImageID'] == image_id, ['Xmax', 'Ymin']].values.reshape(-1)
        return img_size, xy_limits, image

    def _convert_xy_to_img_scale(self, xy, img_size, xy_limits):
        """
        Convert a list of xy tuples that are between [0, 1] and [-1, 0] and convert to the corresponding coordinates of an image

        Follows transformation provided here:
        https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/details/data-processing-tutorial

        Args:
            xy - list of (x, y) tuples where x & y are normalized
            img_size - Tuple of image (height, width)
            xy_limits - Tuple of Xmax and Ymin needed to properly scale image
        Returns:
            x_scaled, y_scaled - x and y coordinates scaled to the image
        """
        # Extract arrays of x and y
        x, y = np.array(xy).T
        h, w = img_size
        xmax, ymin = xy_limits
        x_scaled = (w * w / (w + 1)) * x / xmax
        y_scaled = (h * h / (h + 1)) * y / ymin
        xy_scaled = np.vstack([x_scaled, y_scaled]).T
        return xy_scaled

    def _convert_img_to_xy_scale(self, xy, img_size, xy_limits):
        """ Convert from coordinates on the image scale back to the normalized xy scale of the Polygons

        Args:
            xy - list of (x, y) tuples where x & y are in image coordinates
            img_size - Tuple of image (height, width)
            xy_limits - Tuple of Xmax and Ymin needed to properly scale image
        Returns:
            x_norm, y_norm - x and y coordinates normalized to be between 0 < x < 1 and -1 < y < 0
        """
        x, y = np.array(xy).T
        h, w = img_size
        xmax, ymin = xy_limits
        x_norm = x * xmax / (w * w / (w + 1))
        y_norm = y * ymin / (h * h / (h + 1))
        xy_norm = np.vstack([x_norm, y_norm]).T
        return xy_norm

    def _split_data(self, data, ratios=[60, 80], shuffle=True):
        """
        Shuffle and split a dataset according to the ratios given

        Args:
            data - dataset to split, should be of size (m, n+1), where m = number of examples, n = number of features (and the last column is the corresponding answers)
            ratios - Percentiles at which to split the data. For a training/cross-validation/test split of 60/20/20, use ratios = [60, 80]
        Returns:
            results - a tuple, where each element is a sub-array of data. There will be N+1 elements, where N is the number of ratios provided in argument. So, for ratios=[60, 80], will return a tuple of (train, CV, test), where train is 60% of data, CV is 20% of data, and test is 20% of data.

        """
        m, _ = data.shape
        if shuffle:
            shuffled_ind = np.random.permutation(m)
            shuffled_data = data[shuffled_ind]
        else:
            shuffled_data = data
        # Split by 60%, 20%, 20%
        inds = [round(ratio * m) for ratio in ratios]
        results = np.split(shuffled_data, indices_or_sections=inds)
        return results


############
# Run Code #
############
if __name__ == '__main__':
    dstl = Train_DSTL()
    # X, Y = dstl.get_training_set()
    train, cv, test = dstl.split_training_set()
