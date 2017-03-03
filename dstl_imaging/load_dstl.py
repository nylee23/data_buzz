#! /usr/bin/env python
#
# Program: load_dstl
#
# Author: Nick Lee
#
# Description: Code to extract training sets.
#
# Date: Feb 3, 2017
#

# Import Libraries
import numpy as np
import pandas as pd
import tifffile as tiff
from shapely.wkt import loads
from PIL import Image, ImageDraw
import os
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Load_DSTL(object):
    """
    Class to hold methods for reading and loading data from DSTL images
    """
    def __init__(self):
        # Save useful tables
        self.training_set = pd.read_csv('train_wkt_v4.csv')
        self.grid_sizes = pd.read_csv('grid_sizes.csv')
        self.grid_sizes.columns = pd.Index(['ImageID', 'Xmax', 'Ymin'])
        self.object_colors = {1: 'gray', 2: 'blue', 3: 'black', 4: 'brown', 5: 'green', 6: 'yellow', 7: 'turquoise', 8: 'blue', 9: 'red', 10: 'orange'}
        self._set_test_images()

    def _set_test_images(self):
        # Find image to use as test set - Use the image with 2nd highest number of polygons
        all_classes = self.training_set['ClassType'].unique()
        idx_test = []
        for obj_class in all_classes:
            object_df = self.training_set.query('ClassType == {:d} & MultipolygonWKT != "MULTIPOLYGON EMPTY"'.format(obj_class))
            ind_test = object_df['MultipolygonWKT'].apply(len).sort_values(ascending=False).index[1]
            idx_test.append(ind_test)
        self.test_images = pd.DataFrame({'index_test': idx_test}, index=pd.Index(all_classes, name='ClassType'))

    def load_subset(self, object_class=1, test_size=0.001, **kwargs):
        pass

    def create_subset(self, object_class=1, single_image_test=False, fraction=0.001):
        """
        Create a small subset of the full training set.

        New training set will be created by randomly selecting a fraction of the full training set (size controlled by fraction).

        New CV set is either created from splitting the reduced training set by 70/30, or chosen as a small region around one of the polygons in the test image.

        New test set is either randomly chosen from the full test set, or is chosen as a small region around a different polygon in the test image

        Keywords:
            object_class - Desired ClassType

            single_image_test - Boolean that controls how CV & test set are created. True uses polygons from the test image, False uses fractional splits from the original data set. Default=False

            fraction - Float between 0 and 1. Sets the test_size keyword for train_test_split that determines what fraction of the data to return.
        Returns:
            X_train - Feature matrix for training set
            y_train - Answer vector for training set
            X_cv - Feature matrix for CV set
            y_cv - Answer vector for CV set
            X_test - Feature matrix for test set
            y_test - Answer vector for test set
        """
        fraction = float(fraction)  # Make sure test_size is in correct dtype
        # Training set and scaler
        X_all, y_all, X_test, y_test, scaler = self.load_full_training_set(object_class=object_class)
        print('training sets loaded')
        # Reduce Training set to test_size
        X, _, y, _ = train_test_split(X_all, y_all, test_size=fraction, random_state=42)
        print('Training set reduced')
        if single_image_test:
            X_cv, y_cv = self._extract_test_region(object_class=object_class, seed=0)
            X_test, y_test = self._extract_test_region(object_class=object_class, seed=42)
            X_test = scaler.transform(X_test)
            X_cv = scaler.transform(X_cv)
        else:
            # Create CV set
            X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.7, random_state=0)
            print('CV set created')
            X_test, _, Y_test, _ = train_test_split(X_test, y_test, test_size=fraction)
            print('Test set reduced')
        return X_train, y_train, X_cv, y_cv, X_test, y_test

    def _extract_test_region(self, object_class=1, buffer_size=4, return_ind=False, seed=0):
        """
        Extract a small region around a single polygon.

        Will create a matrix of features and answers that can be used for a test set. The test set contains pixels in a region around the polygon given by ind_shape.

        Note: Does not work for object_class = 3

        Keywords:
            seed - Use to set randomization seed. Call this function with different seeds to try to get two different polygons from the Multipolygon
        """
        # Select Training data row that contains test image & masks
        test_row = self.training_set.loc[self.test_images.loc[8].values[0]]
        # Find first image that contains the desired object_class
        mask = self._create_mask(test_row)
        img_size, xy_limits, image = self._get_image_properties(test_row['ImageId'])
        all_shapes = loads(test_row['MultipolygonWKT'])
        # Find region around first polygon
        np.random.seed(seed)
        flag = 0
        while flag < 1:
            # Randomly choose a polygon
            ind_shape = np.random.randint(0, len(all_shapes))
            shape = all_shapes[ind_shape]
            xy_perimeter = np.array(shape.exterior.coords)
            xy_scaled = self._convert_xy_to_img_scale(xy_perimeter, img_size, xy_limits)
            # Cut out region around polygon
            xlim = np.round([xy_scaled[:, 0].min() - buffer_size, xy_scaled[:, 0].max() + buffer_size])
            ylim = np.round([xy_scaled[:, 1].min() - buffer_size, xy_scaled[:, 1].max() + buffer_size])
            yy, xx = np.meshgrid(np.arange(*xlim), np.arange(*ylim))
            try:
                triples = np.array([image[int(x), int(y), :] for (x, y) in zip(xx.ravel(), yy.ravel())])
            except IndexError:
                flag = 0
            else:
                flag = 1
                mask = np.array([mask[int(x), int(y)] for (x, y) in zip(xx.ravel(), yy.ravel())], dtype='b')
        if return_ind:
            results = (triples, mask, ind_shape)
        else:
            results = (triples, mask)
        return results

    # Methods to create and save the full training set
    def load_full_training_set(self, object_class=1, new=False):
        h5f_name = 'h5_files/training_set_class_{:}.h5'.format(object_class)
        scaler_keys = ['mean_', 'var_', 'scale_', 'n_samples_seen_']
        if new and os.path.isfile(h5f_name):
            os.remove(h5f_name)
        try:
            h5f = h5py.File(h5f_name, 'r')
        except OSError:
            X_train, Y_train, X_test, Y_test, scaler = self._create_training_set(object_class=object_class)
            with h5py.File(h5f_name, 'w') as h5f:
                h5f.create_dataset('X_train', data=X_train)
                h5f.create_dataset('Y_train', data=Y_train)
                h5f.create_dataset('X_test', data=X_test)
                h5f.create_dataset('Y_test', data=Y_test)
                for key in scaler_keys:
                    h5f.create_dataset(key, data=getattr(scaler, key))
        else:
            X_train = h5f['X_train'].value
            Y_train = h5f['Y_train'].value
            X_test = h5f['X_test'].value
            Y_test = h5f['Y_test'].value
            # Reconstruct scaler
            scaler = StandardScaler()
            for key in scaler_keys:
                setattr(scaler, key, h5f[key].value)
            h5f.close()
        return X_train, Y_train, X_test, Y_test, scaler

    def _save_training_sets(self, classes=None):
        """ Create a h5f file for each object class """
        if classes is None:
            class_types = self.training_set['ClassType'].unique()
        elif len(classes) == 1:
            class_types = [classes]
        else:
            class_types = classes
        for obj_class in class_types:
            self.load_full_training_set(object_class=obj_class, new=True)
            print('Training Set for Class {:d} complete'.format(obj_class))

    def _create_training_set(self, object_class=1):
        """ Create a normalized training and test set """
        X_raw, y_train, X_test, y_test = self._get_training_set(object_class=object_class)
        # Normalize features
        scaler = StandardScaler().fit(X_raw)
        X_train = scaler.transform(X_raw)
        X_test = scaler.transform(X_test)
        return X_train, y_train, X_test, y_test, scaler

    def _get_training_set(self, object_class=1):
        """
        Create a training set and test set for a given object class based on individual pixels.

        Test set is chosen to be the image that contains the 2nd most polygons of the given object_class.
        Returns:
            training_set - A (m x n) sized numpy array corresponding to the pixel values in each band for the training examples.
            training_answers - A (m x 1) sized numpy array corresponding to the answer (1 = classified as part of class, 0 = not)
        """
        object_df = self.training_set.query('ClassType == {:d} & MultipolygonWKT != "MULTIPOLYGON EMPTY"'.format(object_class))
        # Create training and test sets
        train_examples = []
        train_answers = []
        for idx, row in object_df.iterrows():
            image = self._get_image(row['ImageId'])
            mask = self._create_mask(row)
            # Check if this image is the test image
            if idx == self.test_images.loc[object_class].values[0]:
                test_examples = image.reshape(-1, 3)
                test_answers = mask.ravel().astype('b')
            else:
                train_examples.append(image.reshape(-1, 3))
                train_answers.append(mask.ravel().astype('d'))
        training_set = np.concatenate((train_examples))
        training_answers = np.concatenate((train_answers))
        return training_set, training_answers, test_examples, test_answers

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


############
# Run Code #
############
if __name__ == '__main__':
    dstl = Load_DSTL()
    # X, Y = dstl._get_training_set()

