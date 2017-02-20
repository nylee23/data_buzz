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
import os
import h5py
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


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

    # Training
    def train_logistic(self, object_class=1):
        """ Train a Logistic Regression """
        pass

    def cross_validate_svc(self, object_class=1):
        """
        http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html#sphx-glr-auto-examples-model-selection-grid-search-digits-py
        """
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = self.load_training_set(object_class=object_class)

        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        scores = ['precision', 'recall']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                               scoring='%s_macro' % score)
            clf.fit(X_train, Y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = Y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()

    # Methods to create and save training sets
    def load_training_set(self, object_class=1, new=False):
        """
        Load the training set, cross-validation set, and test set for a given object class.

        Includes code to load from a H5F file, which was the most efficient way to store the information.
        """
        h5f_name = 'h5_files/training_set_class_{:}.h5'.format(object_class)
        if new and os.path.isfile(h5f_name):
            os.remove(h5f_name)
        try:
            h5f = h5py.File(h5f_name, 'r')
        except OSError:
            X, Y = self._get_training_set(object_class=object_class)
            X_train, Y_train, X_cv, Y_cv, X_test, Y_test = self._split_training_set_indices(X, Y)
            with h5py.File(h5f_name, 'w') as h5f:
                h5f.create_dataset('X_train', data=X_train)
                h5f.create_dataset('Y_train', data=Y_train)
                h5f.create_dataset('X_cv', data=X_cv)
                h5f.create_dataset('Y_cv', data=Y_cv)
                h5f.create_dataset('X_test', data=X_test)
                h5f.create_dataset('Y_test', data=Y_test)
        else:
            X_train = h5f['X_train'][:]
            Y_train = h5f['Y_train'][:]
            X_cv = h5f['X_cv'][:]
            Y_cv = h5f['Y_cv'][:]
            X_test = h5f['X_test'][:]
            Y_test = h5f['Y_test'][:]
            h5f.close()
        return X_train, Y_train, X_cv, Y_cv, X_test, Y_test

    def _save_all_training_sets(self):
        """ Create a h5f file for each object class """
        for obj_class in self.training_set['ClassType'].unique():
            self.load_training_set(object_class=obj_class, new=True)

    def _split_training_set_indices(self, X, Y):
        """
        Given a dataset with known answers, split the dataset into a training set, cross-validation set, and test set.

        Note: SKLEARN has a built-in method to do shuffle & splitting: sklearn.model_selection.train_test_split, but it's not any faster (and in fact is a little bit slower)
        """
        np.random.seed = 0
        m = len(Y)
        indices = np.arange(m)
        sets = np.zeros(m, dtype=np.int8)
        for answer in [0, 1]:
            ind_train, ind_cv, ind_test = self._shuffle_split(indices[Y == answer])
            sets[ind_cv] = 1
            sets[ind_test] = 2
        X_train = X[sets == 0]
        Y_train = Y[sets == 0]
        X_cv = X[sets == 1]
        Y_cv = Y[sets == 1]
        X_test = X[sets == 2]
        Y_test = Y[sets == 2]
        return X_train, Y_train, X_cv, Y_cv, X_test, Y_test

    def _shuffle_split(self, indices, ratios=[60, 80]):
        """
        Shuffle and split the elements of an array according to the ratios given

        Args:
            indices - An m-element array
            ratios - Percentiles at which to split the data. For a training/cross-validation/test split of 60/20/20, use ratios = [60, 80]
        Returns:
            results - a tuple, where each element is a sub-array of the input. There will be N+1 elements, where N is the number of ratios provided in argument. So, for ratios=[60, 80], will return a tuple of (train, CV, test), where train is a vector containing 60% of data, CV is 20% of data, and test is 20% of data.
        """
        shuffled_indices = np.random.permutation(indices)
        m = len(indices)
        inds = [round(ratio / 100 * m) for ratio in ratios]
        results = np.split(shuffled_indices, indices_or_sections=inds)
        return results

    def _get_training_set(self, object_class=1):
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
        inds = [round(ratio / 100 * m) for ratio in ratios]
        results = np.split(shuffled_data, indices_or_sections=inds)
        return results


############
# Run Code #
############
if __name__ == '__main__':
    dstl = Train_DSTL()
    # X, Y = dstl._get_training_set()

