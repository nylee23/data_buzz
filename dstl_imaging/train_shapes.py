#! /usr/bin/env python
#
# Program: train_shapes
#
# Author: Nick Lee
#
# Description: Code to classify shapes
#
# Date: Feb 3, 2017
#

# Import libraries
import numpy as np
from shapely.wkt import loads
from PIL import Image, ImageDraw
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# Local libraries
from nickPickle import loadPickle, savePickle
from train_dstl import DSTL_XGB, DSTL_LogReg
from load_dstl import Load_DSTL
from visualize_dstl import display_three_band_image


class Train_Shapes(object):
    """ Train and display shapes """
    def __init__(self, image_id='6120_2_2'):
        all_shapes = Load_Shapes()
        self.image_id = image_id
        self.full_set = all_shapes.load_shapes(image_id=image_id)

    def compare_predictions(self):
        """ Make a plot with all of the shapes in the test set """
        ax = display_three_band_image(image_id=self.image_id, overlay=False, showplot=False)
        agree = (self.test_set['pred'] == 1) & (self.test_set['labels'] == 1)
        false_pos = (self.test_set['labels'] == 0) & (self.test_set['pred'] == 1)
        missed_pos = (self.test_set['labels'] == 1) & (self.test_set['pred'] == 0)
        # Make a mask for each category
        ax.contour(self._custom_mask(self.test_set.loc[agree, 'shape'].values), colors='blue')
        ax.contour(self._custom_mask(self.test_set.loc[false_pos, 'shape'].values), colors='green')
        ax.contour(self._custom_mask(self.test_set.loc[missed_pos, 'shape'].values), colors='red')
        plt.show()

    def _custom_mask(self, shapes):
        """
        Given a row in the training set dataframe, return the mask corresponding to the given Multi-Polygon.
        """
        dstl = Load_DSTL()
        img_size, xy_limits, _ = dstl._get_image_properties(self.image_id)
        # Make a True/False image using PIL
        img = Image.new('1', img_size[-1::-1], 0)
        # Fill in the Image with the polygons
        for shape in shapes:
            xy_perimeter = np.array(shape.exterior.coords)
            xy_scaled = dstl._convert_xy_to_img_scale(xy_perimeter, img_size, xy_limits)
            poly = [(coord[0], coord[1]) for coord in xy_scaled]
            ImageDraw.Draw(img).polygon(poly, fill=1)
        mask = np.array(img)
        return mask

    def train_logreg(self, class_type=1, test_size=0.2, cv_size=0.3, plot_df=True, explore_db=False):
        """ Train logistic regression """
        # Make sure we have features
        self._select_features(class_type=class_type, test_size=test_size, cv_size=cv_size)
        feat_col = [col for col in self.full_set.columns if 'feat' in col]
        # Train logistic regressioin
        log_reg = DSTL_LogReg()
        log_reg.train(self.train_set[feat_col].values, self.train_set['labels'].values, X_cv=self.cv_set[feat_col].values, y_cv=self.cv_set['labels'], plot_df=plot_df)
        # Predict using best decision boundary
        self.test_set['pred'] = log_reg.predict(self.test_set[feat_col].values)
        self.compare_predictions()
        # Evaluate Jaccard Similarity Score
        jacc = metrics.jaccard_similarity_score(self.test_set['labels'], self.test_set['pred'])
        return jacc

    def train_xgb(self, class_type=1, test_size=0.2, cv_size=0.3):
        """ Train a XGB classifier """
        # Make sure we have features
        self._select_features(class_type=class_type, test_size=test_size, cv_size=cv_size)
        feat_col = [col for col in self.full_set.columns if 'feat' in col]
        # Create and train classifier
        xgb_classifier = DSTL_XGB()
        xgb_classifier.train(self.train_set[feat_col].values, self.train_set['labels'].values, X_cv=self.cv_set[feat_col].values, y_cv=self.cv_set['labels'])
        # Predict using default boundaries
        self.test_set['pred'] = xgb_classifier.predict(self.test_set[feat_col].values)
        self.compare_predictions()
        # Evaluate Jaccard Similarity Score
        jacc = metrics.jaccard_similarity_score(self.test_set['labels'], self.test_set['pred'])
        return jacc

    def _select_features(self, class_type=1, test_size=0.2, cv_size=0.3):
        """
        Take an overall feature set with all classes labeled and create a true labels vector that is 1 or 0 if the feature is in that class type
        """
        # Create labels for each example
        self.full_set['labels'] = (self.full_set['class_type'] == class_type).astype('b')
        # Split into training and (optionally CV) set
        train_set, test_set = train_test_split(self.full_set, stratify=self.full_set['labels'])
        self.test_set = test_set
        if cv_size is not None:
            train_set, cv_set = train_test_split(train_set, stratify=train_set['labels'])
            self.cv_set = cv_set
        self.train_set = train_set


# Class to go through all shapes and extract the desired features.
class Load_Shapes(Load_DSTL):
    def __init__(self):
        super().__init__()

    def load_shapes(self, image_id='6120_2_2', new=False):
        """
        Quickly load the feature set corresponding to all of the shapes in the image
        """
        pkl_name = 'pkl_files/{:}_shape_features_df.pkl'.format(image_id)
        if new and os.path.isfile(pkl_name):
            os.remove(pkl_name)
        try:
            full_set = pd.read_pickle(pkl_name)
        except OSError:
            full_set = self._get_shapes(image_id)
            full_set.to_pickle(pkl_name)
        return full_set

    def _get_shapes(self, image_id='6120_2_2'):
        """ Get all the shapes belonging to a certain class """
        table = self.training_set.loc[self.training_set['ImageId'] == image_id]
        image = self._get_image(image_id)
        shapes = []
        for idx, row in table.iterrows():
            if row['MultipolygonWKT'] != 'MULTIPOLYGON EMPTY':
                feat, label = self._extract_features(row, image=image)
                shapes.append(list(loads(row['MultipolygonWKT'])))
                try:
                    features = np.vstack([features, feat])
                except NameError:
                    features = feat
                    labels = label
                else:
                    labels = np.hstack([labels, label])
            print('{:} class done'.format(row['ClassType']))
        full_set = pd.DataFrame(features, columns=['feat_{:}'.format(i) for i in range(features.shape[1])])
        full_set['class_type'] = labels
        full_set['shapes'] = np.hstack(shapes)
        return full_set

    def _extract_features(self, row, image=None):
        shapes = loads(row['MultipolygonWKT'])
        feats = []
        if image is None:
            image = self._get_image(row['ImageId'])
        for ii, shape in enumerate(shapes):
            shape_mask = self._create_single_mask(shape, row['ImageId'])
            colors_in_shape = image[shape_mask]
            avg_color = colors_in_shape.mean(axis=0)
            std_color = colors_in_shape.std(axis=0)
            # Find size
            shape_size = colors_in_shape.shape[0]
            shape_corners = np.array(shape.exterior.coords).shape[0]
            # Make feature array
            feats.append(np.hstack([avg_color, std_color, shape_size, shape_corners]))
        features = np.vstack(feats)
        if features.ndim > 1:
            labels = np.zeros(features.shape[0]) + row['ClassType']
        else:
            labels = np.zeros(1) + row['ClassType']
        return features, labels

    def _create_single_mask(self, shape, image_id):
        # Image properties
        img_size, xy_limits, _ = self._get_image_properties(image_id)
        # Pixels that are in shape
        # Make a True/False image using PIL
        img = Image.new('1', img_size[-1::-1], 0)
        xy_perimeter = np.array(shape.exterior.coords)
        xy_scaled = self._convert_xy_to_img_scale(xy_perimeter, img_size, xy_limits)
        poly = [(coord[0], coord[1]) for coord in xy_scaled]
        ImageDraw.Draw(img).polygon(poly, fill=1)
        mask = np.array(img)
        return mask


############
# Run Code #
############
if __name__ == '__main__':
    shapes = Train_Shapes()



