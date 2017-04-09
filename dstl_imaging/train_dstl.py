#! /usr/bin/env python
#
# Program: train_dstl
#
# Author: Nick Lee
#
# Description: Code to train the data
#
# Date: Feb 3, 2017
#

# Import Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
# Local libraries
from load_dstl import Load_DSTL
from visualize_dstl import display_three_band


class Train_Pixels(object):
    """ Class that handles training on individual pixels """
    def __init__(self, class_type=1, windows=True, radius=3):
        """ Select training set and create features set """
        if windows:
            self.radius = radius
        else:
            self.radius = None
        self.class_type = class_type
        self._choose_regions()
        self._make_features()

    def train_logreg(self, plot_df=True, explore_db=False):
        """ Train logistic regression """
        log_reg = DSTL_LogReg()
        log_reg.train(self.features['train'], self.labels['train'], X_cv=self.features['cv'], y_cv=self.labels['cv'], plot_df=plot_df)
        # Predict using best decision boundary
        pred = log_reg.predict(self.features['test'])
        self._display_test_image(pred)
        # Try out some different decision boundaries
        if explore_db:
            for db in np.linspace(-1, 1, 5):
                log_reg.decision_boundary = db
                pred = log_reg.predict(self.features['test'])
                self._display_test_image(pred)
        # Evaluate Jaccard Similarity Score
        jacc = metrics.jaccard_similarity_score(self.labels['test'], pred)
        return jacc

    def train_xgb(self, explore_bound=False):
        """ Train a XGB classifier """
        # Create and train classifier
        xgb_classifier = DSTL_XGB()
        xgb_classifier.train(self.features['train'], self.labels['train'], X_cv=self.features['cv'], y_cv=self.labels['cv'])
        # Predict using default boundaries
        pred = xgb_classifier.predict(self.features['test'])
        self._display_test_image(pred)
        # Try out a couple of test boundaries
        if explore_bound:
            for boundary in [0.3, 0.35, 0.4, 0.45, 0.5]:
                pred = xgb_classifier.predict(self.features['test'], boundary=boundary)
                self._display_test_image(pred)
        # Evaluate Jaccard Similarity Score
        jacc = metrics.jaccard_similarity_score(self.labels['test'], pred)
        return jacc

    def _display_test_image(self, pred):
        """ Plot test image and mask """
        # Check if we're using sliding windows
        if self.radius is not None:
            test_img = self.images['test'][self.radius: -1 * self.radius, self.radius: -1 * self.radius]
            test_mask = self.masks['test'][self.radius: -1 * self.radius, self.radius: -1 * self.radius]
        else:
            test_img = self.images['test']
            test_mask = self.images['test']
        display_three_band(test_img, pred.reshape(test_mask.shape), true_mask=test_mask)

    def _choose_regions(self, display_regions=False):
        """
        Select training sets, CV sets, and test sets as regions where we know there are objects.  Right now, working only with class 1 (buildings)
        """
        dstl = Load_DSTL()
        if self.class_type == 1:
            # Select regions where there are buildings (with red roofs)
            test_image, test_mask = dstl.extract_region_pos(2300, 3000, cutout_size=[400, 400], object_class=self.class_type)
            train_image, train_mask = dstl.extract_region_pos(1900, 3100, cutout_size=[400, 400], object_class=self.class_type)
            cv_image, cv_mask = dstl.extract_region_pos(950, 1450, cutout_size=[200, 200], object_class=self.class_type)
        elif self.class_type == 5:
            train_image, train_mask = dstl.extract_region_pos(1150, 2150, cutout_size=[400, 400], object_class=self.class_type)
            test_image, test_mask = dstl.extract_region_pos(2300, 3000, cutout_size=[400, 400], object_class=self.class_type)
            cv_image, cv_mask = dstl.extract_region_pos(1900, 1950, cutout_size=[400, 400], object_class=self.class_type)
        else:
            pass
        self.images = {'train': train_image, 'cv': cv_image, 'test': test_image}
        self.masks = {'train': train_mask, 'cv': cv_mask, 'test': test_mask}
        if display_regions:
            for key in self.images.keys():
                display_three_band(self.images[key], self.masks[key], colors='green', title='{:} region'.format(key))

    def _make_features(self):
        """ Create feature and label sets from the images """
        self.features = {}
        self.labels = {}
        for key in ['train', 'cv', 'test']:
            if self.radius is not None:
                feat, label = self._sliding_window(self.images[key], self.masks[key], window_radius=self.radius)
                self.features[key] = feat
                self.labels[key] = label
            else:
                self.features[key] = self.images[key].reshape(-1, 3)
                self.labels[key] = self.masks[key].ravel()

    def _automatic_training_set(self, n_cutouts=100):
        """
        Create a new training set based on taking cutouts around random shapes
        """
        dstl = Load_DSTL()
        np.random.seed(42)
        for ii in range(n_cutouts):
            # Get region around a shape
            triples, mask, ind_shape, img_dim = dstl.extract_region(object_class=self.class_type, image_id='6120_2_2', buffer_size=10)
            if self.radius is not None:
                triples, mask = self._sliding_window(triples.reshape(*img_dim, 3), mask.reshape(img_dim), window_radius=self.radius)
            # Add to Feature Matrix
            if ii == 0:
                features = triples
                labels = mask
            else:
                features = np.vstack([features, triples])
                labels = np.hstack([labels, mask])
        return features, labels

    def _sliding_window(self, image, mask, window_radius=3):
        """
        Given an image and a mask, create a feature set that consists of a sliding square, with the corresponding label being the mask value in the center pixel.
        """
        height, width = image.shape[:2]
        features = []
        for yy in range(window_radius, height - window_radius):
            for xx in range(window_radius, width - window_radius):
                features.append(image[yy - window_radius: yy + window_radius + 1, xx - window_radius: xx + window_radius + 1].ravel())
        labels = mask[window_radius: -1 * window_radius, window_radius: -1 * window_radius].ravel()
        return np.array(features), labels


# Classes that do the machine learning - normalization, training, predicting
class DSTL_XGB(object):
    """
    XGBoost Classifier
    """
    def predict(self, features, boundary=0.5):
        d_test = xgb.DMatrix(self.scaler.transform(features))
        preds = self.bst.predict(d_test)
        return (preds > boundary).astype('b')

    def train(self, X_train, y_train, X_cv=None, y_cv=None):
        # Normalize features
        self.scaler = StandardScaler().fit(X_train)
        X_norm = self.scaler.transform(X_train)
        # Get in XGB format
        d_train = xgb.DMatrix(X_norm, label=y_train)
        d_cv = xgb.DMatrix(self.scaler.transform(X_cv), label=y_cv)
        # Train XGBoost
        params = {'bst:max_depth': 3, 'bst:eta': 0.3, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
        watchlist = [(d_cv, 'eval'), (d_train, 'train')]
        num_round = 2
        self.bst = xgb.train(params, d_train, num_round, watchlist)


class DSTL_LogReg(object):
    """
    Logistic Regression Classifier
    """
    def predict(self, features):
        """ """
        # Normalize features using scaler
        norm_features = self.scaler.transform(features)
        # Run algorithm
        if hasattr(self, 'decision_boundary'):
            dec_func = self.classifier.decision_function(norm_features)
            pred = (dec_func > self.decision_boundary).astype('b')
        else:
            pred = self.classifier.predict(norm_features)
        # Return predictions
        return pred

    def train(self, X_train, y_train, X_cv=None, y_cv=None, plot_df=False, **kwargs):
        """ Train a Logistic Regression algorithm """
        # Normalize features
        self.scaler = StandardScaler().fit(X_train)
        X_norm = self.scaler.transform(X_train)
        # Train and Cross-validate
        log_reg = LogisticRegression(**kwargs)
        if X_cv is not None and y_cv is not None:
            # Train classifier
            self.classifier = log_reg
            self.classifier.fit(X_norm, y_train)
            # Find best decision boundary using CV set
            self.find_decision_boundary(X_cv, y_cv, plot=plot_df)
        else:
            # Cross-validate using training set
            weights = ['balanced']
            # Make list of class weight fractions
            for weight0 in np.logspace(-1.2, -0.8, 10):
                weights.append({0: weight0, 1: 1 - weight0})
            parameters = {'class_weight': weights, 'C': [0.1, 1, 10]}
            self.classifier = GridSearchCV(log_reg, parameters, scoring='f1')
            self.classifier.fit(X_train, y_train)

    def find_decision_boundary(self, X_cv, y_cv, plot=False):
        """
        Find the best decision boundary by minimizing F1 score using CV data
        """
        # Find possible decision functions
        dec_func = self.classifier.decision_function(self.scaler.transform(X_cv))
        dec_func_range = np.linspace(dec_func[y_cv == 1].min(), dec_func[y_cv == 1].max(), 20)
        # Empty arrays
        prec = []
        recall = []
        f1 = []
        for boundary in dec_func_range:
            # Make predictions using this decision boundary
            y_pred = (dec_func > boundary).astype('b')
            # Score prediction
            prec.append(metrics.precision_score(y_cv, y_pred))
            recall.append(metrics.recall_score(y_cv, y_pred))
            f1.append(metrics.f1_score(y_cv, y_pred))
        # Find decision boundary that corresponds to best F1 score
        db_ind = np.argmax(f1)
        if plot:
            fig, ax = plt.subplots()
            ax.plot(dec_func_range, prec, 'r-', label='precision')
            ax.plot(dec_func_range, recall, 'b-', label='recall')
            ax.plot(dec_func_range, f1, 'k-', label='F1 score')
            ax.plot([dec_func_range[db_ind]] * 2, [0, 1], 'k--')
            ax.legend()
            plt.show()
        self.decision_boundary = dec_func_range[db_ind]


def evaluate_set_size(algorithm='log reg', class_type=1):
    """ See how the training set size impacts accuracy """
    model = Train_DSTL(class_type=class_type)
    if algorithm == 'xgb':
        classifier = model.train_xgb
    elif algorithm == 'log reg':
        classifier = model.train_logreg
    # Train on user-defined images
    # Loop through training set size
    shape_sizes = [0, 50, 100, 150, 200]
    jacc_shapes = [classifier()]
    for n_shapes in shape_sizes[1:]:
        feat, label = model._automatic_training_set()
        model.features['train'] = feat
        model.labels['train'] = label
        jacc_shapes.append(classifier())
    plt.plot(shape_sizes, jacc_shapes, 'k-')
    plt.show()


############
# Run Code #
############
if __name__ == '__main__':
    pixel_model = Train_Pixels(class_type=1)



