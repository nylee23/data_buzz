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
from load_dstl import Load_DSTL
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb


class DSTL_XGBoost(object):
    """ Use XGBoost to train a model """
    def __init__(self):
        """ Initialize """
        self._get_data()

    def _get_data(self):
        """ Load all of the relevant training sets """
        self.dstl = Load_DSTL()
        training_subset = self.dstl.load_subset()
        # Split training set to create train_cv set
        fraction =
        X_train, X_train_cv, y_train, y_train_cv = train_test_split(X_test, y_test, test_size=fraction)



    def train_xgboost(self):
        """ Train an XGBoost system """
        X_train, y_train, X_cv, y_cv, X_test, y_test = self.data.load_subset()
        d_train = xgb.DMatrix(X_train, label=y_train)
        d_cv = xgb.DMatrix(X_cv, label=y_cv)
        d_test = xgb.DMatrix(X_test, label=y_test)

        # Train XGBoost
        params = {'bst:max_depth': 3, 'bst:eta': 0.3, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
        watchlist  = [(d_cv,'eval'), (d_train,'train')]
        num_round = 2
        bst = xgb.train(params, d_train, num_round, watchlist)

        # this is prediction
        preds = bst.predict(d_test)
        labels = d_test.get_label()
        print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))



class DSTL_Logistic(Load_DSTL):
    """ Class to do logistic regression """
    def __init__(self):
        super().__init__()

    def train_logistic(self, object_class=1, fraction=0.001, plot_df=False):
        # Get data
        self._get_training_set(object_class=object_class, fraction=fraction)
        # Train classifier
        log_reg = LogisticRegression(solver='sag', random_state=42)
        log_reg.fit(self.X_train, self.y_train)
        # Cross validate
        self.decision_boundary = self.find_decision_boundary(log_reg, plot=plot_df)
        # Test accuracy
        jacc_score = self.test_model(log_reg)
        return jacc_score

    def test_model(self, model, pprint=True):
        try:
            boundary = self.decision_boundary
        except:
            boundary = self.find_decision_boundary(model)
        # Evaluate test set
        dec_func = model.decision_function(self.X_test)
        y_pred = (dec_func > boundary).astype('b')
        jaccard_score = metrics.jaccard_similarity_score(self.y_cv, y_pred)
        if pprint:
            print(metrics.classification_report(self.y_cv, y_pred))
            print('Jaccard Score is {:}'.format(jaccard_score))
        return jaccard_score

    def find_decision_boundary(self, model, plot=False):
        """
        Find the best decision boundary by minimizing F1 score using CV data
        """
        # Find possible decision functions
        dec_func = model.decision_function(self.X_cv)
        dec_func_range = np.linspace(dec_func[self.y_cv == 1].min(), dec_func[self.y_cv == 1].max(), 20)
        # Empty arrays
        prec = []
        recall = []
        f1 = []
        for boundary in dec_func_range:
            # Make predictions using this decision boundary
            y_pred = (dec_func > boundary).astype('b')
            # Score prediction
            prec.append(metrics.precision_score(self.y_cv, y_pred))
            recall.append(metrics.recall_score(self.y_cv, y_pred))
            f1.append(metrics.f1_score(self.y_cv, y_pred))
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
        return dec_func_range[db_ind]

    def _get_training_set(self, object_class=1, fraction=0.001, **kwargs):
        """
        Retrieve subset, but perform stratified sampling of training set
        """
        X_train, y_train, self.X_cv, self.y_cv, self.X_test, self.y_test = self.load_subset(object_class=object_class, fraction=fraction)
        # Generate a CV sample that is half positive, half negative
        ind_all = np.arange(len(y_train))
        ind_pos = ind_all[y_train == 1]
        ind_neg = ind_all[y_train == 0]
        neg_subsample = ind_all[np.random.choice(ind_neg, size=len(ind_pos))]
        self.X_train = np.vstack((X_train[ind_pos], X_train[neg_subsample]))
        self.y_train = np.hstack((y_train[ind_pos], y_train[neg_subsample]))


# Functions
def evaluate_sample_size(model=None, object_class=1):
    """ See how the jaccard score changes with sample size """
    if model is None:
        model = DSTL_Logistic()
    frac_range = np.logspace(-4, -2, 5)
    jacc = []
    for fraction in frac_range:
        jacc.append(model.train_logistic(object_class=object_class, fraction=fraction))
    plt.plot(frac_range, jacc, 'k-')
    plt.show()


def grid_search():
    """ Find best parameters for logistic regression """
    dstl = Load_DSTL()
    X_train, y_train, X_cv, y_cv, X_test, y_test = dstl.load_subset(fraction=0.01)
    weights = ['balanced']
    # Make list of class weight fractions
    for weight0 in np.logspace(-1.2, -0.8, 10):
        weights.append({0: weight0, 1: 1 - weight0})
    parameters = {'class_weight':weights, 'C':[0.1, 1, 10]}
    log_reg = LogisticRegression()
    clf = GridSearchCV(log_reg, parameters, scoring='f1')
    clf.fit(X_train, y_train)
    return clf


def choose_model():
    dstl_log = DSTL_Logistic()
    dstl_log._get_training_set(fraction=0.01)
    log_reg = LogisticRegression(solver='sag', random_state=0)
    log_reg_weighted = LogisticRegression(solver='sag', random_state=0, class_weight={0: 0.11, 1: 0.89})
    log_reg_balanced = LogisticRegression(solver='sag', random_state=0, class_weight='balanced')
    model_names = ['Unweighted', 'Weighted', 'Balanced']
    for ii, model in enumerate([log_reg, log_reg_weighted, log_reg_balanced]):
        model.fit(dstl_log.X_train, dstl_log.y_train)
        # Evaluate using decision boundary in model
        print('{:} model has F1 score of {:}'.format(model_names[ii], metrics.f1_score(dstl_log.y_cv, model.predict(dstl_log.X_test))))
        # Use custom decision boundary
        dec_bound = dstl_log.find_decision_boundary(model)
        y_pred = (model.decision_function(dstl_log.X_test) > dec_bound).astype('b')
        print('{:} model with custom decision boundary has F1 score of {:}'.format(model_names[ii], metrics.f1_score(dstl_log.y_cv, y_pred)))
    return dstl_log


def visualize_training_set(object_class=1):
    dstl = Load_DSTL()
    X_train, y_train, X_test, y_test = dstl.load_subset(object_class=1, fraction=1e-4)
    train_df = pd.DataFrame(X_train, columns=['r', 'g', 'b'])
    train_df['y'] = y_train
    sns.pairplot(train_df, hue='y')
    plt.show()


def plot_learning_curve(model='logistic', object_class=1):
    dstl = Load_DSTL()
    log_reg = LogisticRegression(solver='sag', random_state=42)
    train_err = []
    test_err = []
    prec = []
    recall = []
    f1 = []
    sample_sizes = np.logspace(-4, -2, 6)
    for fraction in sample_sizes:
        X_train, y_train, X_test, y_test = dstl.load_subset(object_class=object_class, fraction=fraction)
        log_reg.fit(X_train, y_train)
        y_pred = log_reg.predict(X_test)
        # Save metrics
        train_err.append(log_reg.score(X_train, y_train))
        test_err.append(log_reg.score(X_test, y_test))
        prec.append(metrics.precision_score(y_test, y_pred))
        recall.append(metrics.recall_score(y_test, y_pred))
        f1.append(metrics.f1_score(y_test, y_pred))
    # plt.plot(sample_sizes, train_err, 'r-')
    # plt.plot(sample_sizes, test_err, 'b-')
    plt.plot(sample_sizes, f1, 'k-')
    plt.show()


def svm_learning_curve(object_class=1):
    dstl = Load_DSTL()
    svm = SVC()
    X_train, y_train, X_test, y_test = dstl.load_subset(object_class=object_class, fraction=1e-4)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)


    train_err = []
    train_err = []
    test_err = []
    prec = []
    recall = []
    f1 = []
    # sample_sizes = np.logspace(-4, -2, 6)
    # for fraction in sample_sizes:
    #     X_train, y_train, X_test, y_test = dstl.load_subset(object_class=object_class, fraction=fraction)
    #     svm.fit(X_train, y_train)
    #     y_pred = log_reg.predict(X_test)
    #     # Save metrics
    #     train_err.append(log_reg.score(X_train, y_train))
    #     test_err.append(log_reg.score(X_test, y_test))
    #     prec.append(metrics.precision_score(y_test, y_pred))
    #     recall.append(metrics.recall_score(y_test, y_pred))
    #     f1.append(metrics.f1_score(y_test, y_pred))
    # # plt.plot(sample_sizes, train_err, 'r-')
    # # plt.plot(sample_sizes, test_err, 'b-')
    # plt.plot(sample_sizes, f1, 'k-')
    # plt.show()

############
# Run Code #
############
if __name__ == '__main__':
    dstl_log = DSTL_Logistic()
    dstl = Load_DSTL()

    """ Train an XGBoost system """
    X_train, y_train, X_cv, y_cv, X_test, y_test = dstl.load_subset()
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_cv = xgb.DMatrix(X_cv, label=y_cv)
    d_test = xgb.DMatrix(X_test, label=y_test)

    # Train XGBoost
    params = {'bst:max_depth': 3, 'bst:eta': 0.3, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
    watchlist  = [(d_cv,'eval'), (d_train,'train')]
    num_round = 2
    bst = xgb.train(params, d_train, num_round, watchlist)

    # this is prediction
    preds = bst.predict(d_test)
    labels = d_test.get_label()
    print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))



