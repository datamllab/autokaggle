from sklearn.base import BaseEstimator
from abc import abstractmethod
import numpy as np
import os
import random
import json
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from joblib import dump, load

from autokaggle.preprocessor import TabularPreprocessor
from autokaggle.estimators import *
from autokaggle.utils import rand_temp_folder_generator, ensure_dir, write_json, read_json


class AutoKaggle(BaseEstimator):
    objective = None
    model = None

    def __init__(self, config=Config(), **kwargs):
        """
        Initialization function for tabular supervised learner.
        """
        self.is_trained = False
        self.config = config
        self.config.update(kwargs)
        self.config.objective = self.objective
        self.preprocessor = TabularPreprocessor(config)

    def fit(self, x, y, time_limit=None, data_info=None):
        """
        This function should train the model parameters.
        Args:
            x: A numpy.ndarray instance containing the training data.
            y: training label vector.
            time_limit: remaining time budget.
            data_info: meta-features of the dataset, which is an numpy.ndarray describing the
             feature type of each column in raw_x. The feature type include:
                     'TIME' for temporal feature, 'NUM' for other numerical feature,
                     and 'CAT' for categorical feature.
        Both inputs X and y are numpy arrays.
        If fit is called multiple times on incremental data (train, test1, test2, etc.)
        you should warm-start your training from the pre-trained model. Past data will
        NOT be available for re-training.
        """
        self.config.time_limit = time_limit if time_limit else 24 * 60 * 60
        
        if x.shape[1] == 0:
            raise ValueError("No feature exist!")

        while x.shape[0] < 60:
            x = np.concatenate([x, x], axis=0)
            y = np.concatenate([y, y], axis=0)
        
        # # Init model and preprocessor
        # self.model = self.estimator_class(verbose=self.verbose, path=self.path, time_limit=self.time_limit)
        # self.preprocessor = TabularPreprocessor()
            
        # Fit Model and preprocessor
        self.preprocessor.fit(x, y, data_info)
        x = self.preprocessor.transform(x)
        self.model.fit(x, y)
        self.is_trained = True

        if self.config.verbose:
            print("The whole available data is: ")
            print("Real-FIT: dim(X)= [{:d}, {:d}]".format(x.shape[0], x.shape[1]))

    def predict(self, x_test, predict_proba=False):
        """
        This function should provide predictions of labels on (test) data.
        The function predict eventually can return probabilities or continuous values.
        """
        x_test = self.preprocessor.transform(x_test)
        if predict_proba:
            y = self.model.predict_proba(x_test, )
        else:
            y = self.model.predict(x_test, )
        if y is None:
            raise ValueError("Tabular predictor does not exist")
        return y

    def evaluate(self, x_test, y_test):
        if self.config.verbose:
            print('objective:', self.config.objective)
        y_pred = self.predict(x_test)
        results = None
        if self.config.objective == 'binary':
            results = roc_auc_score(y_test, y_pred)
        elif self.config.objective == 'multiclass':
            results = f1_score(y_test, y_pred, average='weighted')
        elif self.config.objective == 'regression':
            results = mean_squared_error(y_test, y_pred)
        return results

    def final_fit(self, x_train, y_train):
        x_train = self.preprocessor.transform(x_train)
        self.model.fit(x_train, y_train)


class AutoKaggleClassifier(AutoKaggle):
    objective = 'classification'

    def __init__(self, config=Config(), **kwargs):
        super().__init__(config, **kwargs)
        self.model = Classifier(config)


class AutoKaggleRegressor(AutoKaggle):
    objective = 'regression'

    def __init__(self, config=Config(), **kwargs):
        super().__init__(config, **kwargs)
        self.model = Regressor(config)
