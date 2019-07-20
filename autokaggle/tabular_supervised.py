from sklearn.base import BaseEstimator
from tabular_preprocessor import TabularPreprocessor
from utils import rand_temp_folder_generator, ensure_dir, write_json, read_json
from abc import abstractmethod
import numpy as np
import os
import random
import json

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from joblib import dump, load


class AutoKagggle(BaseEstimator):
    def __init__(self, estimator_class=LgbmClassifier, path=None, verbose=True):
        """
        Initialization function for tabular supervised learner.
        """
        self.verbose = verbose
        self.is_trained = False
        self.objective = None
        self.tabular_preprocessor = None
        self.model = None
        self.estimator_class = estimator_class
        self.path = path if path is not None else rand_temp_folder_generator()
        ensure_dir(self.path)
        if self.verbose:
            print('Path:', path)
        self.time_limit = None

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

        if time_limit is None:
            time_limit = 24 * 60 * 60
        self.time_limit = time_limit
        
        if x.shape[1] == 0:
            raise ValueError("No feature exist!")

        while x.shape[0] < 60:
            x = np.concatenate([x, x], axis=0)
            y = np.concatenate([y, y], axis=0)
        
        # Init model and preprocessor
        self.model = self.estimator_class(verbose=self.verbose, path=self.path, time_limit=self.time_limit)
        self.tabular_preprocessor = TabularPreprocessor()
            
        # Fit Model and preprocessor
        x = self.tabular_preprocessor.fit(x, y, self.time_limit, data_info)
        self.model.fit(x, y)
        self.model.save_model()
        self.is_trained = True

        if self.verbose:
            print("The whole available data is: ")
            print("Real-FIT: dim(X)= [{:d}, {:d}]".format(x.shape[0], x.shape[1]))

    def predict(self, x_test):
        """
        This function should provide predictions of labels on (test) data.
        The function predict eventually casdn return probabilities or continuous values.
        """
        x_test = self.tabular_preprocessor.encode(x_test)
        y = self.model.predict(x_test, )
        if y is None:
            raise ValueError("Tabular predictor does not exist")
        return y

    def evaluate(self, x_test, y_test):
        if self.verbose:
            print('objective:', self.model.objective)
        y_pred = self.predict(x_test)
        results = None
        if self.model.objective == 'binary':
            results = roc_auc_score(y_test, y_pred)
        elif self.model.objective == 'multiclass':
            results = f1_score(y_test, y_pred, average='weighted')
        elif self.model.objective == 'regression':
            results = mean_squared_error(y_test, y_pred)
        return results

    def final_fit(self, x_train, y_train):
        x_train = self.tabular_preprocessor.encode(x_train)
        self.model.fit(x_train, y_train)

class TabularEstimator(BaseEstimator):
    def __init__(self, path=None, verbose=True, time_limit=None):
        """
        Initialization function for tabular supervised learner.
        """
        self.verbose = verbose
        self.path = path
        self.time_limit = time_limit
        self.objective = None
        self.hparams = read_json(self._default_hyperparams)
        self.clf = None
        self.estimator = None
    
    def fit(self, x, y):
        self.init_model(y)
        self.search(x, y)
        self.clf.fit(x, y)
        self.save_model()
    
    def predict(self, x, y=None):
        y = self.clf.predict(x, )
        return y
    
    def search(self, x, y, search_iter=40, folds=3):
        # Set small sample for hyper-param search
        if x.shape[0] > 600:
            grid_train_percentage = max(600.0 / x.shape[0], 0.1)
        else:
            grid_train_percentage = 1
        grid_n = int(x.shape[0] * grid_train_percentage)
        idx = random.sample(list(range(x.shape[0])), grid_n)
        grid_train_x, grid_train_y = x[idx, :], y[idx]
        
        if self.verbose: print(self.hparams)
        score_metric, skf = self.get_skf(folds)
        random_search = RandomizedSearchCV(self.estimator, param_distributions=self.hparams, n_iter=search_iter,
                                           scoring=score_metric,
                                           n_jobs=1, cv=skf, verbose=0, random_state=1001)

        random_search.fit(grid_train_x, grid_train_y)
        self.clf = random_search.best_estimator_

        return random_search.best_params_
            
    @abstractmethod
    def save_model(self):
        pass
    
    @abstractmethod
    def init_model(self, y):
        pass
    
    @abstractmethod
    def get_skf(self, folds):
        pass
    
    def __repr__(self):
        return "Estimator model"
class LGBMMixIn:
    _default_hyperparams = "lgbm_hp.json"
    
    def save_model(self):
        self.clf.booster_.save_model(self.save_filename)
    
    def get_feature_importance(self):
        if self.estimator:
            print('Feature Importance:')
            print(self.clf.feature_importances_)
            
class SklearnMixIn:
    
    def save_model(self):
        dump(self.clf, self.save_filename)
        
    def load_model(self):
        self.clf = load(self.save_filename)