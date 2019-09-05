from sklearn.base import BaseEstimator
from autokaggle.preprocessor import TabularPreprocessor
from autokaggle.utils import rand_temp_folder_generator, ensure_dir, write_json, read_json
from abc import abstractmethod
import numpy as np
import os
import random
import json
from statistics import mode

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from joblib import dump, load
from scipy import stats
from lightgbm import LGBMClassifier, LGBMRegressor
import collections
from sklearn.model_selection import RandomizedSearchCV, cross_val_score


class RankedEnsembler:
    def __init__(self, estimator_list, ensemble_method='max_voting'):
        self.ensemble_method = ensemble_method
        self.estimators = estimator_list
        
    def fit(self, X, y):
        for est in self.estimators:
            est.fit(X, y)
    
    def predict(self, X):
        predictions = np.zeros((len(X), len(self.estimators)))
        for i, est in enumerate(self.estimators):
            predictions[:, i] = est.predict(X)

        if self.ensemble_method == 'median':
            return np.median(predictions, axis=1)
        elif self.ensemble_method == 'mean':
            return np.mean(predictions, axis=1)
        elif self.ensemble_method == 'max':
            return np.max(predictions, axis=1)
        elif self.ensemble_method == 'min':
            return np.min(predictions, axis=1)
        elif self.ensemble_method == 'max_voting':
            return stats.mode(predictions, axis=1)[0]


class StackingEnsembler:
    def __init__(self, estimator_list, objective):
        self.estimator_list = estimator_list
        self.objective = objective
        if self.objective == 'regression':
            self.stacking_estimator = LGBMRegressor(silent=False,
                                           verbose=-1,
                                           n_jobs=1,
                                           objective=self.objective)
        elif self.objective == 'multiclass' or self.objective == 'binary':
            self.stacking_estimator = LGBMClassifier(silent=False,
                                            verbose=-1,
                                            n_jobs=1,
                                            objective=self.objective)

    def fit(self, X, y):
        for est in self.estimator_list:
            est.fit(X, y)
        predictions = np.zeros((len(X), len(self.estimator_list)))
        for i, est in enumerate(self.estimator_list):
            predictions[:, i] = est.predict(X)
        self.stacking_estimator.fit(predictions, y)

    def predict(self, X):
        predictions = np.zeros((len(X), len(self.estimator_list)))
        for i, est in enumerate(self.estimator_list):
            predictions[:, i] = est.predict(X)
        return self.stacking_estimator.predict(predictions)


class EnsembleSelection:
    indices_ = None
    weights_ = None

    def __init__(self, estimator_list, objective, ensemble_size=25):
        self.estimator_list = estimator_list
        self.objective = objective
        self.indices_, self.weights_ = [], []
        self.ensemble_size = min(len(estimator_list), ensemble_size)
        if self.objective == 'regression':
            self.score_metric = 'neg_mean_squared_error'
            self.skf = KFold(n_splits=3, shuffle=True, random_state=1001)
        else:
            self.score_metric = 'neg_mean_squared_error'
            self.skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1001)

    def fit(self, X, y_true):
        """Rich Caruana's ensemble selection method. (fast version)"""
        ensemble = []
        trajectory = []
        order = []

        for i in range(self.ensemble_size):
            scores = np.zeros((len(self.estimator_list)))
            s = len(ensemble)
            if s == 0:
                weighted_ensemble_prediction = np.zeros(len(self.estimator_list))
            else:
                # Memory-efficient averaging!
                ensemble_prediction = np.zeros(ensemble[0].shape)
                for est in ensemble:
                    ensemble_prediction += est
                ensemble_prediction /= s

                weighted_ensemble_prediction = (s / float(s + 1)) * ensemble_prediction
            fant_ensemble_prediction = np.zeros(weighted_ensemble_prediction.shape)
            for j, est in enumerate(self.estimator_list):
                fant_ensemble_prediction[:,:] = weighted_ensemble_prediction + \
                                             (1. / float(s + 1)) * est
                scores[j] = cross_val_score(self.estimator_list[j], X, y_true, scoring=self.score_metric,
                                            cv=self.skf).mean()

            all_best = np.argwhere(scores == np.nanmin(scores)).flatten()
            best = np.random.RandomState.choice(all_best)
            ensemble.append(self.estimator_list[best])
            trajectory.append(scores[best])
            order.append(best)

            # Handle special case
            if len(self.estimator_list) == 1:
                break

        self.indices_ = order
        ensemble_members = collections.Counter(self.indices_)
        weights = np.zeros((self.ensemble_size,), dtype=float)
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights
    
    def predict(self, X):
        return np.average(X, axis=1, weights=self.weights_)
