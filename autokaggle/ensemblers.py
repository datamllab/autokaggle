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
    def __init__(self, ensemble_size=25):
        self.ensemble_size = ensemble_size
        
    def fit(self, predictions, y_true):
        """Rich Caruana's ensemble selection method. (fast version)"""
        ensemble = []
        trajectory = []
        order = []

        for i in range(self.ensemble_size):
            scores = np.zeros((len(predictions)))
            s = len(ensemble)
            if s == 0:
                weighted_ensemble_prediction = np.zeros(predictions[0].shape)
            else:
                # Memory-efficient averaging!
                ensemble_prediction = np.zeros(ensemble[0].shape)
                for pred in ensemble:
                    ensemble_prediction += pred
                ensemble_prediction /= s

                weighted_ensemble_prediction = (s / float(s + 1)) * \
                                               ensemble_prediction
            fant_ensemble_prediction = np.zeros(weighted_ensemble_prediction.shape)
            for j, pred in enumerate(predictions):
                fant_ensemble_prediction[:,:] = weighted_ensemble_prediction + \
                                             (1. / float(s + 1)) * pred
                scores[j] = calculate_score(
                    solution=labels,
                    prediction=fant_ensemble_prediction,
                    task_type=self.task_type,
                    metric=self.metric,
                    all_scoring_functions=False)

            all_best = np.argwhere(scores == np.nanmin(scores)).flatten()
            best = np.random.RandomState.choice(all_best)
            ensemble.append(predictions[best])
            trajectory.append(scores[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = order
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros((self.ensemble_size,), dtype=float)
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights
        self._calculate_weights()
    
    def predict(self, predictions):
        return np.average(predictions, axis=1, weights=self.weights_)
