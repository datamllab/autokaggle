from sklearn.base import BaseEstimator
from autokaggle.utils import rand_temp_folder_generator, ensure_dir, write_json, read_json
from abc import abstractmethod
import numpy as np
import os
import random
import json
from statistics import mode

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from joblib import dump, load
from scipy import stats
from lightgbm import LGBMClassifier, LGBMRegressor
import collections
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import hyperopt
from hyperopt import tpe, hp, fmin, space_eval, Trials, STATUS_OK
from autokaggle.config import classification_hspace, regression_hspace


lgbm_classifier_params = {
    'n_estimators': hp.choice('n_estimators', [100, 150, 200]),
}

_classification_hspace = {
    'lgbm': {
        'model': LGBMClassifier,
        'param': lgbm_classifier_params
    },
}


class RankedEnsembler:
    def __init__(self, estimator_list, config):
        self.config = config
        self.ensemble_method = config.ensemble_method
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
    stacking_estimator = None

    def __init__(self, estimator_list, config):
        self.config = config
        self.estimator_list = estimator_list
        self.objective = config.objective
        if self.config.objective == 'regression':
            self.hparams = hp.choice('regressor', [regression_hspace[m] for m in ['lgbm']])
            self.config.stack_probabilities = False
        else:
            self.hparams = hp.choice('classifier', [_classification_hspace[m] for m in ['lgbm']])

    def get_model_predictions(self, X):
        if self.config.stack_probabilities:
            predictions = np.zeros((len(X), 1))
            for i, est in enumerate(self.estimator_list):
                try:
                    new = est.predict_proba(X)[:, :-1]
                    predictions = np.hstack([predictions, new])
                except AttributeError:
                    new = np.reshape(est.predict(X), (-1, 1))
                    predictions = np.hstack([predictions, new])
            predictions = predictions[:, 1:]
        else:
            predictions = np.zeros((len(X), len(self.estimator_list)))
            for i, est in enumerate(self.estimator_list):
                predictions[:, i] = est.predict(X)
        return predictions

    def fit(self, X, y):
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        for est in self.estimator_list:
            est.fit(x_train, y_train)
        predictions = self.get_model_predictions(x_val)
        self.stacking_estimator = self.search(predictions, y_val)
        self.stacking_estimator.fit(predictions, y_val)

    def search(self, x, y):
        score_metric, skf = self.get_skf(self.config.cv_folds)

        def objective_func(args):
            clf = args['model'](**args['param'])
            try:
                eval_score = cross_val_score(clf, x, y, scoring=score_metric, cv=skf).mean()
            except ValueError:
                eval_score = 0
            if self.config.verbose:
                print("Ensembling CV Score:", eval_score)
                print("\n=================")
            return {'loss': 1 - eval_score, 'status': STATUS_OK, 'space': args}

        trials = Trials()
        best = fmin(objective_func, self.hparams, algo=self.config.ensembling_algo, trials=trials,
                    max_evals=self.config.ensembling_search_iter)

        opt = space_eval(self.hparams, best)
        best_estimator_ = opt['model'](**opt['param'])
        if self.config.verbose:
            print("The best hyperparameter setting found for stacking:")
            print(opt)
        return best_estimator_

    def predict(self, X):
        predictions = self.get_model_predictions(X)
        return self.stacking_estimator.predict(predictions)

    def get_skf(self, folds):
        if self.config.objective == 'binary':
            score_metric = 'roc_auc'
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.config.random_state)
        elif self.config.objective == 'multiclass':
            score_metric = 'f1_weighted'
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.config.random_state)
        elif self.config.objective == 'regression':
            score_metric = 'neg_mean_squared_error'
            skf = KFold(n_splits=folds, shuffle=True, random_state=self.config.random_state)
        else:
            ValueError("Invalid objective")
        return score_metric, skf


class EnsembleSelection:
    indices_ = None
    weights_ = None

    def __init__(self, estimator_list, config):
        self.estimator_list = estimator_list
        self.config = config
        self.objective = config.objective
        self.indices_, self.weights_ = [], []
        self.ensemble_size = len(estimator_list)
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
