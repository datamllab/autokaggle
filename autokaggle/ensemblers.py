from sklearn.base import BaseEstimator
from autokaggle.utils import rand_temp_folder_generator, ensure_dir, write_json, \
    read_json
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
from autokaggle.config import REGRESSION_HPARAM_SPACE, CLASSIFICATION_HPARAM_SPACE, \
    CLASSIFICATION_BASE_HPARAM_SPACE, \
    REGRESSION_BASE_HPARAM_SPACE


class EnsemblingModel:
    """ Base class for ensembling estimators.

        This class creates an ensembling estimator from a given list of estimators.
        The user can call fit() and predict() methods, similar to the scikit-learn
        estimators.

        # Arguments
            config: Config. Defines the configuration of various components of the
            autoML pipeline.
            estimator_list: List. List of the estimators, to be used for building an
            ensemble.
    """

    def __init__(self, estimator_list, config):
        self.config = config
        self.estimator_list = estimator_list

    @abstractmethod
    def fit(self, x, y):
        """ Trains the ensemble of estimators on the training data.
        # Arguments
            X: A numpy array instance containing the training data.
        # Returns
            None
        """
        pass

    @abstractmethod
    def predict(self, x):
        """ Generate prediction on the test data for the given task.
        # Arguments
            X: A numpy array instance containing the test data.
        # Returns
            A numpy array for the predictions on the x_test.
        This function provides predicts on the input data using the ensemble of
        estimators.
        """
        pass


class RankedEnsemblingModel(EnsemblingModel):
    """ Implements ensembling using ranking based methods.

        This class implements randing based ensembling using ensembling methods
        amongst: ('mean', 'median', 'max' and 'majority_voting')
    """

    def fit(self, x, y):
        for est in self.estimator_list:
            est.fit(x, y)

    def predict(self, x):
        predictions = np.zeros((len(x), len(self.estimator_list)))
        for i, est in enumerate(self.estimator_list):
            predictions[:, i] = est.predict(x)

        if self.config.ensemble_method == 'median':
            return np.median(predictions, axis=1)
        elif self.config.ensemble_method == 'mean':
            return np.mean(predictions, axis=1)
        elif self.config.ensemble_method == 'max':
            return np.max(predictions, axis=1)
        elif self.config.ensemble_method == 'min':
            return np.min(predictions, axis=1)
        elif self.config.ensemble_method == 'max_voting':
            return stats.mode(predictions, axis=1)[0]


class StackedEnsemblingModel(EnsemblingModel):
    """ Implements a stacking based ensembling estimator.

        This class creates an ensembling estimator using stacking. It trains an
        Light-GBM model on the predictions of the base estimator.

        # Arguments
            stacking_estimator: LightGBM estimator. Meta-learning algorithm for the
            stacking estimator.
    """

    def __init__(self, estimator_list, config):
        super().__init__(estimator_list, config)
        self.stacking_estimator = None

        if self.config.objective == 'regression':
            self.hparams = hp.choice('regressor',
                                     [REGRESSION_BASE_HPARAM_SPACE['lgbm']])
            self.config.stack_probabilities = False
        else:
            self.hparams = hp.choice('classifier',
                                     [CLASSIFICATION_BASE_HPARAM_SPACE['lgbm']])

    def get_model_predictions(self, X):
        """ Generate the combined predictions from the list of the estimators.
        # Arguments
            X: A numpy array instance containing the training/test data.
        # Returns
            A numpy array for the predictions of all the estimators in the list.
        """
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

    def fit(self, x, y):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
        for est in self.estimator_list:
            est.fit(x_train, y_train)
        predictions = self.get_model_predictions(x_val)
        self.stacking_estimator = self.search(predictions, y_val)
        self.stacking_estimator.fit(predictions, y_val)

    def search(self, x, y):
        """ Search function to find best hyper-param setting for the stacking model.
        # Arguments
            x: A numpy array instance containing the training data
        # Returns
            List of trials on various hyper-parameter settings.
        """
        score_metric, skf = self.get_skf(self.config.cv_folds)

        def objective_func(args):
            clf = args['model'](**args['param'])
            try:
                eval_score = cross_val_score(clf, x, y, scoring=score_metric,
                                             cv=skf).mean()
            except ValueError:
                eval_score = 0
            if self.config.verbose:
                print("Ensembling CV Score:", eval_score)
                print("\n=================")
            return {'loss': 1 - eval_score, 'status': STATUS_OK, 'space': args}

        trials = Trials()
        best = fmin(objective_func, self.hparams, algo=self.config.search_algo,
                    trials=trials,
                    max_evals=self.config.ensembling_search_iter,
                    rstate=np.random.RandomState(self.config.random_state))

        opt = space_eval(self.hparams, best)
        best_estimator_ = opt['model'](**opt['param'])
        if self.config.verbose:
            print("The best hyperparameter setting found for stacking:")
            print(opt)
        return best_estimator_

    def predict(self, x):
        predictions = self.get_model_predictions(x)
        return self.stacking_estimator.predict(predictions)

    def get_skf(self, folds):
        """ Get scoring metric and cross validation folds for the task type
        # Arguments
            folds: Number of cross validation folds
        # Returns
            Scoring metric and CV folds
        """
        if self.config.objective == 'binary':
            score_metric = 'roc_auc'
            skf = StratifiedKFold(n_splits=folds, shuffle=True,
                                  random_state=self.config.random_state)
        elif self.config.objective == 'multiclass':
            score_metric = 'f1_weighted'
            skf = StratifiedKFold(n_splits=folds, shuffle=True,
                                  random_state=self.config.random_state)
        elif self.config.objective == 'regression':
            score_metric = 'neg_mean_squared_error'
            skf = KFold(n_splits=folds, shuffle=True,
                        random_state=self.config.random_state)
        else:
            ValueError("Invalid objective")
        return score_metric, skf
