from sklearn.base import BaseEstimator, is_classifier
from abc import abstractmethod
import numpy as np
import os
import random
import json
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from joblib import dump, load

from autokaggle.preprocessor import TabularPreprocessor
from autokaggle.utils import rand_temp_folder_generator, ensure_dir, write_json, read_json
from lightgbm import LGBMClassifier, LGBMRegressor
from autokaggle.config import Config, classification_hspace, regression_hspace
from sklearn.model_selection import StratifiedKFold, KFold
import hyperopt
from hyperopt import tpe, hp, fmin, Trials, STATUS_OK, STATUS_FAIL
from sklearn.model_selection import cross_val_score
from autokaggle.ensemblers import RankedEnsembler, StackingEnsembler
from imblearn.over_sampling import SMOTE, SMOTENC
import collections


# TODO: Further clean the design of this file
class AutoKaggle(BaseEstimator):
    pipeline = None
    hparams = None

    def __init__(self, config=None, **kwargs):
        """
        Initialization function for tabular supervised learner.
        """
        self.is_trained = False
        self.config = config if config else Config()
        self.config.update(kwargs)
        if not self.config.path:
            self.config.path = rand_temp_folder_generator()

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

        # Extract or read data info
        self.config.data_info = data_info if data_info is not None else self.extract_data_info(x)

        if self.config.verbose:
            print('DATA_INFO: {}'.format(self.config.data_info))
            print('#TIME features: {}'.format(sum(self.config.data_info == 'TIME')))
            print('#NUM features: {}'.format(sum(self.config.data_info == 'NUM')))
            print('#CAT features: {}'.format(sum(self.config.data_info == 'CAT')))
        
        if x.shape[1] == 0:
            raise ValueError("No feature exist!")

        x, y = self.resample(x, y)

        if self.config.objective == 'classification':
            n_classes = len(set(y))
            self.config.objective = 'binary' if n_classes == 2 else 'multiclass'

        # self.pipeline = AutoPipe(LGBMClassifier, {}, {}, self.config)
        prep_space = {'prep': hp.choice('data_source', ['a', 'b'])}
        self.pipeline = self.get_best_pipeline(self.search(x, y, prep_space, self.hparams))
        self.pipeline.fit(x, y)
        self.is_trained = True

    def predict(self, x_test):
        """
        This function should provide predictions of labels on (test) data.
        The function predict eventually can return probabilities or continuous values.
        """
        y = self.pipeline.predict(x_test)
        if y is None:
            raise ValueError("Tabular predictor does not exist")
        return y

    def predict_proba(self, x_test):
        y = self.pipeline.predict_proba(x_test)
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
        self.pipeline.fit(x_train, y_train)

    def resample(self, x, y):
        if self.config.balance_class_dist:
            x, y = SMOTE(sampling_strategy=self.config.resampling_strategy).fit_resample(x, y)
        while x.shape[0] < 60:
            x = np.concatenate([x, x], axis=0)
            y = np.concatenate([y, y], axis=0)
        return x, y

    def subsample(self, x, y, sample_percent):
        # TODO: Add way to balance the subsample
        # Set small sample for hyper-param search
        if x.shape[0] > 600:
            grid_train_percentage = max(600.0 / x.shape[0], sample_percent)
        else:
            grid_train_percentage = 1
        grid_n = int(x.shape[0] * grid_train_percentage)
        idx = random.sample(list(range(x.shape[0])), grid_n)
        grid_train_x, grid_train_y = x[idx, :], y[idx]
        return grid_train_x, grid_train_y

    def search(self, x, y, prep_space, model_space):
        grid_train_x, grid_train_y = self.subsample(x, y, sample_percent=self.config.subsample_ratio)
        score_metric, skf = self.get_skf(self.config.cv_folds)

        def objective_func(params):
            model_class = params['estimator']['model']
            m_params = params['estimator']['param']
            p_params = params['prep']
            pipeline = AutoPipe(model_class=model_class, m_params=m_params, p_params=p_params, config=self.config)
            try:
                eval_score = cross_val_score(pipeline, grid_train_x, grid_train_y, scoring=score_metric, cv=skf).mean()
                status = STATUS_OK
            except ValueError:
                eval_score = float('-inf')
                status = STATUS_FAIL
            if self.config.verbose:
                print("CV Score:", eval_score)
                print("\n=================")
            loss = 1 - eval_score if status == STATUS_OK else float('inf')
            return {'loss': loss, 'status': status, 'model_class': model_class, 'm_params': m_params,
                    'p_params': p_params}

        trials = Trials()
        search_space = {'prep': prep_space, 'estimator': model_space}
        _ = fmin(objective_func, search_space, algo=hyperopt.rand.suggest, trials=trials,
                 max_evals=self.config.search_iter)
        return trials

    def get_best_pipeline(self, trials):
        if self.config.use_ensembling:
            best_pipeline = self.setup_ensemble(trials)
        else:
            opt = trials.best_trial['result']
            best_pipeline = AutoPipe(opt['model_class'], opt['m_params'], opt['p_params'], self.config)
            if self.config.verbose:
                print("The best hyperparameter setting found:")
                print(opt)
        return best_pipeline

    @abstractmethod
    def get_skf(self, folds):
        pass

    def pick_diverse_estimators(self, trial_list, k):
        groups = collections.defaultdict(list)

        for obj in trial_list:
            groups[obj['model_class']].append(obj)
        estimator_list = []
        idx, j = 0, 0
        while idx < k:
            for grp in groups.values():
                if j < len(grp):
                    est = AutoPipe(grp[j]['model_class'], grp[j]['m_params'], grp[j]['p_params'], self.config)
                    estimator_list.append(est)
                    idx += 1
            j += 1
        return estimator_list

    def setup_ensemble(self, trials):
        # Filter the unsuccessful hparam spaces i.e. 'loss' == float('inf')
        best_trials = [t for t in trials.results if t['loss'] != float('inf')]
        best_trials = sorted(best_trials, key=lambda k: k['loss'], reverse=False)

        self.config.num_estimators_ensemble = min(self.config.num_estimators_ensemble, len(best_trials))

        if self.config.random_ensemble:
            np.random.shuffle(best_trials)

        if self.config.diverse_ensemble:
            estimator_list = self.pick_diverse_estimators(best_trials, self.config.num_estimators_ensemble)
        else:
            estimator_list = []
            for i in range(self.config.num_estimators_ensemble):
                est = AutoPipe(best_trials[i]['model_class'], best_trials[i]['m_params'], best_trials[i]['p_params'],
                               self.config)
                estimator_list.append(est)

        if self.config.ensemble_strategy == 'stacking':
            best_estimator_ = StackingEnsembler(estimator_list, config=self.config)
        else:
            best_estimator_ = RankedEnsembler(estimator_list, config=self.config)
        return best_estimator_

    @staticmethod
    def extract_data_info(raw_x):
        """
        This function extracts the data info automatically based on the type of each feature in raw_x.

        Args:
            raw_x: a numpy.ndarray instance containing the training data.
        """
        data_info = []
        row_num, col_num = raw_x.shape
        for col_idx in range(col_num):
            try:
                raw_x[:, col_idx].astype(np.float)
                data_info.append('NUM')
            except:
                data_info.append('CAT')
        return np.array(data_info)


class AutoKaggleClassifier(AutoKaggle):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.config.objective = 'classification'
        self.hparams = hp.choice('classifier', [classification_hspace[m] for m in self.config.classification_models])

    def get_skf(self, folds):
        if self.config.objective == 'binary':
            score_metric = 'roc_auc'
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.config.random_state)
        else:
            score_metric = 'f1_weighted'
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.config.random_state)
        return score_metric, skf


class AutoKaggleRegressor(AutoKaggle):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.config.objective = 'regression'
        self.hparams = hp.choice('regressor', [regression_hspace[m] for m in self.config.regression_models])

    def get_skf(self, folds):
        return 'neg_mean_squared_error', KFold(n_splits=folds, shuffle=True, random_state=self.config.random_state)


class AutoPipe(BaseEstimator):
    prep = None
    model = None
    config = None
    m_params = None
    p_params = None
    model_class = None

    def __init__(self, model_class, m_params, p_params, config):
        self.config = config
        self.m_params = m_params
        self.p_params = p_params
        self.model_class = model_class
        self._estimator_type = 'classifier' if is_classifier(model_class) else 'regressor'

    def fit(self, x, y):
        self.prep = TabularPreprocessor(self.config)
        self.model = self.model_class(**self.m_params)
        x = self.prep.fit_transform(x, y)
        self.model.fit(x, y)

    def predict(self, x):
        x = self.prep.transform(x)
        return self.model.predict(x)

    def predict_proba(self, x):
        x = self.prep.transform(x)
        try:
            return self.model.predict_proba(x)
        except AttributeError:
            return self.model.predict(x)

    def decision_function(self, x):
        x = self.prep.transform(x)
        try:
            return self.model.decision_function(x)
        except AttributeError:
            raise AttributeError
