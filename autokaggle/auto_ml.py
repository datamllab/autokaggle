from sklearn.base import BaseEstimator, is_classifier
from abc import abstractmethod
import numpy as np
import os
import random
import json
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from joblib import dump, load

from autokaggle.preprocessor import Preprocessor
from autokaggle.utils import rand_temp_folder_generator, ensure_dir, write_json, read_json
from lightgbm import LGBMClassifier, LGBMRegressor
from autokaggle.config import Config, CLASSIFICATION_PREP_HPARAM_SPACE, REGRESSION_PREP_HPARAM_SPACE, \
    REGRESSION_BASE_HPARAM_SPACE, CLASSIFICATION_BASE_HPARAM_SPACE, CLASSIFICATION_HPARAM_SPACE, REGRESSION_HPARAM_SPACE
from sklearn.model_selection import StratifiedKFold, KFold
import hyperopt
from hyperopt import tpe, hp, fmin, Trials, STATUS_OK, STATUS_FAIL
from sklearn.model_selection import cross_val_score
from autokaggle.ensemblers import RankedEnsemblingModel, StackedEnsemblingModel
from imblearn.over_sampling import SMOTE, SMOTENC
import collections


class AutoKaggle(BaseEstimator):
    """ Automated Machine Learning system class.

        AutoKaggle implements an end to end automated ML system. It initiates and searches for the optimum ML pipeline.
        The user can use it with the simple `fit()` and  `predict()` methods like Sci-kit learn estimators.
        The user can specify various parameters controlling different components of the system.
        # Arguments
            path: String. OS path for storing temporary model parameters.
            verbose: Bool. Defines the verbosity of the logging.
            time_limit: Int. Time budget for performing search and fit pipeline.
            use_ensembling: Bool. Defines whether to use an ensemble of models
            num_estimators_ensemble: Int. Maximum number of estimators to be used in an ensemble
            ensemble_strategy: String. Strategy to ensemble models
            ensemble_method: String. Aggregation method if ensemble_strategy is set to ranked_ensembling
            random_ensemble: Bool. Whether the ensembling estimators are picked randomly.
            diverse_ensemble: Bool. Whether estimators from different families are picked.
            ensembling_search_iter: Int. Search iterations for ensembling hyper-parameter search
            search_algo: String. Search strategy for hyper-parameter search.
            search_iter: Int. Number of iterations used for hyper-parameter search.
            cv_folds: Int. Number of Cross Validation folds.
            subsample_ratio: Percent of subsample used for for hyper-parameter search.
            data_info: list(String). Lists the datatypes of each feature column.
            stack_probabilities: Bool. Whether to use class probabilities in ensembling.
            upsample_classes: Bool. Whether to upsample less represented classes
            num_p_hparams: Int. Number of preprocessor search spaces.
    """

    def __init__(self, path=None, verbose=True, time_limit=None, use_ensembling=True,
                 num_estimators_ensemble=50, ensemble_strategy='stacking', ensemble_method='max_voting',
                 search_iter=500, cv_folds=3, subsample_ratio=0.1, random_ensemble=False, diverse_ensemble=True,
                 stack_probabilities=False, data_info=None, upsample_classes=False, ensembling_search_iter=10,
                 search_algo='random', num_p_hparams=10):
        self.is_trained = False
        if not path:
            path = rand_temp_folder_generator()
        self.config = Config(path=path, verbose=verbose, time_limit=time_limit, use_ensembling=use_ensembling,
                             num_estimators_ensemble=num_estimators_ensemble, ensemble_strategy=ensemble_strategy,
                             ensemble_method=ensemble_method, search_iter=search_iter, cv_folds=cv_folds,
                             subsample_ratio=subsample_ratio, random_ensemble=random_ensemble,
                             diverse_ensemble=diverse_ensemble, stack_probabilities=stack_probabilities,
                             data_info=data_info, upsample_classes=upsample_classes,
                             ensembling_search_iter=ensembling_search_iter, search_algo=search_algo,
                             num_p_hparams=num_p_hparams)
        self.pipeline = None
        self.m_hparams = None
        self.m_hparams_base = None
        self.p_hparams_base = None

    def fit(self, x, y, time_limit=None, data_info=None):
        """ Train an autoML system.
        # Arguments
            x: A numpy.ndarray instance containing the training data.
            y: training label vector.
            time_limit: remaining time budget.
            data_info: meta-features of the dataset, which is an numpy.ndarray describing the feature type of each
             column in raw_x. The feature type include: 'TIME' for temporal feature, 'NUM' for other numerical feature,
             and 'CAT' for categorical feature.
        # Returns
            None
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
        # Search the top preprocessing setting
        trials = self.search(x, y, self.p_hparams_base, self.m_hparams_base)
        p_hparams = self.get_top_prep(trials, self.config.num_p_hparams)
        # Search the best pipelines
        trials = self.search(x, y, p_hparams, self.m_hparams_base)
        self.pipeline = self.get_best_pipeline(trials)
        # Fit data
        self.pipeline.fit(x, y)
        self.is_trained = True

    def predict(self, x_test):
        """ Generate prediction on the test data for the given task.
        # Arguments
            x_test: A numpy.ndarray instance containing the test data.
        # Returns
            A numpy array for the predictions on the x_test.
        This function provides predictions of labels on (test) data.
        """
        y = self.pipeline.predict(x_test)
        if y is None:
            raise ValueError("Tabular predictor does not exist")
        return y

    def predict_proba(self, x_test):
        """ Predict label probabilities on the test data for the given classification task.
        # Arguments
            x_test: A numpy.ndarray instance containing the test data.
        # Returns
            A numpy array for the prediction probabilities on the x_test.
        The function returns predicted probabilities for every class label.
        """
        y = self.pipeline.predict_proba(x_test)
        if y is None:
            raise ValueError("Tabular predictor does not exist")
        return y

    def evaluate(self, x_test, y_test):
        """ Predict label probabilities on the test data for the given classification task.
        # Arguments
            x_test: A numpy.ndarray instance containing the training data.
            y_test: A numpy array with ground truth labels for the test data
        # Returns
            An evaluation score based on the task type.
        """
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

    def resample(self, x, y):
        """ Up-samples the input data
        # Arguments
            x: A numpy array for features
            y: A numpy array for target
        # Returns
            Up-sampled version of the dataset
        """
        if self.config.upsample_classes:
            x, y = SMOTE(sampling_strategy=self.config.resampling_strategy).fit_resample(x, y)
        while x.shape[0] < 60:
            x = np.concatenate([x, x], axis=0)
            y = np.concatenate([y, y], axis=0)
        return x, y

    def subsample(self, x, y, sample_percent):
        """ Takes a sub-sample of the input data, for the hyper-parameter search.
        # Arguments
            x: A numpy array for features
            y: A numpy array for target
            sample_percent: Minimum percentage of the  data to be maintained
        # Returns
            Down-sampled dataset
        """
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
        """ Do hyper-parameter search to find optimal machine learning pipeline.
        # Arguments
            x: A numpy array for features
            y: A numpy array for target
            prep_space: Hyper-parameter search space for preprocessors
            model_space: Hyper-parameter search space for estimators
        # Returns
            List of hyper-parameter trials
        """
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
            except ValueError as e:
                print(e)
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
        _ = fmin(objective_func, search_space, algo=self.config.search_algo, trials=trials,
                 max_evals=self.config.search_iter, rstate=np.random.RandomState(self.config.random_state))
        return trials

    def get_best_pipeline(self, trials):
        """ Finds the optimal pipeline from the given list of search trials.
        # Arguments
            trials: List of hyper-parameter search trials
        # Returns
            Optimal pipeline based on the given list of trials
        """
        if self.config.use_ensembling:
            best_pipeline = self.setup_ensemble(trials)
        else:
            opt = trials.best_trial['result']
            best_pipeline = AutoPipe(opt['model_class'], opt['m_params'], opt['p_params'], self.config)
            if self.config.verbose:
                print("The best hyperparameter setting found:")
                print(opt)
        return best_pipeline

    @staticmethod
    def get_top_prep(trials, n):
        """ Find the list of top N preprocessor settings.
        # Arguments
            trials: List of hyper-parameter search trials
            n: Maximum number of preprocessor settings required
        # Returns
            List of the top N optimal preprocessor settings.
        """
        best_trials = [t for t in trials.results if t['loss'] != float('inf')]
        best_trials = sorted(best_trials, key=lambda k: k['loss'], reverse=False)
        top_p_hparams, count = [], 0
        for trial in best_trials:
            if trial['p_params'] not in top_p_hparams:
                top_p_hparams.append(trial)
                count += 1
                if count > n:
                    break

        return hp.choice('p_params', top_p_hparams)

    @abstractmethod
    def get_skf(self, folds):
        """ Get the scoring metric and the cross validation folds for evaluation.
        # Arguments
            folds: NUmber of cross validation folds
        # Returns
            Scoring metric and cross validation folds.
        """
        pass

    def pick_diverse_estimators(self, trial_list):
        """ Selects the best hyper-parameter settings from each estimator family.
        # Arguments
            trial_list: List of the hyper-parameter search trials.
        # Returns
            List of top hyper-parameter spaces equally selected from each estimator family.
        """
        groups = collections.defaultdict(list)

        for obj in trial_list:
            groups[obj['model_class']].append(obj)
        estimator_list = []
        idx, j = 0, 0
        while idx < self.config.num_estimators_ensemble:
            for grp in groups.values():
                if j < len(grp):
                    est = AutoPipe(grp[j]['model_class'], grp[j]['m_params'], grp[j]['p_params'], self.config)
                    estimator_list.append(est)
                    idx += 1
            j += 1
        return estimator_list

    def setup_ensemble(self, trials):
        """ Generates the optimal ensembling estimator based on the given setting.
        # Arguments
            trials: List of the hyper-parameter search trials.
        # Returns
            An ensembling estimator to be trained using the base estimators picked from trials.
        """
        # Filter the unsuccessful hparam spaces i.e. 'loss' == float('inf')
        best_trials = [t for t in trials.results if t['loss'] != float('inf')]
        best_trials = sorted(best_trials, key=lambda k: k['loss'], reverse=False)

        self.config.num_estimators_ensemble = min(self.config.num_estimators_ensemble, len(best_trials))

        if self.config.random_ensemble:
            np.random.shuffle(best_trials)

        if self.config.diverse_ensemble:
            estimator_list = self.pick_diverse_estimators(best_trials)
        else:
            estimator_list = []
            for i in range(self.config.num_estimators_ensemble):
                est = AutoPipe(best_trials[i]['model_class'], best_trials[i]['m_params'], best_trials[i]['p_params'],
                               self.config)
                estimator_list.append(est)

        if self.config.ensemble_strategy == 'stacking':
            best_estimator_ = StackedEnsemblingModel(estimator_list, config=self.config)
        else:
            best_estimator_ = RankedEnsemblingModel(estimator_list, config=self.config)
        return best_estimator_

    @staticmethod
    def extract_data_info(raw_x):
        """
        Extracts the data info automatically based on the type of each feature in raw_x.
        # Arguments
            raw_x: a numpy.ndarray instance containing the training data.
        # Returns
            A list of data-types for each feature in the data.
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


class Classifier(AutoKaggle):
    """ Extends AutoKaggle for Classification.

        Extends the AutoKaggle specific to the classification requirements.
        # Arguments
            path: String. OS path for storing temporary model parameters.
            verbose: Bool. Defines the verbosity of the logging.
            time_limit: Int. Time budget for performing search and fit pipeline.
            use_ensembling: Bool. Defines whether to use an ensemble of models
            num_estimators_ensemble: Int. Maximum number of estimators to be used in an ensemble
            ensemble_strategy: String. Strategy to ensemble models
            ensemble_method: String. Aggregation method if ensemble_strategy is set to ranked_ensembling
            random_ensemble: Bool. Whether the ensembling estimators are picked randomly.
            diverse_ensemble: Bool. Whether estimators from different families are picked.
            ensembling_search_iter: Int. Search iterations for ensembling hyper-parameter search
            search_algo: String. Search strategy for hyper-parameter search.
            search_iter: Int. Number of iterations used for hyper-parameter search.
            cv_folds: Int. Number of Cross Validation folds.
            subsample_ratio: Percent of subsample used for for hyper-parameter search.
            data_info: list(String). Lists the datatypes of each feature column.
            stack_probabilities: Bool. Whether to use class probabilities in ensembling.
            upsample_classes: Bool. Whether to upsample less represented classes
            num_p_hparams: Int. Number of preprocessor search spaces.
    """
    def __init__(self, path=None, verbose=True, time_limit=None, use_ensembling=True,
                 num_estimators_ensemble=50, ensemble_strategy='stacking', ensemble_method='max_voting',
                 search_iter=500, cv_folds=3, subsample_ratio=0.1, random_ensemble=False, diverse_ensemble=True,
                 stack_probabilities=False, data_info=None, upsample_classes=False, ensembling_search_iter=10,
                 search_algo='random', num_p_hparams=10):
        super().__init__(path=path, verbose=verbose, time_limit=time_limit, use_ensembling=use_ensembling,
                         num_estimators_ensemble=num_estimators_ensemble, ensemble_strategy=ensemble_strategy,
                         ensemble_method=ensemble_method, search_iter=search_iter, cv_folds=cv_folds,
                         subsample_ratio=subsample_ratio, random_ensemble=random_ensemble, diverse_ensemble=diverse_ensemble,
                         stack_probabilities=stack_probabilities, data_info=data_info,
                         upsample_classes=upsample_classes, ensembling_search_iter=ensembling_search_iter,
                         search_algo=search_algo, num_p_hparams=num_p_hparams)
        self.config.objective = 'classification'
        self.m_hparams = hp.choice('classifier', [CLASSIFICATION_HPARAM_SPACE[m] for m in
                                                  self.config.classification_models])
        self.m_hparams_base = hp.choice('classifier',
                                        [CLASSIFICATION_BASE_HPARAM_SPACE[m] for m in
                                         self.config.classification_models])
        self.p_hparams_base = CLASSIFICATION_PREP_HPARAM_SPACE

    def get_skf(self, folds):
        """
            See the base class.
        """
        if self.config.objective == 'binary':
            score_metric = 'roc_auc'
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.config.random_state)
        else:
            score_metric = 'f1_weighted'
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.config.random_state)
        return score_metric, skf


class Regressor(AutoKaggle):
    """ Extends AutoKaggle for Regression

        Extends the AutoKaggle specific to the regression requirements.
        # Arguments
            path: String. OS path for storing temporary model parameters.
            verbose: Bool. Defines the verbosity of the logging.
            time_limit: Int. Time budget for performing search and fit pipeline.
            use_ensembling: Bool. Defines whether to use an ensemble of models
            num_estimators_ensemble: Int. Maximum number of estimators to be used in an ensemble
            ensemble_strategy: String. Strategy to ensemble models
            ensemble_method: String. Aggregation method if ensemble_strategy is set to ranked_ensembling
            random_ensemble: Bool. Whether the ensembling estimators are picked randomly.
            diverse_ensemble: Bool. Whether estimators from different families are picked.
            ensembling_search_iter: Int. Search iterations for ensembling hyper-parameter search
            search_algo: String. Search strategy for hyper-parameter search.
            search_iter: Int. Number of iterations used for hyper-parameter search.
            cv_folds: Int. Number of Cross Validation folds.
            subsample_ratio: Percent of subsample used for for hyper-parameter search.
            data_info: list(String). Lists the datatypes of each feature column.
            stack_probabilities: Bool. Whether to use class probabilities in ensembling.
            upsample_classes: Bool. Whether to upsample less represented classes
            num_p_hparams: Int. Number of preprocessor search spaces.
    """
    def __init__(self, path=None, verbose=True, time_limit=None, use_ensembling=True,
                 num_estimators_ensemble=50, ensemble_strategy='stacking', ensemble_method='max_voting',
                 search_iter=500, cv_folds=3, subsample_ratio=0.1, random_ensemble=False, diverse_ensemble=True,
                 stack_probabilities=False, data_info=None, upsample_classes=False, ensembling_search_iter=10,
                 search_algo='random', num_p_hparams=10):
        super().__init__(path=path, verbose=verbose, time_limit=time_limit, use_ensembling=use_ensembling,
                         num_estimators_ensemble=num_estimators_ensemble, ensemble_strategy=ensemble_strategy,
                         ensemble_method=ensemble_method, search_iter=search_iter, cv_folds=cv_folds,
                         subsample_ratio=subsample_ratio, random_ensemble=random_ensemble,
                         diverse_ensemble=diverse_ensemble,
                         stack_probabilities=stack_probabilities, data_info=data_info,
                         upsample_classes=upsample_classes, ensembling_search_iter=ensembling_search_iter,
                         search_algo=search_algo, num_p_hparams=num_p_hparams)
        self.config.objective = 'regression'
        self.m_hparams = hp.choice('regressor', [REGRESSION_HPARAM_SPACE[m] for m in self.config.regression_models])
        self.m_hparams_base = hp.choice('regressor',
                                        [REGRESSION_BASE_HPARAM_SPACE[m] for m in self.config.classification_models])
        self.p_hparams_base = REGRESSION_PREP_HPARAM_SPACE

    def get_skf(self, folds):
        """
            See the base class.
        """
        return 'neg_mean_squared_error', KFold(n_splits=folds, shuffle=True, random_state=self.config.random_state)


class AutoPipe(BaseEstimator):
    """ Implements a machine learning pipeline.

        Implements a machine learning pipeline with preprocessor and estimator. A user can call fit(), and predict()
        methods on it. It is used as a search unit in AutoKaggle's hyeper-parameter search.
        # Arguments
            config: Config. Defines the configuration of various components of the pipeline.
            m_params: Dict. Hyper-parameter search space for estimator.
            p_params: Dict. Hyper-parameter search space for preprocessor.
            model_class: Estimator. Class name of the estimator used in the pipeline.
            _estimator_type: String. Denotes if the estimator is 'classifier' or 'regressor'
            prep: Preprocessor. Instance of the Preprocessor class, which does basic feature preprocessing and feature
            engineering
            model: Estimator. Instance of the estimator class which learns a machine learning model and predicts on the
            given data.
    """
    def __init__(self, model_class, m_params, p_params, config):
        self.prep = None
        self.model = None
        self.config = config
        self.m_params = m_params
        self.p_params = p_params
        self.model_class = model_class
        self._estimator_type = 'classifier' if is_classifier(model_class) else 'regressor'

    def fit(self, x, y):
        """ Trains the given pipeline.
        # Arguments
            x: A numpy.ndarray instance containing the training data.
            y: training label vector.
        # Returns
            None
        """
        self.prep = Preprocessor(self.config, self.p_params)
        self.model = self.model_class(**self.m_params)
        x = self.prep.fit_transform(x, y)
        self.model.fit(x, y)

    def predict(self, x):
        """ Generate prediction on the test data for the given task.
        # Arguments
            x: A numpy.ndarray instance containing the test data.
        # Returns
            A numpy array for the predictions on the x.
        This function provides predictions of labels on (test) data.
        """
        x = self.prep.transform(x)
        return self.model.predict(x)

    def predict_proba(self, x):
        """ Predict label probabilities on the test data for the given classification task.
        # Arguments
            x: A numpy.ndarray instance containing the test data.
        # Returns
            A numpy array for the prediction probabilities on the x.
        The function returns predicted probabilities for every class label.
        """
        x = self.prep.transform(x)
        try:
            return self.model.predict_proba(x)
        except AttributeError:
            return self.model.predict(x)

    def decision_function(self, x):
        """ Returns the decision function learned by the estimator.
        # Arguments
            x: A numpy.ndarray instance containing the test data.
        # Returns
            Decision function learned by the estimator.
        This is used by the scorers to evaluate the pipeline.
        """
        x = self.prep.transform(x)
        try:
            return self.model.decision_function(x)
        except AttributeError:
            raise AttributeError
