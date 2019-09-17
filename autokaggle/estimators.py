from sklearn.base import BaseEstimator
from abc import abstractmethod
import numpy as np
import os
import random
import json

from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, make_scorer
from joblib import dump, load

from autokaggle.utils import rand_temp_folder_generator, ensure_dir, write_json, read_json
from autokaggle.ensemblers import RankedEnsembler, StackingEnsembler
from autokaggle.config import Config, classification_hspace, regression_hspace
import hyperopt
from hyperopt import tpe, hp, fmin, space_eval, Trials, STATUS_OK
from imblearn.over_sampling import SMOTE, SMOTENC


class TabularEstimator(BaseEstimator):
    def __init__(self, config=Config(), **kwargs):
        """
        Initialization function for tabular supervised learner.
        """
        self.config = config
        self.best_estimator_ = None
        self.hparams = None

    def fit(self, x, y):
        if self.config.objective == 'classification':
            n_classes = len(set(y))
            self.config.objective = 'binary' if n_classes == 2 else 'multiclass'
        # x, y = self.resample(x, y)
        self.best_estimator_, _ = self.search(x, y)
        self.best_estimator_.fit(x, y)
        self.save_model()
    
    def predict(self, x, y=None):
        y_pred = self.best_estimator_.predict(x, )
        return y_pred

    def predict_proba(self, x, y=None):
        try:
            y_pred = self.best_estimator_.predict_proba(x, )
        except:
            y_pred = self.best_estimator_.predict(x, )
        return y_pred

    def resample(self, X, y):
        return SMOTE(sampling_strategy=self.config.resampling_strategy).fit_resample(X, y)

    @staticmethod
    def subsample(x, y, sample_percent):
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

    def search(self, x, y):
        grid_train_x, grid_train_y = self.subsample(x, y, sample_percent=self.config.subsample_ratio)
        score_metric, skf = self.get_skf(self.config.cv_folds)

        def objective_func(args):
            clf = args['model'](**args['param'])
            try:
                eval_score = cross_val_score(clf, grid_train_x, grid_train_y, scoring=score_metric, cv=skf).mean()
            except ValueError:
                eval_score = 0
            if self.config.verbose:
                print("CV Score:", eval_score)
                print("\n=================")
            return {'loss': 1 - eval_score, 'status': STATUS_OK, 'space': args}

        trials = Trials()
        best = fmin(objective_func, self.hparams, algo=hyperopt.rand.suggest, trials=trials,
                    max_evals=self.config.search_iter)

        if self.config.use_ensembling:
            best_estimator_ = self.setup_ensemble(trials)
        else:
            opt = space_eval(self.hparams, best)
            best_estimator_ = opt['model'](**opt['param'])
            if self.config.verbose:
                print("The best hyperparameter setting found:")
                print(opt)
        return best_estimator_, trials
            
    @abstractmethod
    def save_model(self):
        pass
    
    @abstractmethod
    def get_skf(self, folds):
        pass

    def setup_ensemble(self, trials):
        best_trials = sorted(trials.results, key=lambda k: k['loss'], reverse=False)
        # Filter the unsuccessful hparam spaces i.e. 'loss' == 1
        best_trials = [t for t in best_trials if t['loss'] < 1]
        self.config.num_estimators_ensemble = min(self.config.num_estimators_ensemble, len(best_trials))
        if self.config.random_ensemble:
            np.random.shuffle(best_trials)
        estimator_list = []
        for i in range(self.config.num_estimators_ensemble):
            model_params = best_trials[i]['space']
            est = model_params['model'](**model_params['param'])
            estimator_list.append(est)

        if self.config.ensemble_strategy == 'ranked_ensembling':
            best_estimator_ = RankedEnsembler(estimator_list, ensemble_method=self.config.ensemble_method)
        elif self.config.ensemble_strategy == 'stacking':
            best_estimator_ = StackingEnsembler(estimator_list, objective=self.config.objective)
        else:
            best_estimator_ = RankedEnsembler(estimator_list, ensemble_method=self.config.ensemble_method)
        return best_estimator_
    
    
class Classifier(TabularEstimator):
    """Classifier class.
     It is used for tabular data classification.
    """ 
    def __init__(self, config=Config(), **kwargs):
        super().__init__(config, **kwargs)
        self.config.objective = 'classification'
        # TODO: add choice to the set of estimators
        self.hparams = hp.choice('classifier', [classification_hspace[m] for m in self.config.classification_models])

    def get_skf(self, folds):
        if self.config.objective == 'binary':
            score_metric = 'roc_auc'
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.config.random_state)
        else:
            score_metric = 'f1_weighted'
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.config.random_state)
        return score_metric, skf
    
    
class Regressor(TabularEstimator):
    """Regressor class.
    It is used for tabular data regression.
    """
    def __init__(self, config=Config(), **kwargs):
        super().__init__(config, **kwargs)
        self.config.objective = 'regression'
        # TODO: add choice to the set of estimators
        self.hparams = hp.choice('regressor', [regression_hspace[m] for m in self.config.regression_models])

    def get_skf(self, folds):
        return 'neg_mean_squared_error', KFold(n_splits=folds, shuffle=True, random_state=self.config.random_state)
