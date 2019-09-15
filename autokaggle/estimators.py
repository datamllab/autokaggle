from sklearn.base import BaseEstimator
from abc import abstractmethod
import numpy as np
import os
import random
import json

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor,\
    ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, make_scorer
from joblib import dump, load

from autokaggle.utils import rand_temp_folder_generator, ensure_dir, write_json, read_json
from autokaggle.ensemblers import RankedEnsembler, StackingEnsembler
import hyperopt
from hyperopt import tpe, hp, fmin, space_eval, Trials, STATUS_OK
from imblearn.over_sampling import SMOTE, SMOTENC


# TODO: Way to change the default hparams
knn_classifier_params = {
    'n_neighbors': hp.choice('n_neighbors', range(2, 20)),
    'algorithm': hp.choice('algorithm', ['ball_tree', 'kd_tree', 'brute']),
    'leaf_size': hp.choice('leaf_size', range(5, 50)),
    'metric': hp.choice('metric', ["euclidean", "manhattan", "chebyshev", "minkowski"]),
    'p': hp.choice('p', range(1, 4)),
}

svc_params = {
    'C': hp.lognormal('C', 0, 1),
    'kernel': hp.choice('kernel', ['rbf', 'poly', 'linear', 'sigmoid']),
    'degree': hp.choice('degree', range(1, 6)),
    'gamma': hp.uniform('gamma', 0.001, 10000),
    'max_iter': 50000,
}

random_forest_classifier_params = {
    'criterion': hp.choice('criterion', ['entropy', 'gini']),
    'max_features': hp.uniform('max_features', 0, 1.0),
    'n_estimators': hp.choice('rf_n_estimators', range(50, 200)),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10))
}

lgbm_classifier_params = {
    'boosting_type': 'gbdt',
    'min_split_gain': 0.1,
    'subsample': 0.8,
    'num_leaves': 80,
    'colsample_bytree': hp.uniform('colsample_bytree', 0.4, 0.8),
    'min_child_weight': hp.choice('min_child_weight', range(1, 100)),
    'max_depth': hp.choice('max_depth', range(5, 10)),
    'n_estimators': hp.choice('n_estimators', range(50, 200)),
    'learning_rate': hp.lognormal('learning_rate', 0, 1),
}

adaboost_classifier_params = {
    'algorithm': hp.choice('algorithm_adaboost', ['SAMME.R', 'SAMME']),
    'n_estimators': hp.choice('n_estimators_adaboost', range(50, 200)),
    'learning_rate': hp.lognormal('learning_rate_adaboost', 0, 1),
}

extra_trees_regressor_params = {
    'n_estimators': hp.choice('n_estimators_extra_trees', range(50, 200)),
    'criterion': hp.choice('criterion_extra_trees', ['mse', 'friedman_mse', 'mae']),
    'max_features': hp.uniform('max_features_extra_trees', 0, 1.0),
    'min_samples_leaf': hp.choice('min_samples_leaf_extra_trees', range(1, 10)),
    'min_impurity_decrease': 0.0
}

ridge_params = {
    'fit_intercept': True,
    'tol': hp.loguniform('tol_ridge', 1e-5, 1e-1),
    'alpha': hp.loguniform('alpha_ridge', 1e-5, 10)
}

random_forest_regressor_params = {
    'criterion': hp.choice('criterion', ['mse', 'friedman_mse', 'mae']),
    'max_features': hp.uniform('max_features', 0, 1.0),
    'n_estimators': hp.choice('rf_n_estimators', range(50, 200)),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10))
}

lgbm_regressor_params = {
    'boosting_type': 'gbdt',
    'min_split_gain': 0.1,
    'subsample': 0.8,
    'num_leaves': 80,
    'colsample_bytree': hp.uniform('colsample_bytree', 0.4, 0.8),
    'min_child_weight': hp.choice('min_child_weight', range(1, 100)),
    'max_depth': hp.choice('max_depth', range(5, 10)),
    'n_estimators': hp.choice('n_estimators', range(50, 200)),
    'learning_rate': hp.lognormal('learning_rate', 0, 1),
}

adaboost_regressor_params = {
    'loss': hp.choice('loss_adaboost', ["linear", "square", "exponential"]),
    'n_estimators': hp.choice('n_estimators_adaboost', range(50, 200)),
    'learning_rate': hp.lognormal('learning_rate_adaboost', 0, 1),
}


class TabularEstimator(BaseEstimator):
    def __init__(self, path=None, verbose=True, time_limit=None, use_ensembling=False, num_estimators_ensemble=25,
                 ensemble_strategy='ranked_ensembling', ensemble_method='max_voting'):
        """
        Initialization function for tabular supervised learner.
        """
        self.verbose = verbose
        self.path = path
        self.time_limit = time_limit
        self.objective = None
        abs_cwd = os.path.split(os.path.abspath(__file__))[0]
        self.best_estimator_ = None
        self.use_ensembling = use_ensembling
        self.hparams = None
        self.num_estimators_ensemble = num_estimators_ensemble
        self.ensemble_strategy = ensemble_strategy
        self.ensemble_method = ensemble_method
    
    def fit(self, x, y):
        if self.objective == 'classification':
            n_classes = len(set(y))
            self.objective = 'binary' if n_classes == 2 else 'multiclass'
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
    
    @staticmethod
    def resample(X, y):
        return SMOTE(sampling_strategy='auto').fit_resample(X, y)

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

    def search(self, x, y, search_iter=100, folds=3, sample_percent=0.1):
        grid_train_x, grid_train_y = self.subsample(x, y, sample_percent=sample_percent)
        score_metric, skf = self.get_skf(folds)

        def objective_func(args):
            clf = args['model'](**args['param'])
            try:
                eval_score = cross_val_score(clf, grid_train_x, grid_train_y, scoring=score_metric, cv=skf).mean()
            except ValueError:
                eval_score = 0
            if self.verbose:
                print("CV Score:", eval_score)
                print("\n=================")
            return {'loss': 1 - eval_score, 'status': STATUS_OK, 'space': args}

        trials = Trials()
        best = fmin(objective_func, self.hparams, algo=hyperopt.rand.suggest, trials=trials, max_evals=search_iter)
        if self.use_ensembling:
            best_trials = sorted(trials.results, key=lambda k: k['loss'], reverse=False)
            estimator_list = []
            for i in range(self.num_estimators_ensemble):
                model_params = best_trials[i]['space']
                est = model_params['model'](**model_params['param'])
                estimator_list.append(est)
            if self.ensemble_strategy == 'ranked_ensembling':
                best_estimator_ = RankedEnsembler(estimator_list, ensemble_method=self.ensemble_method)
            elif self.ensemble_strategy == 'stacking':
                best_estimator_ = StackingEnsembler(estimator_list, objective=self.objective)
            else:
                best_estimator_ = RankedEnsembler(estimator_list, ensemble_method=self.ensemble_method)
        else:
            opt = space_eval(self.hparams, best)
            best_estimator_ = opt['model'](**opt['param'])
        return best_estimator_, trials
            
    @abstractmethod
    def save_model(self):
        pass
    
    @abstractmethod
    def get_skf(self, folds):
        pass
    
    
class Classifier(TabularEstimator):
    """Classifier class.
     It is used for tabular data classification.
    """ 
    def __init__(self, path=None, verbose=True, time_limit=None):
        super().__init__(path, verbose, time_limit)
        self.objective = 'classification'
        # TODO: add choice to the set of estimators
        self.hparams = hp.choice('classifier', [
            {'model': KNeighborsClassifier,
             'param': knn_classifier_params
             },
            {'model': SVC,
             'param': svc_params
             },
            {'model': RandomForestClassifier,
             'param': random_forest_classifier_params
             },
            {'model': LGBMClassifier,
             'param': lgbm_classifier_params
             },
            # {'model': AdaBoostClassifier,
            #  'param': adaboost_classifier_params
            #  }
        ])

    def get_skf(self, folds):
        if self.objective == 'binary':
            score_metric = 'roc_auc'
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
        else:
            score_metric = 'f1_weighted'
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
        return score_metric, skf
    
    
class Regressor(TabularEstimator):
    """Regressor class.
    It is used for tabular data regression.
    """
    def __init__(self, path=None, verbose=True, time_limit=None):
        super().__init__(path, verbose, time_limit)
        self.objective = 'regression'
        # TODO: add choice to the set of estimators
        self.hparams = hp.choice('regressor', [
            {'model': ExtraTreesRegressor,
             'param': extra_trees_regressor_params
             },
            {'model': Ridge,
             'param': ridge_params
             },
            {'model': RandomForestRegressor,
             'param': random_forest_regressor_params
             },
            {'model': LGBMRegressor,
             'param': lgbm_regressor_params
             },
            {'model': AdaBoostRegressor,
             'param': adaboost_regressor_params
             }
            ])

    def get_skf(self, folds):
        return 'neg_mean_squared_error', KFold(n_splits=folds, shuffle=True, random_state=1001)

