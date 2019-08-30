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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, make_scorer
from joblib import dump, load

from autokaggle.utils import rand_temp_folder_generator, ensure_dir, write_json, read_json
import hyperopt
from hyperopt import tpe, hp, fmin, space_eval


class TabularEstimator(BaseEstimator):
    def __init__(self, path=None, verbose=True, time_limit=None):
        """
        Initialization function for tabular supervised learner.
        """
        self.verbose = verbose
        self.path = path
        self.time_limit = time_limit
        self.objective = None
        abs_cwd = os.path.split(os.path.abspath(__file__))[0]
        self.hparams = read_json(abs_cwd + "/hparam_space/" + self._default_hyperparams)
        self.best_estimator_ = None
    
    def fit(self, x, y):
        self.init_model(y)
        self.search(x, y)
        self.best_estimator_.fit(x, y)
        self.save_model()
    
    def predict(self, x, y=None):
        y = self.best_estimator_.predict(x, )
        return y
    
    @staticmethod
    def subsample(x, y, sample_percent):
        # Set small sample for hyper-param search
        if x.shape[0] > 600:
            grid_train_percentage = max(600.0 / x.shape[0], sample_percent)
        else:
            grid_train_percentage = 1
        grid_n = int(x.shape[0] * grid_train_percentage)
        idx = random.sample(list(range(x.shape[0])), grid_n)
        grid_train_x, grid_train_y = x[idx, :], y[idx]
        return grid_train_x, grid_train_y

    def search(self, x, y, search_iter=40, folds=3):
        grid_train_x, grid_train_y = self.subsample(x, y, sample_percent=0.1)
        score_metric, skf = self.get_skf(folds)

        self.hparams = space = hp.choice('classifier', [
            {'model': KNeighborsClassifier,
             'param': {'n_neighbors':
                           hp.choice('n_neighbors', range(3, 11)),
                       'algorithm': hp.choice('algorithm', ['ball_tree', 'kd_tree']),
                       'leaf_size': hp.choice('leaf_size', range(1, 50)),
                       'metric': hp.choice('metric', ["euclidean", "manhattan",
                                                      "chebyshev", "minkowski"
                                                      ])}
             },
            {'model': SVC,
             'param': {'C': hp.lognormal('C', 0, 1),
                       'kernel': hp.choice('kernel', ['rbf', 'poly', 'rbf', 'sigmoid']),
                       'degree': hp.choice('degree', range(1, 15)),
                       'gamma': hp.uniform('gamma', 0.001, 10000)}
             }
        ])

        def objective_func(args):
            clf = args['model'](**args['param'])
            loss = cross_val_score(clf, grid_train_x, grid_train_y, scoring=score_metric, cv=skf).mean()
            print("CV Score:", loss)
            print("\n=================")
            return 1 - loss

        opt = space_eval(self.hparams, fmin(objective_func, self.hparams, algo=hyperopt.rand.suggest,
                                            max_evals=search_iter))
        self.best_estimator_ = opt['model'](**opt['param'])

        return opt

    # def search(self, x, y, search_iter=40, folds=3):
    #     grid_train_x, grid_train_y = self.subsample(x, y, sample_percent=0.1)
    #
    #     if type(self.hparams) != list:
    #         self.hparams = [self.hparams]
    #
    #     best_params = {}
    #     for idx, search_space in enumerate(self.hparams):
    #         best_params.update(search_space)
    #         if self.verbose:
    #             print("Step: {}".format(idx+1))
    #             print("Search space:")
    #             print(best_params)
    #             score_metric, skf = self.get_skf(folds)
    #         random_search = RandomizedSearchCV(self.estimator, param_distributions=best_params, n_iter=search_iter,
    #                                    scoring=score_metric,
    #                                    n_jobs=1, cv=skf, verbose=0, random_state=1001, iid=False)
    #         random_search.fit(grid_train_x, grid_train_y)
    #         best_params = random_search.best_params_
    #         for key, value in best_params.items():
    #             best_params[key] = [value]
    #
    #     self.best_estimator_ = random_search.best_estimator_
    #
    #     return random_search.best_params_
            
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
        return "TabularEstimator model"
    
    
class Classifier(TabularEstimator):
    """Classifier class.
     It is used for tabular data classification with lightgbm classifier.
    """ 
    def __init__(self, path=None, verbose=True, time_limit=None):
        super().__init__(path, verbose, time_limit)
        self.objective = 'classification'

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
    It is used for tabular data regression with lightgbm regressor.
    """
    def __init__(self, path=None, verbose=True, time_limit=None):
        super().__init__(path, verbose, time_limit)
        self.objective = 'regression'

    def get_skf(self, folds):
        return 'neg_mean_squared_error', KFold(n_splits=folds, shuffle=True, random_state=1001)
    
    
class LGBMMixIn:
    _default_hyperparams = "lgbm_hp.json"
    
    def save_model(self):
        self.best_estimator_.booster_.save_model(self.save_filename)
    
    def get_feature_importance(self):
        if self.best_estimator_:
            print('Feature Importance:')
            print(self.best_estimator_.feature_importances_)
            
            
class SklearnMixIn:
    
    def save_model(self):
        dump(self.best_estimator_, self.save_filename)
        
    def load_model(self):
        self.best_estimator_ = load(self.save_filename)

        
class SVMClassifier(Classifier, SklearnMixIn):
    _default_hyperparams = "svm_hp.json"
        
    def init_model(self, y):
        n_classes = len(set(y))
        self.objective = 'binary' if n_classes == 2 else 'multiclass'
        self.estimator = SVC()

        
class RFClassifier(Classifier, SklearnMixIn):
    _default_hyperparams = "rf_hp.json"
        
    def init_model(self, y):
        n_classes = len(set(y))
        self.objective = 'binary' if n_classes == 2 else 'multiclass'
        self.estimator = RandomForestClassifier()
        
class LgbmClassifier(Classifier, LGBMMixIn):
    def init_model(self, y):
        n_classes = len(set(y))
        if n_classes == 2:
            self.objective = 'binary'
            self.estimator = LGBMClassifier(silent=False,
                                       verbose=-1,
                                       n_jobs=1,
                                       objective=self.objective)
        else:
            self.objective = 'multiclass'
            self.estimator = LGBMClassifier(silent=False,
                                       verbose=-1,
                                       n_jobs=1,
                                       num_class=n_classes,
                                       objective=self.objective)

            
class LgbmRegressor(Regressor, LGBMMixIn):
    def init_model(self, y):
        self.estimator = LGBMRegressor(silent=False,
                                  verbose=-1,
                                  n_jobs=1,
                                  objective=self.objective)