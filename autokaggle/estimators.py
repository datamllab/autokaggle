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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, make_scorer
from joblib import dump, load

from autokaggle.utils import rand_temp_folder_generator, ensure_dir, write_json, read_json
import hyperopt
from hyperopt import tpe, hp, fmin, space_eval, Trials, STATUS_OK

knn_classifier_params = {'n_neighbors': hp.choice('n_neighbors', range(2, 20)),
                       'algorithm': hp.choice('algorithm', ['ball_tree', 'kd_tree', 'brute']),
                       'leaf_size': hp.choice('leaf_size', range(5, 50)),
                       'metric': hp.choice('metric', ["euclidean", "manhattan",
                                                      "chebyshev", "minkowski"
                                                      ]),
                       'p': hp.choice('p', range(1, 4)),
                       }
svc_params = {'C': hp.lognormal('C', 0, 1),
                       'kernel': hp.choice('kernel', ['rbf', 'poly', 'linear', 'sigmoid']),
                       'degree': hp.choice('degree', range(1, 6)),
                       'gamma': hp.uniform('gamma', 0.001, 10000),
                       'max_iter': 50000,
                       }

random_forest_classifier_params = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
                       'max_features': hp.uniform('max_features', 0, 1.0),
                       'n_estimators': hp.choice('rf_n_estimators', range(50, 200)),
                       'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10))}

lgbm_classifier_params = {'boosting_type': 'gbdt',
                       'min_split_gain': 0.1,
                       'subsample': 0.8,
                       'num_leaves': 80,
                       'colsample_bytree': hp.uniform('colsample_bytree', 0.4, 0.8),
                       'min_child_weight': hp.choice('min_child_weight', range(1, 100)),
                       'max_depth': hp.choice('max_depth', range(5, 10)),
                       'n_estimators': hp.choice('n_estimators', range(50, 200)),
                       'learning_rate': hp.lognormal('learning_rate', 0, 1),
                       }

adaboost_classifier_params = {'algorithm': hp.choice('algorithm_adaboost', ['SAMME.R', 'SAMME']),
                       'n_estimators': hp.choice('n_estimators_adaboost', range(50, 200)),
                       'learning_rate': hp.lognormal('learning_rate_adaboost', 0, 1),
                       }


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
        self.ensemble_models = True
    
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

    def search(self, x, y, search_iter=4, folds=3):
        grid_train_x, grid_train_y = self.subsample(x, y, sample_percent=0.1)
        score_metric, skf = self.get_skf(folds)

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
            {'model': AdaBoostClassifier,
             'param': adaboost_classifier_params
             }
        ])

        def objective_func(args):
            clf = args['model'](**args['param'])
            loss = cross_val_score(clf, grid_train_x, grid_train_y, scoring=score_metric, cv=skf).mean()
            print("CV Score:", loss)
            print("\n=================")
            return {'loss': 1 - loss, 'status': STATUS_OK, 'space': args}

        trials = Trials()
        opt = space_eval(self.hparams, fmin(objective_func, self.hparams, algo=hyperopt.rand.suggest, trials=trials,
                                            max_evals=search_iter))
        if self.ensemble_models:
            best_trials = sorted(trials.results, key=lambda k: k['loss'], reverse=False)
            estimator_list = []
            for i in range(2):
                model_params = best_trials[i]['space']
                est = model_params['model'](**model_params['param'])
                estimator_list.append(est)
            self.best_estimator_ = Ensembler(x, y, estimator_list)
        else:
            self.best_estimator_ = opt['model'](**opt['param'])

        return opt
            
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