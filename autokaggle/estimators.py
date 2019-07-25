from sklearn.base import BaseEstimator
from abc import abstractmethod
import numpy as np
import os
import random
import json

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from joblib import dump, load

from autokaggle.utils import rand_temp_folder_generator, ensure_dir, write_json, read_json

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
        self.clf = None
        self.estimator = None
    
    def fit(self, x, y):
        self.init_model(y)
        self.search(x, y)
        self.clf.fit(x, y)
        self.save_model()
    
    def predict(self, x, y=None):
        y = self.clf.predict(x, )
        return y
    
    def search(self, x, y, search_iter=40, folds=3):
        # Set small sample for hyper-param search
        if x.shape[0] > 600:
            grid_train_percentage = max(600.0 / x.shape[0], 0.1)
        else:
            grid_train_percentage = 1
        grid_n = int(x.shape[0] * grid_train_percentage)
        idx = random.sample(list(range(x.shape[0])), grid_n)
        grid_train_x, grid_train_y = x[idx, :], y[idx]
        
        if type(self.hparams) != list:
            self.hparams = [self.hparams]
            
        best_params = {}
        for idx, search_space in enumerate(self.hparams):
            best_params.update(search_space)
            if self.verbose:
                print("Step: {}".format(idx+1))
                print("Search space:")
                print(best_params)
                score_metric, skf = self.get_skf(folds)
            random_search = RandomizedSearchCV(self.estimator, param_distributions=best_params, n_iter=search_iter,
                                       scoring=score_metric,
                                       n_jobs=1, cv=skf, verbose=0, random_state=1001, iid=False)
            random_search.fit(grid_train_x, grid_train_y)
            best_params = random_search.best_params_
            for key, value in best_params.items():
                best_params[key] = [value]

        self.clf = random_search.best_estimator_

        return random_search.best_params_
            
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
        self.clf.booster_.save_model(self.save_filename)
    
    def get_feature_importance(self):
        if self.estimator:
            print('Feature Importance:')
            print(self.clf.feature_importances_)
            
            
class SklearnMixIn:
    
    def save_model(self):
        dump(self.clf, self.save_filename)
        
    def load_model(self):
        self.clf = load(self.save_filename)

        
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