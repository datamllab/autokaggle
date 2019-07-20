from sklearn.base import BaseEstimator
from tabular_preprocessor import TabularPreprocessor
from tabular_supervised import AutoKaggle, TabularEstimator, LGBMMixIn, SklearnMixIn
from utils import rand_temp_folder_generator, ensure_dir, write_json, read_json
from abc import abstractmethod
import numpy as np
import os
import random
import json

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from joblib import dump, load

class TabularClassifier(TabularEstimator):
    """TabularClassifier class.
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
    
class SVMClassifier(TabularClassifier, SklearnMixIn):
    _default_hyperparams = "svm_hp.json"
        
    def init_model(self, y):
        n_classes = len(set(y))
        self.objective = 'binary' if n_classes == 2 else 'multiclass'
        self.estimator = LinearSVC()
    
class LgbmClassifier(TabularClassifier, LGBMMixIn):
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