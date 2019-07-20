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


class TabularRegressor(TabularEstimator):
    """TabularRegressor class.
    It is used for tabular data regression with lightgbm regressor.
    """
    def __init__(self, path=None, verbose=True, time_limit=None):
        super().__init__(path, verbose, time_limit)
        self.objective = 'regression'

    def get_skf(self, folds):
        return 'neg_mean_squared_error', KFold(n_splits=folds, shuffle=True, random_state=1001)

class LgbmRegressor(TabularRegressor, LGBMMixIn):
    def init_model(self, y):
        self.estimator = LGBMRegressor(silent=False,
                                  verbose=-1,
                                  n_jobs=1,
                                  objective=self.objective)