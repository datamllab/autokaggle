from sklearn.base import BaseEstimator
from autokaggle.utils import rand_temp_folder_generator, ensure_dir
import hyperopt
from hyperopt import hp
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    RandomForestRegressor, AdaBoostRegressor, \
    ExtraTreesRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
import numpy as np


class Config:
    """ Configuration for various autoML components.

        Defines the common configuration of different auto ML components. It is
        shared between AutoKaggle, AutoPipe, Preprocessor and Ensembling class.

        # Arguments
            path: String. OS path for storing temporary model parameters.
            verbose: Bool. Defines the verbosity of the logging.
            time_limit: Int. Time budget for performing search and fit pipeline.
            use_ensembling: Bool. Defines whether to use an ensemble of models
            num_estimators_ensemble: Int. Maximum number of estimators to be used
            in an ensemble
            ensemble_strategy: String. Strategy to ensemble models
            ensemble_method: String. Aggregation method if ensemble_strategy is
            set to ranked_ensembling
            random_ensemble: Bool. Whether the ensembling estimators are picked
            randomly.
            diverse_ensemble: Bool. Whether estimators from different families are
            picked.
            ensembling_search_iter: Int. Search iterations for ensembling
            hyper-parameter search
            search_algo: String. Search strategy for hyper-parameter search.
            search_iter: Int. Number of iterations used for hyper-parameter search.
            cv_folds: Int. Number of Cross Validation folds.
            subsample_ratio: Percent of subsample used for for hyper-parameter
            search.
            data_info: list(String). Lists the datatypes of each feature column.
            stack_probabilities: Bool. Whether to use class probabilities in
            ensembling.
            upsample_classes: Bool. Whether to upsample less represented classes
            num_p_hparams: Int. Number of preprocessor search spaces.
    """

    def __init__(self, path=None, verbose=True, time_limit=None, use_ensembling=True,
                 num_estimators_ensemble=50,
                 ensemble_strategy='stacking', ensemble_method='max_voting',
                 search_iter=500, cv_folds=3,
                 subsample_ratio=0.1, random_ensemble=False, diverse_ensemble=True,
                 stack_probabilities=False,
                 data_info=None, upsample_classes=False, ensembling_search_iter=10,
                 search_algo='random',
                 num_p_hparams=10):
        self.verbose = verbose
        self.path = path if path is not None else rand_temp_folder_generator()
        ensure_dir(self.path)
        if self.verbose:
            print('Path:', self.path)
        self.time_limit = time_limit
        self.objective = None
        self.use_ensembling = use_ensembling
        self.hparams = None
        self.num_estimators_ensemble = num_estimators_ensemble
        self.ensemble_strategy = ensemble_strategy
        self.ensemble_method = ensemble_method
        self.random_ensemble = random_ensemble
        self.search_iter = search_iter
        self.cv_folds = cv_folds
        self.subsample_ratio = subsample_ratio
        self.resampling_strategy = 'auto'
        self.random_state = 1001
        self.classification_models = ['knn', 'svm', 'lgbm', 'random_forest',
                                      'adaboost']
        # self.classification_models = ['knn', 'lgbm', 'random_forest',]
        self.regression_models = ['extratree', 'ridge', 'lgbm', 'random_forest',
                                  'adaboost', 'catboost']
        self.diverse_ensemble = diverse_ensemble
        self.stack_probabilities = stack_probabilities
        self.data_info = data_info
        self.upsample_classes = upsample_classes
        self.ensembling_search_iter = ensembling_search_iter
        self.search_algo = hyperopt.rand.suggest if search_algo == 'random' else \
            hyperopt.tpe.suggest
        self.num_p_hparams = num_p_hparams

    def update(self, options):
        for k, v in options.items():
            if hasattr(self, k):
                setattr(self, k, v)


KNN_CLASSIFIER_PARAMS = {
    'n_neighbors': hp.choice('n_neighbors_knn', [1, 2, 4, 8, 16, 32, 64, 100]),
    'weights': hp.choice('weight_knn', ['uniform', 'distance']),
    'metric': hp.choice('metric_knn',
                        ["euclidean", "manhattan", "chebyshev", "minkowski"]),
    'p': hp.choice('p_knn', range(1, 3)),
}

SVM_CLASSIFIER_PARAMS = {
    'C': hp.loguniform('C_svm', np.log(0.03125), np.log(32768)),
    'kernel': hp.choice('kernel_svm', ['rbf', 'poly', 'sigmoid']),
    'degree': hp.choice('degree_svm', range(2, 6)),
    'gamma': hp.loguniform('gamma_svm', np.log(3e-5), np.log(8)),
    'max_iter': 50000,
}

RANDOM_FOREST_CLASSIFIER_PARAMS = {
    'criterion': hp.choice('criterion_rf', ['entropy', 'gini']),
    'max_features': hp.uniform('max_features_rf', 0, 1.0),
    'n_estimators': hp.choice('n_estimators_rf', [100, 50]),
    'min_samples_leaf': hp.choice('min_samples_leaf_rf', range(1, 20)),
    'min_samples_split': hp.choice('min_samples_split_rf', range(2, 20)),
}

LGBM_CLASSIFIER_PARAMS = {
    'boosting_type': 'gbdt',
    'min_split_gain': 0.1,
    'subsample': 0.8,
    'num_leaves': 80,
    'colsample_bytree': hp.uniform('colsample_bytree_lgbm', 0.4, 0.8),
    'min_child_weight': hp.choice('min_child_weight_lgbm', range(1, 100)),
    'max_depth': hp.choice('max_depth_lgbm', range(5, 10)),
    'n_estimators': hp.choice('n_estimators_lgbm', range(50, 200)),
    'learning_rate': hp.loguniform('learning_rate_lgbm', low=np.log(1e-2),
                                   high=np.log(2)),
}

ADABOOST_CLASSIFIER_PARAMS = {
    'algorithm': hp.choice('algorithm_adaboost', ['SAMME.R', 'SAMME']),
    'n_estimators': hp.choice('n_estimators_adaboost', range(50, 500)),
    'learning_rate': hp.loguniform('learning_rate_adaboost', low=np.log(1e-2),
                                   high=np.log(2)),
}

CATBOOST_CLASSIFIER_PARAMS = {
    'iterations': hp.choice('iterations_catboost', [5, 10]),
    'depth': hp.choice('depth_catboost', range(4, 11)),
    'learning_rate': hp.loguniform('learning_rate_catboost', low=np.log(1e-3),
                                   high=np.log(1)),
    'loss_function': hp.choice('loss_function_catboost',
                               ['Logloss', 'CrossEntropy']),
    'verbose': True,
    'leaf_estimation_iterations': 10,
    'l2_leaf_reg': hp.choice('l2_leaf_reg_catboost', np.logspace(-20, -19, 3))
}

EXTRA_TREES_REGRESSOR_PARAMS = {
    'n_estimators': hp.choice('n_estimators_extra_trees', [50, 100, 200]),
    'criterion': hp.choice('criterion_extra_trees', ['mse', 'friedman_mse', 'mae']),
    'max_features': hp.uniform('max_features_extra_trees', 0, 1.0),
    'min_samples_leaf': hp.choice('min_samples_leaf_extra_trees', range(1, 20)),
    'min_samples_split': hp.choice('min_samples_split_extra_trees', range(2, 20)),
    'min_impurity_decrease': 0.0,
    'bootstrap': hp.choice('bootstrap_extra_trees', [True, False]),
}

RIDGE_REGRESSOR_PARAMS = {
    'fit_intercept': True,
    'tol': hp.loguniform('tol_ridge', 1e-5, 1e-1),
    'alpha': hp.loguniform('alpha_ridge', np.log(1e-5), np.log(10))
}

RANDOM_FOREST_REGRESSOR_PARAMS = {
    'criterion': hp.choice('criterion_rf', ['mse', 'friedman_mse', 'mae']),
    'max_features': hp.uniform('max_features_rf', 0.1, 1.0),
    'n_estimators': hp.choice('n_estimators_rf', [50, 100, 200]),
    'min_samples_leaf': hp.choice('min_samples_leaf_rf', range(1, 10)),
    'min_samples_split': hp.choice('min_samples_split_rf', range(2, 10)),
    'bootstrap': hp.choice('bootstrap_rf', [True, False]),
}

LGBM_REGRESSOR_PARAMS = {
    'boosting_type': 'gbdt',
    'min_split_gain': 0.1,
    'subsample': 0.8,
    'num_leaves': 80,
    'colsample_bytree': hp.uniform('colsample_bytree_lgbm', 0.4, 0.8),
    'min_child_weight': hp.choice('min_child_weight_lgbm', range(1, 100)),
    'max_depth': hp.choice('max_depth_lgbm', range(5, 10)),
    'n_estimators': hp.choice('n_estimators_lgbm', range(50, 200)),
    'learning_rate': hp.loguniform('learning_rate_lgbm', low=np.log(1e-5),
                                   high=np.log(1)),
}

ADABOOST_REGRESSOR_PARAMS = {
    'loss': hp.choice('loss_adaboost', ["linear", "square", "exponential"]),
    'n_estimators': hp.choice('n_estimators_adaboost', range(50, 300)),
    'learning_rate': hp.loguniform('learning_rate_adaboost', low=np.log(1e-2),
                                   high=np.log(2)),
    # 'max_depth': hp.choice('max_depth_adaboost', range(1, 11)),
}

CATBOOST_REGRESSOR_PARAMS = {
    'iterations': 2,
    'depth': hp.choice('depth_catboost', range(4, 10)),
    'learning_rate': 1,
    'loss_function': 'RMSE',
    'verbose': True
}

REGRESSION_HPARAM_SPACE = {
    'extratree': {
        'model': ExtraTreesRegressor,
        'param': EXTRA_TREES_REGRESSOR_PARAMS
    },
    'ridge': {
        'model': Ridge,
        'param': RIDGE_REGRESSOR_PARAMS
    },
    'random_forest': {
        'model': RandomForestRegressor,
        'param': RANDOM_FOREST_REGRESSOR_PARAMS
    },
    'lgbm': {
        'model': LGBMRegressor,
        'param': LGBM_REGRESSOR_PARAMS
    },
    'adaboost': {
        'model': AdaBoostRegressor,
        'param': ADABOOST_REGRESSOR_PARAMS
    },
    'catboost': {
        'model': CatBoostRegressor,
        'param': CATBOOST_REGRESSOR_PARAMS
    }
}

CLASSIFICATION_HPARAM_SPACE = {
    'knn': {
        'model': KNeighborsClassifier,
        'param': KNN_CLASSIFIER_PARAMS
    },
    'svm': {
        'model': SVC,
        'param': SVM_CLASSIFIER_PARAMS
    },
    'random_forest': {
        'model': RandomForestClassifier,
        'param': RANDOM_FOREST_CLASSIFIER_PARAMS
    },
    'lgbm': {
        'model': LGBMClassifier,
        'param': LGBM_CLASSIFIER_PARAMS
    },
    'adaboost': {
        'model': AdaBoostClassifier,
        'param': ADABOOST_CLASSIFIER_PARAMS
    },
    'catboost': {
        'model': CatBoostClassifier,
        'param': CATBOOST_CLASSIFIER_PARAMS
    }
}

CLASSIFICATION_BASE_HPARAM_SPACE = {
    'knn': {
        'model': KNeighborsClassifier,
        'param': {}
    },
    'svm': {
        'model': SVC,
        'param': {}
    },
    'random_forest': {
        'model': RandomForestClassifier,
        'param': {}
    },
    'lgbm': {
        'model': LGBMClassifier,
        'param': {}
    },
    'adaboost': {
        'model': AdaBoostClassifier,
        'param': {}
    },
    'catboost': {
        'model': CatBoostClassifier,
        'param': {}
    }
}

REGRESSION_BASE_HPARAM_SPACE = {
    'extratree': {
        'model': ExtraTreesRegressor,
        'param': {}
    },
    'ridge': {
        'model': Ridge,
        'param': {}
    },
    'random_forest': {
        'model': RandomForestRegressor,
        'param': {}
    },
    'lgbm': {
        'model': LGBMRegressor,
        'param': {}
    },
    'adaboost': {
        'model': AdaBoostRegressor,
        'param': {}
    },
    'catboost': {
        'model': CatBoostRegressor,
        'param': {}
    }
}

REGRESSION_PREP_HPARAM_SPACE = {
    'cat_encoding': hp.choice('cat_enc',
                              ['count', 'target+count', 'target+label', 'label']),
    'scaling': hp.choice('scaling', [True, False]),
    'log_transform': hp.choice('log_transform', [True, False]),
    'power_transform': hp.choice('power_transform', [True, False]),
    'pca': hp.choice('pca', [True, False]),
    'binning': hp.choice('binning', [True, False]),
    'add_time_offset': hp.choice('add_time_offset', [True, False]),
    'add_time_diff': hp.choice('add_time_diff', [True, False]),
    # 'cat_num_strategy': hp.choice('cat_num_strategy', ['mean', 'std', 'max',
    # 'min', None]),
    # 'cat_cat_strategy': hp.choice('cat_cat_strategy', ['count', 'nunique', None]),
    'imputation_strategy': hp.choice('imputation_strategy',
                                     ['most_frequent', 'zero']),
    'pearson_thresh': hp.uniform('pearson_thresh', 0.001, 0.01),
    'feat_importance_thresh': hp.uniform('feat_importance_thresh', 0.001, 0.01)
}

CLASSIFICATION_PREP_HPARAM_SPACE = {
    'cat_encoding': hp.choice('cat_enc',
                              ['target', 'count', 'target+count', 'target+label']),
    'scaling': hp.choice('scaling', [True, False]),
    'log_transform': hp.choice('log_transform', [True, False]),
    'power_transform': hp.choice('power_transform', [True, False]),
    'pca': hp.choice('pca', [True, False]),
    'binning': hp.choice('binning', [True, False]),
    'add_time_offset': hp.choice('add_time_offset', [True, False]),
    'add_time_diff': hp.choice('add_time_diff', [True, False]),
    # 'cat_num_strategy': hp.choice('cat_num_strategy', ['mean', 'std', 'max',
    # 'min', None]),
    # 'cat_cat_strategy': hp.choice('cat_cat_strategy', ['count', 'nunique', None]),
    'imputation_strategy': hp.choice('imputation_strategy',
                                     ['most_frequent', 'zero']),
    'pearson_thresh': hp.uniform('pearson_thresh', 0.001, 0.01),
    'feat_importance_thresh': hp.uniform('feat_importance_thresh', 0.001, 0.01)
}
